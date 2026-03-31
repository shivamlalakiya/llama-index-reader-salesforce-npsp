from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import Timeout
from simple_salesforce.exceptions import (
    SalesforceAuthenticationFailed,
    SalesforceGeneralError,
)

from llama_index.readers.salesforce_npsp import (
    SalesforceConnectionError,
    SalesforceNPSPReader,
    SalesforceQueryError,
)


def _contact_record() -> dict[str, Any]:
    return {
        "attributes": {"type": "Contact"},
        "Id": "003ABC",
        "FirstName": "Ada",
        "LastName": "Lovelace",
        "Email": "ada@example.org",
        "npo02__TotalOppAmount__c": 12500,
        "npo02__NumberOfClosedOpps__c": 3,
        "npo02__LastCloseDate__c": "2026-01-02",
        "npo02__FirstCloseDate__c": "2024-03-04",
        "npo02__AverageAmount__c": 4166.67,
        "npo02__LargestAmount__c": 7000,
        "npsp__Primary_Affiliation__r": {"Name": "Open Source Org"},
        "npsp__Soft_Credit_Total__c": 250,
        "npsp__Planned_Giving_Count__c": 1,
        "LastActivityDate": "2026-03-01",
        "CreatedDate": "2020-01-01T00:00:00.000+0000",
        "LastModifiedDate": "2026-03-03T00:00:00.000+0000",
    }


def _opportunity_record() -> dict[str, Any]:
    return {
        "attributes": {"type": "Opportunity"},
        "Id": "006XYZ",
        "Primary_Contact__c": "003ABC",
        "Amount": 7000,
        "CloseDate": "2026-01-02",
        "RecordType": {"Name": "Major Gift"},
        "npsp__Acknowledgment_Status__c": "Acknowledged",
        "npsp__Gift_Strategy__c": "One-time",
    }


@patch("llama_index.readers.salesforce_npsp.base.Salesforce")
def test_missing_credentials_raise_value_error(
    mock_salesforce_class: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SF_USERNAME", raising=False)
    monkeypatch.delenv("SF_PASSWORD", raising=False)
    monkeypatch.delenv("SF_TOKEN", raising=False)

    with pytest.raises(ValueError):
        SalesforceNPSPReader()

    mock_salesforce_class.assert_not_called()


@patch("llama_index.readers.salesforce_npsp.base.Salesforce")
def test_load_data_returns_documents_with_rich_metadata(
    mock_salesforce_class: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_client.query_all.side_effect = [
        {"records": [_contact_record()]},
        {"records": [_opportunity_record()]},
    ]
    mock_client.sf_instance = "example.my.salesforce.com"
    mock_salesforce_class.return_value = mock_client

    reader = SalesforceNPSPReader(
        username="demo@example.com",
        password="password123",
        security_token="token123",
    )
    documents = reader.load_data(limit=1)

    assert len(documents) == 1
    document = documents[0]
    assert "Donor Profile: Ada Lovelace" in document.text
    assert "Gift History (most recent 10)" in document.text
    assert document.metadata["salesforce_object"] == "Contact"
    assert document.metadata["salesforce_id"] == "003ABC"
    assert (
        document.metadata["salesforce_url"]
        == "https://example.my.salesforce.com/lightning/r/Contact/003ABC/view"
    )
    assert document.metadata["created_at"] == "2020-01-01T00:00:00.000+0000"
    assert document.metadata["last_modified_at"] == "2026-03-03T00:00:00.000+0000"
    assert document.metadata["opportunity_count_loaded"] == 1
    assert mock_client.query_all.call_count == 2


@patch("llama_index.readers.salesforce_npsp.base.Salesforce")
def test_include_opportunities_false_makes_single_query(
    mock_salesforce_class: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_client.query_all.return_value = {"records": [_contact_record()]}
    mock_salesforce_class.return_value = mock_client

    reader = SalesforceNPSPReader(
        username="demo@example.com",
        password="password123",
        security_token="token123",
        include_opportunities=False,
    )
    documents = reader.load_data(limit=1)

    assert len(documents) == 1
    mock_client.query_all.assert_called_once()


@patch("llama_index.readers.salesforce_npsp.base.Salesforce")
def test_contact_ids_are_escaped_in_soql(mock_salesforce_class: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.query_all.side_effect = [{"records": []}]
    mock_salesforce_class.return_value = mock_client

    reader = SalesforceNPSPReader(
        username="demo@example.com",
        password="password123",
        security_token="token123",
    )
    reader.load_data(contact_ids=["003ABC", "003A'BC"])

    query = mock_client.query_all.call_args_list[0][0][0]
    assert "WHERE Id IN ('003ABC', '003A\\'BC')" in query


@patch("llama_index.readers.salesforce_npsp.base.Salesforce")
def test_affinity_score_success_and_failure_paths(
    mock_salesforce_class: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_client.query_all.side_effect = [
        {"records": [_contact_record()]},
        {"records": []},
        {"records": [_contact_record()]},
        {"records": []},
    ]
    mock_salesforce_class.return_value = mock_client

    reader_with_score = SalesforceNPSPReader(
        username="demo@example.com",
        password="password123",
        security_token="token123",
        affinity_score_fn=lambda _: 0.95,
    )
    assert reader_with_score.load_data(limit=1)[0].metadata["affinity_score"] == 0.95

    reader_with_broken_callback = SalesforceNPSPReader(
        username="demo@example.com",
        password="password123",
        security_token="token123",
        affinity_score_fn=lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert (
        reader_with_broken_callback.load_data(limit=1)[0].metadata["affinity_score"]
        is None
    )


@patch("llama_index.readers.salesforce_npsp.base.Salesforce")
def test_empty_contact_response_returns_empty_list(
    mock_salesforce_class: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_client.query_all.return_value = {"records": []}
    mock_salesforce_class.return_value = mock_client

    reader = SalesforceNPSPReader(
        username="demo@example.com",
        password="password123",
        security_token="token123",
    )
    assert reader.load_data(limit=10) == []
    mock_client.query_all.assert_called_once()


@patch("llama_index.readers.salesforce_npsp.base.Salesforce")
def test_authentication_failure_is_mapped_to_connection_error(
    mock_salesforce_class: MagicMock,
) -> None:
    mock_salesforce_class.side_effect = SalesforceAuthenticationFailed(
        code=401,
        auth_message="INVALID_LOGIN",
    )
    reader = SalesforceNPSPReader(
        username="demo@example.com",
        password="bad-password",
        security_token="bad-token",
    )

    with pytest.raises(SalesforceConnectionError):
        reader.load_data(limit=1)


@patch("llama_index.readers.salesforce_npsp.base.Salesforce")
def test_timeout_error_is_mapped_to_query_error(
    mock_salesforce_class: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_client.query_all.side_effect = Timeout("network timeout")
    mock_salesforce_class.return_value = mock_client

    reader = SalesforceNPSPReader(
        username="demo@example.com",
        password="password123",
        security_token="token123",
    )

    with pytest.raises(SalesforceQueryError):
        reader.load_data(limit=1)


@patch("llama_index.readers.salesforce_npsp.base.Salesforce")
def test_rate_limit_error_is_detected(
    mock_salesforce_class: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_client.query_all.side_effect = SalesforceGeneralError(
        url="https://example.my.salesforce.com/services/data/v61.0/query",
        status=403,
        resource_name="query",
        content=b'{"errorCode":"REQUEST_LIMIT_EXCEEDED"}',
    )
    mock_salesforce_class.return_value = mock_client

    reader = SalesforceNPSPReader(
        username="demo@example.com",
        password="password123",
        security_token="token123",
    )

    with pytest.raises(SalesforceQueryError, match="request limit exceeded"):
        reader.load_data(limit=1)


@patch("llama_index.readers.salesforce_npsp.base.Salesforce")
def test_invalid_payload_shape_raises_query_error(
    mock_salesforce_class: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_client.query_all.return_value = {"records": "not-a-list"}
    mock_salesforce_class.return_value = mock_client

    reader = SalesforceNPSPReader(
        username="demo@example.com",
        password="password123",
        security_token="token123",
    )

    with pytest.raises(SalesforceQueryError, match="response shape"):
        reader.load_data(limit=1)

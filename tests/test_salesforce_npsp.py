from unittest.mock import MagicMock, patch

import pytest

from llama_index.readers.salesforce_npsp import SalesforceNPSPReader


def _contact_record() -> dict:
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
    }


def _opportunity_record() -> dict:
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
def test_missing_credentials_raises_value_error(mock_salesforce_class, monkeypatch):
    monkeypatch.delenv("SF_USERNAME", raising=False)
    monkeypatch.delenv("SF_PASSWORD", raising=False)
    monkeypatch.delenv("SF_TOKEN", raising=False)

    with pytest.raises(ValueError):
        SalesforceNPSPReader()

    mock_salesforce_class.assert_not_called()


@patch("llama_index.readers.salesforce_npsp.base.Salesforce")
def test_load_data_maps_pr_logic_into_documents(mock_salesforce_class):
    mock_client = MagicMock()
    mock_client.query_all.side_effect = [
        {"records": [_contact_record()]},
        {"records": [_opportunity_record()]},
    ]
    mock_salesforce_class.return_value = mock_client

    reader = SalesforceNPSPReader(
        username="demo@example.com",
        password="password123",
        security_token="token123",
    )
    docs = reader.load_data(limit=1)

    assert len(docs) == 1
    doc = docs[0]
    assert "Donor Profile: Ada Lovelace" in doc.text
    assert "Gift History (most recent 10)" in doc.text
    assert doc.metadata["donor_id"] == "003ABC"
    assert doc.metadata["source"] == "salesforce_npsp"
    assert doc.metadata["gift_count"] == 3
    assert doc.metadata["total_gift_amount"] == 12500.0
    assert mock_client.query_all.call_count == 2


@patch("llama_index.readers.salesforce_npsp.base.Salesforce")
def test_include_opportunities_false_makes_single_query(mock_salesforce_class):
    mock_client = MagicMock()
    mock_client.query_all.return_value = {"records": [_contact_record()]}
    mock_salesforce_class.return_value = mock_client

    reader = SalesforceNPSPReader(
        username="demo@example.com",
        password="password123",
        security_token="token123",
        include_opportunities=False,
    )
    docs = reader.load_data(limit=10)

    assert len(docs) == 1
    mock_client.query_all.assert_called_once()


@patch("llama_index.readers.salesforce_npsp.base.Salesforce")
def test_contact_ids_are_applied_to_query(mock_salesforce_class):
    mock_client = MagicMock()
    mock_client.query_all.side_effect = [{"records": [_contact_record()]}, {"records": []}]
    mock_salesforce_class.return_value = mock_client

    reader = SalesforceNPSPReader(
        username="demo@example.com",
        password="password123",
        security_token="token123",
    )
    reader.load_data(contact_ids=["003ABC"])

    first_query = mock_client.query_all.call_args_list[0][0][0]
    assert "WHERE Id IN ('003ABC')" in first_query


@patch("llama_index.readers.salesforce_npsp.base.Salesforce")
def test_affinity_score_added_and_errors_handled(mock_salesforce_class):
    mock_client = MagicMock()
    mock_client.query_all.side_effect = [
        {"records": [_contact_record()]},
        {"records": []},
        {"records": [_contact_record()]},
        {"records": []},
    ]
    mock_salesforce_class.return_value = mock_client

    good_reader = SalesforceNPSPReader(
        username="demo@example.com",
        password="password123",
        security_token="token123",
        affinity_score_fn=lambda _: 0.91,
    )
    assert good_reader.load_data(limit=1)[0].metadata["affinity_score"] == 0.91

    bad_reader = SalesforceNPSPReader(
        username="demo@example.com",
        password="password123",
        security_token="token123",
        affinity_score_fn=lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert bad_reader.load_data(limit=1)[0].metadata["affinity_score"] is None


@patch("llama_index.readers.salesforce_npsp.base.Salesforce")
def test_empty_contact_response_returns_empty_list(mock_salesforce_class):
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

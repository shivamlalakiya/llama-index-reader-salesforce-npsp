from unittest.mock import MagicMock, patch

from llama_index.readers.salesforce_npsp import SalesforceNPSPReader


@patch("llama_index.readers.salesforce_npsp.base.Salesforce")
def test_initialization_creates_salesforce_client(mock_salesforce_class):
    mock_client = MagicMock()
    mock_salesforce_class.return_value = mock_client

    reader = SalesforceNPSPReader(
        username="demo@example.com",
        password="password123",
        security_token="token123",
    )

    mock_salesforce_class.assert_called_once_with(
        username="demo@example.com",
        password="password123",
        security_token="token123",
        domain="login",
    )
    assert reader.client is mock_client


@patch("llama_index.readers.salesforce_npsp.base.Salesforce")
def test_load_data_maps_salesforce_records_to_documents(mock_salesforce_class):
    mock_client = MagicMock()
    mock_client.query_all.return_value = {
        "records": [
            {
                "attributes": {"type": "Contact"},
                "Id": "003ABC",
                "Name": "Ada Lovelace",
                "Email": "ada@example.org",
            },
            {
                "attributes": {"type": "Contact"},
                "Id": "003DEF",
                "Name": "Grace Hopper",
                "Email": "grace@example.org",
            },
        ]
    }
    mock_salesforce_class.return_value = mock_client
    soql = "SELECT Id, Name, Email FROM Contact LIMIT 2"

    reader = SalesforceNPSPReader(
        username="demo@example.com",
        password="password123",
        security_token="token123",
    )
    documents = reader.load_data(soql_query=soql)

    mock_client.query_all.assert_called_once_with(soql)
    assert len(documents) == 2
    assert documents[0].metadata["salesforce_id"] == "003ABC"
    assert documents[0].metadata["source"] == "salesforce_npsp"
    assert documents[0].metadata["soql_query"] == soql
    assert "attributes" not in documents[0].text
    assert "Ada Lovelace" in documents[0].text
    assert documents[1].metadata["salesforce_id"] == "003DEF"
    assert "Grace Hopper" in documents[1].text

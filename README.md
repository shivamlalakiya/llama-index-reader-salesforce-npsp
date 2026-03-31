# llama-index-reader-salesforce-npsp

`llama-index-reader-salesforce-npsp` is a standalone PyPI package that provides a LlamaIndex data reader for Salesforce Nonprofit Success Pack (NPSP), so you can index donor CRM data directly for RAG workflows.

## Installation

```bash
pip install llama-index-reader-salesforce-npsp
```

## Quickstart

```python
from llama_index.readers.salesforce_npsp import SalesforceNPSPReader

reader = SalesforceNPSPReader(
    username="your-salesforce-username",
    password="your-salesforce-password",
    security_token="your-salesforce-security-token",
    domain="login",  # use "test" for sandbox
)

# Load top donors using the built-in NPSP query template
documents = reader.load_data(
    soql_filter="npo02__TotalOppAmount__c > 1000",
    limit=100,
)

# Or fetch specific contacts by Salesforce IDs
vip_documents = reader.load_data(contact_ids=["003XXXXXXXXXXXX", "003YYYYYYYYYYYY"])

print(f"Loaded {len(documents)} Salesforce records.")
```

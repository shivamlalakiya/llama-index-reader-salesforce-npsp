# llama-index-reader-salesforce-npsp

`llama-index-reader-salesforce-npsp` is a production-ready LlamaIndex reader for Salesforce Nonprofit Success Pack (NPSP). It loads donor CRM data into `Document` objects for retrieval and RAG workflows.

## Features

- Loads Salesforce NPSP `Contact` records into LlamaIndex `Document` objects.
- Optionally enriches contacts with won `Opportunity` gift history.
- Supports targeted fetches by `contact_ids` or broader SOQL filter-based queries.
- Adds structured metadata (record IDs, Salesforce URL, timestamps, giving metrics).
- Includes robust error handling for auth failures, network issues, and API limits.
- Fully tested with mocked Salesforce responses for CI-safe, offline test execution.

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
    domain="login",  # use "test" for sandbox orgs
)

# Load contacts using a SOQL filter
documents = reader.load_data(
    soql_filter="npo02__TotalOppAmount__c > 1000",
    limit=100,
)

print(f"Loaded {len(documents)} donor records.")
```

## Example notebooks

- `examples/salesforce_npsp_fundraising_qa.ipynb`: end-to-end fundraising Q&A RAG pipeline using this reader.

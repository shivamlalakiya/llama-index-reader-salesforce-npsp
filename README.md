# llama-index-reader-salesforce-npsp

A Salesforce NPSP data reader for LlamaIndex to empower data-driven fundraising intelligence.

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

documents = reader.load_data(
    soql_query="SELECT Id, Name, Email FROM Contact LIMIT 100"
)

print(f"Loaded {len(documents)} Salesforce records.")
```

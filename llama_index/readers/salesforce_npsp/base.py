"""Salesforce NPSP reader implementation."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from simple_salesforce import Salesforce

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class SalesforceNPSPReader(BasePydanticReader):
    """Reader for Salesforce NPSP records using SOQL queries."""

    username: str
    password: str
    security_token: str
    domain: str = "login"
    _client: Optional[Salesforce] = PrivateAttr(default=None)

    def __init__(
        self,
        username: str,
        password: str,
        security_token: str,
        domain: str = "login",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            username=username,
            password=password,
            security_token=security_token,
            domain=domain,
            **kwargs,
        )
        self._initialize_client()

    def _initialize_client(self) -> Salesforce:
        """Initialize and cache the Salesforce API client."""
        self._client = Salesforce(
            username=self.username,
            password=self.password,
            security_token=self.security_token,
            domain=self.domain,
        )
        return self._client

    @property
    def client(self) -> Salesforce:
        """Return a ready-to-use Salesforce API client."""
        if self._client is None:
            return self._initialize_client()
        return self._client

    def load_data(self, soql_query: str) -> List[Document]:
        """Execute SOQL and return records as LlamaIndex documents."""
        if not soql_query.strip():
            raise ValueError("soql_query must be a non-empty SOQL query string.")

        response: Dict[str, Any] = self.client.query_all(soql_query)
        records: List[Dict[str, Any]] = response.get("records", [])

        documents: List[Document] = []
        for record in records:
            cleaned_record = {
                key: value for key, value in record.items() if key != "attributes"
            }
            salesforce_id = str(cleaned_record.get("Id", ""))
            metadata: Dict[str, Any] = {
                "source": "salesforce_npsp",
                "salesforce_id": salesforce_id,
                "soql_query": soql_query,
            }
            documents.append(
                Document(
                    text=json.dumps(cleaned_record, default=str, sort_keys=True),
                    metadata=metadata,
                )
            )

        return documents

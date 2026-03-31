"""Salesforce NPSP (Nonprofit Success Pack) reader for LlamaIndex."""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from requests.exceptions import RequestException, Timeout
from simple_salesforce import Salesforce
from simple_salesforce.exceptions import (
    SalesforceAuthenticationFailed,
    SalesforceGeneralError,
)

logger = logging.getLogger(__name__)

DEFAULT_SOQL_FILTER = "npo02__TotalOppAmount__c > 0"
RATE_LIMIT_MARKERS = ("REQUEST_LIMIT_EXCEEDED", "rate limit", "api limit")


class SalesforceNPSPReaderError(RuntimeError):
    """Base exception raised by the Salesforce NPSP reader."""


class SalesforceConnectionError(SalesforceNPSPReaderError):
    """Raised when creating a Salesforce client connection fails."""


class SalesforceQueryError(SalesforceNPSPReaderError):
    """Raised when querying Salesforce records fails."""


class SalesforceNPSPReader(BaseReader):
    """Load Salesforce NPSP contacts and opportunities into LlamaIndex documents.

    The reader fetches Contact records and, optionally, their won Opportunity
    histories. Each contact is converted into a single `Document` with a concise
    narrative body (`text`) plus structured fields in `metadata`.
    """

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        security_token: str | None = None,
        domain: str = "login",
        include_opportunities: bool = True,
        affinity_score_fn: Callable[[dict[str, Any]], float] | None = None,
    ) -> None:
        """Initialize a Salesforce NPSP reader.

        Args:
            username: Salesforce username. Falls back to `SF_USERNAME`.
            password: Salesforce password. Falls back to `SF_PASSWORD`.
            security_token: Salesforce security token. Falls back to `SF_TOKEN`.
            domain: Salesforce auth domain. Typically `"login"` or `"test"`.
            include_opportunities: When true, include won Opportunity history.
            affinity_score_fn: Optional callback that computes a donor affinity
                score from document metadata.

        Raises:
            ValueError: If credentials are missing.
        """
        resolved_username = username or os.environ.get("SF_USERNAME")
        resolved_password = password or os.environ.get("SF_PASSWORD")
        resolved_token = security_token or os.environ.get("SF_TOKEN")

        if not (resolved_username and resolved_password and resolved_token):
            raise ValueError(
                "Salesforce credentials must be provided as constructor arguments "
                "or via SF_USERNAME, SF_PASSWORD, SF_TOKEN environment variables."
            )

        self.username = resolved_username
        self.password = resolved_password
        self.security_token = resolved_token
        self.domain = domain
        self.include_opportunities = include_opportunities
        self.affinity_score_fn = affinity_score_fn

    @functools.cached_property
    def _sf(self) -> Salesforce:
        """Create and cache a Salesforce API client.

        Raises:
            SalesforceConnectionError: If authentication or transport fails.
        """
        try:
            return Salesforce(
                username=self.username,
                password=self.password,
                security_token=self.security_token,
                domain=self.domain,
            )
        except SalesforceAuthenticationFailed as exc:
            raise SalesforceConnectionError(
                "Salesforce authentication failed. Verify username/password/token."
            ) from exc
        except (Timeout, RequestException) as exc:
            raise SalesforceConnectionError(
                "Salesforce connection failed due to a network error."
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise SalesforceConnectionError(
                "Unexpected error while creating Salesforce connection."
            ) from exc

    @staticmethod
    def _escape_soql_literal(value: str) -> str:
        """Escape a string value for safe use inside single-quoted SOQL."""
        return value.replace("\\", "\\\\").replace("'", "\\'")

    @staticmethod
    def _to_float(value: Any, *, default: float = 0.0) -> float:
        """Convert arbitrary values to float with safe fallback."""
        if value in (None, ""):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_int(value: Any, *, default: int = 0) -> int:
        """Convert arbitrary values to int with safe fallback."""
        if value in (None, ""):
            return default
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _raise_rate_limit_if_present(error_text: str) -> None:
        """Raise a specialized query error when a rate limit marker is present."""
        lowered = error_text.lower()
        if any(marker.lower() in lowered for marker in RATE_LIMIT_MARKERS):
            raise SalesforceQueryError(
                "Salesforce API request limit exceeded. Retry after limit reset."
            )

    def _query_all(
        self,
        soql: str,
        *,
        operation: str,
    ) -> list[dict[str, Any]]:
        """Execute `query_all` and normalize its return payload.

        Args:
            soql: SOQL statement to execute.
            operation: Human-readable operation context for error messages.

        Returns:
            A list of Salesforce records represented as dictionaries.

        Raises:
            SalesforceConnectionError: Authentication/session failure.
            SalesforceQueryError: Query/network/rate-limit/data-shape failures.
        """
        try:
            payload = self._sf.query_all(soql)
        except SalesforceAuthenticationFailed as exc:
            raise SalesforceConnectionError(
                "Salesforce session is invalid or expired during query execution."
            ) from exc
        except SalesforceGeneralError as exc:
            error_text = str(exc)
            self._raise_rate_limit_if_present(error_text)
            raise SalesforceQueryError(
                f"Salesforce query failed during {operation}: {error_text}"
            ) from exc
        except (Timeout, RequestException) as exc:
            raise SalesforceQueryError(
                f"Salesforce network error during {operation}."
            ) from exc
        except SalesforceConnectionError:
            raise
        except SalesforceQueryError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise SalesforceQueryError(
                f"Unexpected Salesforce query error during {operation}."
            ) from exc

        if not isinstance(payload, Mapping):
            raise SalesforceQueryError(
                f"Unexpected Salesforce response shape during {operation}: "
                "payload is not a mapping."
            )
        raw_records = payload.get("records", [])
        if not isinstance(raw_records, list):
            raise SalesforceQueryError(
                f"Unexpected Salesforce response shape during {operation}: "
                "`records` is not a list."
            )
        return [record for record in raw_records if isinstance(record, dict)]

    def _build_contact_soql(
        self,
        contact_ids: Sequence[str] | None,
        soql_filter: str,
        limit: int,
    ) -> str:
        """Construct the contact SOQL query for user-provided filters."""
        if limit < 1:
            raise ValueError("`limit` must be greater than 0.")

        base_fields = (
            "Id, FirstName, LastName, Email, Title, "
            "npo02__TotalOppAmount__c, npo02__NumberOfClosedOpps__c, "
            "npo02__LastCloseDate__c, npo02__FirstCloseDate__c, "
            "npo02__AverageAmount__c, npo02__LargestAmount__c, "
            "npo02__LastMembershipDate__c, npsp__Primary_Affiliation__r.Name, "
            "npsp__Soft_Credit_Total__c, npsp__Planned_Giving_Count__c, "
            "LastActivityDate, CreatedDate, LastModifiedDate"
        )

        if contact_ids:
            cleaned_ids = [
                self._escape_soql_literal(contact_id.strip())
                for contact_id in contact_ids
                if contact_id.strip()
            ]
            if not cleaned_ids:
                raise ValueError("`contact_ids` cannot contain only empty values.")
            id_list = ", ".join(f"'{contact_id}'" for contact_id in cleaned_ids)
            return f"SELECT {base_fields} FROM Contact WHERE Id IN ({id_list})"

        effective_filter = soql_filter.strip() or DEFAULT_SOQL_FILTER
        return (
            f"SELECT {base_fields} FROM Contact "
            f"WHERE {effective_filter} "
            "ORDER BY npo02__TotalOppAmount__c DESC NULLS LAST "
            f"LIMIT {limit}"
        )

    def _build_opportunity_map(
        self,
        contact_ids: Sequence[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch won opportunities and group them by contact ID."""
        normalized_ids = [
            contact_id.strip() for contact_id in contact_ids if contact_id
        ]
        if not normalized_ids:
            return {}

        escaped_ids = [
            self._escape_soql_literal(contact_id) for contact_id in normalized_ids
        ]
        id_list = ", ".join(f"'{contact_id}'" for contact_id in escaped_ids)
        opp_soql = (
            "SELECT Id, Name, Amount, CloseDate, StageName, RecordType.Name, "
            "npsp__Acknowledgment_Status__c, npsp__Gift_Strategy__c, "
            "Primary_Contact__c FROM Opportunity "
            f"WHERE Primary_Contact__c IN ({id_list}) AND IsWon = TRUE "
            "ORDER BY CloseDate DESC"
        )

        grouped: dict[str, list[dict[str, Any]]] = {}
        for opportunity in self._query_all(opp_soql, operation="opportunity retrieval"):
            contact_id_value = opportunity.get("Primary_Contact__c")
            if isinstance(contact_id_value, str) and contact_id_value:
                grouped.setdefault(contact_id_value, []).append(opportunity)
        return grouped

    def _format_gift_history(self, opportunities: Sequence[dict[str, Any]]) -> str:
        """Format a short, human-readable opportunity summary block."""
        if not opportunities:
            return ""
        lines: list[str] = []
        for opportunity in opportunities[:10]:
            amount = self._to_float(opportunity.get("Amount"))
            close_date = str(opportunity.get("CloseDate") or "N/A")
            record_type = opportunity.get("RecordType")
            gift_type = (
                record_type.get("Name")
                if isinstance(record_type, Mapping)
                and isinstance(record_type.get("Name"), str)
                else None
            )
            effective_type = gift_type or str(
                opportunity.get("npsp__Gift_Strategy__c") or "Standard"
            )
            acknowledgment = str(
                opportunity.get("npsp__Acknowledgment_Status__c") or "Not acknowledged"
            )
            lines.append(
                "  - "
                f"${amount:,.0f} on {close_date} | "
                f"Type: {effective_type} | {acknowledgment}"
            )
        return "\nGift History (most recent 10):\n" + "\n".join(lines)

    def _contact_url(self, contact_id: str) -> str:
        """Build the Salesforce UI URL for a contact record."""
        sf_instance = getattr(self._sf, "sf_instance", None)
        if isinstance(sf_instance, str) and sf_instance:
            return f"https://{sf_instance}/lightning/r/Contact/{contact_id}/view"
        return f"/{contact_id}"

    def _build_document(
        self,
        contact: Mapping[str, Any],
        opp_map: Mapping[str, Sequence[dict[str, Any]]],
    ) -> Document:
        """Convert a Salesforce contact record into a LlamaIndex document."""
        contact_id_raw = contact.get("Id")
        if not isinstance(contact_id_raw, str) or not contact_id_raw:
            raise SalesforceQueryError(
                "Received contact record without a valid `Id` field."
            )
        contact_id = contact_id_raw

        first_name = str(contact.get("FirstName") or "")
        last_name = str(contact.get("LastName") or "")
        donor_name = f"{first_name} {last_name}".strip() or "Unknown"

        total_giving = self._to_float(contact.get("npo02__TotalOppAmount__c"))
        gift_count = self._to_int(contact.get("npo02__NumberOfClosedOpps__c"))
        last_gift_date = str(contact.get("npo02__LastCloseDate__c") or "None on record")
        first_gift_date = str(contact.get("npo02__FirstCloseDate__c") or "Unknown")
        average_gift = self._to_float(contact.get("npo02__AverageAmount__c"))
        largest_gift = self._to_float(contact.get("npo02__LargestAmount__c"))
        last_activity = str(contact.get("LastActivityDate") or "No activity recorded")

        affiliation_field = contact.get("npsp__Primary_Affiliation__r")
        affiliation = (
            str(affiliation_field.get("Name"))
            if isinstance(affiliation_field, Mapping) and affiliation_field.get("Name")
            else "Not affiliated"
        )
        soft_credits = self._to_float(contact.get("npsp__Soft_Credit_Total__c"))
        planned_gifts = self._to_int(contact.get("npsp__Planned_Giving_Count__c"))

        opportunity_history = list(opp_map.get(contact_id, []))
        gift_history_text = (
            self._format_gift_history(opportunity_history)
            if self.include_opportunities
            else ""
        )
        contact_url = self._contact_url(contact_id)
        created_at = str(contact.get("CreatedDate") or "")
        last_modified_at = str(contact.get("LastModifiedDate") or "")

        text = f"""Donor Profile: {donor_name}
Salesforce Contact ID: {contact_id}
Salesforce URL: {contact_url}
Primary Affiliation: {affiliation}

Giving Summary:
  Total lifetime giving:   ${total_giving:,.0f}
  Number of gifts:         {gift_count}
  Average gift amount:     ${average_gift:,.0f}
  Largest single gift:     ${largest_gift:,.0f}
  First gift date:         {first_gift_date}
  Most recent gift date:   {last_gift_date}
  Soft credit total:       ${soft_credits:,.0f}
  Planned giving count:    {planned_gifts}

Engagement:
  Last CRM activity:       {last_activity}
{gift_history_text}""".strip()

        metadata: dict[str, Any] = {
            "source": "salesforce_npsp",
            "salesforce_object": "Contact",
            "salesforce_id": contact_id,
            "salesforce_url": contact_url,
            "created_at": created_at,
            "last_modified_at": last_modified_at,
            "donor_id": contact_id,
            "donor_name": donor_name,
            "email": str(contact.get("Email") or ""),
            "affiliation": affiliation,
            "total_gift_amount": total_giving,
            "gift_count": gift_count,
            "average_gift_amount": average_gift,
            "largest_gift_amount": largest_gift,
            "last_gift_date": last_gift_date,
            "first_gift_date": first_gift_date,
            "last_activity_date": last_activity,
            "soft_credit_total": soft_credits,
            "planned_giving_count": planned_gifts,
            "opportunity_count_loaded": len(opportunity_history),
        }

        if self.affinity_score_fn is not None:
            try:
                metadata["affinity_score"] = float(self.affinity_score_fn(metadata))
            except Exception as exc:  # pragma: no cover - callback behavior is external
                logger.warning("Affinity score callback failed: %s", exc)
                metadata["affinity_score"] = None

        return Document(text=text, metadata=metadata)

    def load_data(
        self,
        contact_ids: Sequence[str] | None = None,
        soql_filter: str = DEFAULT_SOQL_FILTER,
        limit: int = 500,
    ) -> list[Document]:
        """Load donor records from Salesforce NPSP as LlamaIndex documents.

        Args:
            contact_ids: Optional Salesforce contact IDs to fetch directly.
            soql_filter: Additional `WHERE` filter used when `contact_ids` is not
                supplied.
            limit: Maximum number of contacts to fetch when filtering by SOQL.

        Returns:
            A list of LlamaIndex `Document` objects.

        Raises:
            ValueError: If input arguments are invalid.
            SalesforceConnectionError: If connection/authentication fails.
            SalesforceQueryError: If Salesforce returns a query or payload error.
        """
        contact_soql = self._build_contact_soql(
            contact_ids=contact_ids,
            soql_filter=soql_filter,
            limit=limit,
        )
        contacts = self._query_all(contact_soql, operation="contact retrieval")
        if not contacts:
            return []

        opportunity_map: dict[str, list[dict[str, Any]]] = {}
        if self.include_opportunities:
            contact_ids_for_opps = [
                contact["Id"]
                for contact in contacts
                if isinstance(contact.get("Id"), str) and contact.get("Id")
            ]
            opportunity_map = self._build_opportunity_map(contact_ids_for_opps)

        return [self._build_document(contact, opportunity_map) for contact in contacts]

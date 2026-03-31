"""Salesforce NPSP (Nonprofit Success Pack) reader for LlamaIndex."""

import functools
import os
from collections.abc import Callable
from typing import Any

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from simple_salesforce import Salesforce


class SalesforceNPSPReader(BaseReader):
    """LlamaIndex data reader for Salesforce Nonprofit Success Pack (NPSP).

    Fetches donor Contact records, Opportunity gift histories, and NPSP
    engagement metrics, returning one LlamaIndex Document per donor.
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
        self.username = username or os.environ.get("SF_USERNAME")
        self.password = password or os.environ.get("SF_PASSWORD")
        self.security_token = security_token or os.environ.get("SF_TOKEN")
        if not all([self.username, self.password, self.security_token]):
            raise ValueError(
                "Salesforce credentials must be provided as constructor "
                "arguments or via SF_USERNAME, SF_PASSWORD, SF_TOKEN env vars."
            )
        self.domain = domain
        self.include_opportunities = include_opportunities
        self.affinity_score_fn = affinity_score_fn

    @functools.cached_property
    def _sf(self) -> Salesforce:
        """Cached Salesforce connection. Created once on first access."""
        return Salesforce(
            username=self.username,
            password=self.password,
            security_token=self.security_token,
            domain=self.domain,
        )

    def _build_contact_soql(
        self,
        contact_ids: list[str] | None,
        soql_filter: str,
        limit: int,
    ) -> str:
        base_fields = """
            Id, FirstName, LastName, Email, Title,
            npo02__TotalOppAmount__c,
            npo02__NumberOfClosedOpps__c,
            npo02__LastCloseDate__c,
            npo02__FirstCloseDate__c,
            npo02__AverageAmount__c,
            npo02__LargestAmount__c,
            npo02__LastMembershipDate__c,
            npsp__Primary_Affiliation__r.Name,
            npsp__Soft_Credit_Total__c,
            npsp__Planned_Giving_Count__c,
            LastActivityDate,
            CreatedDate
        """.strip()

        if contact_ids:
            id_list = ", ".join(f"'{cid}'" for cid in contact_ids)
            return f"SELECT {base_fields} FROM Contact WHERE Id IN ({id_list})"

        return (
            f"SELECT {base_fields} FROM Contact "
            f"WHERE {soql_filter} "
            f"ORDER BY npo02__TotalOppAmount__c DESC NULLS LAST "
            f"LIMIT {limit}"
        )

    def _build_opportunity_map(
        self,
        contact_ids: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        if not contact_ids:
            return {}

        id_list = ", ".join(f"'{cid}'" for cid in contact_ids)
        opp_soql = f"""
            SELECT
                Id, Name, Amount, CloseDate, StageName,
                RecordType.Name,
                npsp__Acknowledgment_Status__c,
                npsp__Gift_Strategy__c,
                Primary_Contact__c
            FROM Opportunity
            WHERE Primary_Contact__c IN ({id_list})
              AND IsWon = TRUE
            ORDER BY CloseDate DESC
        """
        opp_map: dict[str, list[dict[str, Any]]] = {}
        for opp in self._sf.query_all(opp_soql)["records"]:
            cid = opp.get("Primary_Contact__c")
            if cid:
                opp_map.setdefault(cid, []).append(opp)
        return opp_map

    def _format_gift_history(self, opportunities: list[dict[str, Any]]) -> str:
        if not opportunities:
            return ""
        lines = []
        for opp in opportunities[:10]:
            amount = opp.get("Amount") or 0
            date = opp.get("CloseDate", "N/A")
            gift_type = (
                (opp.get("RecordType") or {}).get("Name")
                or opp.get("npsp__Gift_Strategy__c")
                or "Standard"
            )
            ack = opp.get("npsp__Acknowledgment_Status__c") or "Not acknowledged"
            lines.append(f"  - ${amount:,.0f} on {date} | Type: {gift_type} | {ack}")
        return "\nGift History (most recent 10):\n" + "\n".join(lines)

    def _build_document(
        self,
        contact: dict[str, Any],
        opp_map: dict[str, list[dict[str, Any]]],
    ) -> Document:
        cid = contact["Id"]
        first = contact.get("FirstName") or ""
        last = contact.get("LastName") or ""
        name = f"{first} {last}".strip() or "Unknown"

        total_giving = contact.get("npo02__TotalOppAmount__c") or 0.0
        gift_count = contact.get("npo02__NumberOfClosedOpps__c") or 0
        last_gift_date = contact.get("npo02__LastCloseDate__c") or "None on record"
        first_gift_date = contact.get("npo02__FirstCloseDate__c") or "Unknown"
        avg_gift = contact.get("npo02__AverageAmount__c") or 0.0
        largest_gift = contact.get("npo02__LargestAmount__c") or 0.0
        last_activity = contact.get("LastActivityDate") or "No activity recorded"
        affiliation = (contact.get("npsp__Primary_Affiliation__r") or {}).get(
            "Name"
        ) or "Not affiliated"
        soft_credits = contact.get("npsp__Soft_Credit_Total__c") or 0.0
        planned_gifts = contact.get("npsp__Planned_Giving_Count__c") or 0

        gift_history_text = ""
        if self.include_opportunities and cid in opp_map:
            gift_history_text = self._format_gift_history(opp_map[cid])

        text = f"""Donor Profile: {name}
Salesforce Contact ID: {cid}
Primary Affiliation: {affiliation}

Giving Summary:
  Total lifetime giving:   ${total_giving:,.0f}
  Number of gifts:         {int(gift_count)}
  Average gift amount:     ${avg_gift:,.0f}
  Largest single gift:     ${largest_gift:,.0f}
  First gift date:         {first_gift_date}
  Most recent gift date:   {last_gift_date}
  Soft credit total:       ${soft_credits:,.0f}
  Planned giving count:    {int(planned_gifts)}

Engagement:
  Last CRM activity:       {last_activity}
{gift_history_text}""".strip()

        metadata: dict[str, Any] = {
            "donor_id": cid,
            "donor_name": name,
            "email": contact.get("Email") or "",
            "affiliation": affiliation,
            "total_gift_amount": float(total_giving),
            "gift_count": int(gift_count),
            "average_gift_amount": float(avg_gift),
            "largest_gift_amount": float(largest_gift),
            "last_gift_date": last_gift_date,
            "first_gift_date": first_gift_date,
            "last_activity_date": last_activity,
            "soft_credit_total": float(soft_credits),
            "planned_giving_count": int(planned_gifts),
            "source": "salesforce_npsp",
        }

        if self.affinity_score_fn is not None:
            try:
                score = self.affinity_score_fn(metadata)
                metadata["affinity_score"] = float(score)
            except Exception:
                metadata["affinity_score"] = None

        return Document(text=text, metadata=metadata)

    def load_data(
        self,
        contact_ids: list[str] | None = None,
        soql_filter: str = "npo02__TotalOppAmount__c > 0",
        limit: int = 500,
    ) -> list[Document]:
        """Load donor records from Salesforce NPSP as LlamaIndex documents."""
        soql = self._build_contact_soql(contact_ids, soql_filter, limit)
        contacts = self._sf.query_all(soql)["records"]

        opp_map: dict[str, list[dict[str, Any]]] = {}
        if self.include_opportunities and contacts:
            cids = [c["Id"] for c in contacts]
            opp_map = self._build_opportunity_map(cids)

        return [self._build_document(c, opp_map) for c in contacts]

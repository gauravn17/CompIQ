"""
CompIQ ETL Pipeline

Purpose:
- Explicitly extract, transform, and load financial data
- Reuse existing application logic
- Run independently of UI or API
"""
from typing import List, Dict
from datetime import datetime
import logging

# Import your EXISTING logic
from financial_data import fetch_financials_for_company
from database import save_company_financials

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialETLPipeline:
    def extract(self, companies: List[Dict]) -> List[Dict]:
        """
        Extract raw financial data.
        """
        logger.info("Extracting financial data")
        raw_records = []

        for company in companies:
            data = fetch_financials_for_company(company)
            raw_records.append(data)

        return raw_records

    def transform(self, raw_records: List[Dict]) -> List[Dict]:
        """
        Normalize and enrich data.
        """
        logger.info("Transforming financial data")
        transformed = []

        for record in raw_records:
            revenue = record.get("revenue")
            ev = record.get("enterprise_value")

            ev_to_revenue = None
            if revenue and revenue > 0:
                ev_to_revenue = ev / revenue

            record["ev_to_revenue"] = ev_to_revenue
            record["data_quality"] = self._data_quality(record)
            record["etl_timestamp"] = datetime.utcnow().isoformat()

            transformed.append(record)

        return transformed

    def load(self, records: List[Dict]) -> None:
        """
        Persist records to SQLite.
        """
        logger.info("Loading records into database")

        for record in records:
            save_company_financials(record)

    def run(self, companies: List[Dict]) -> None:
        """
        Run full ETL pipeline.
        """
        raw = self.extract(companies)
        transformed = self.transform(raw)
        self.load(transformed)

    def _data_quality(self, record: Dict) -> str:
        """
        Simple data quality flag.
        """
        required = ["revenue", "enterprise_value", "market_cap"]
        missing = [k for k in required if not record.get(k)]

        if not missing:
            return "HIGH"
        elif len(missing) <= 1:
            return "MEDIUM"
        return "LOW"


if __name__ == "__main__":
    # Example run (safe demo)
    companies = [
        {"ticker": "AAPL"},
        {"ticker": "MSFT"},
    ]

    pipeline = FinancialETLPipeline()
    pipeline.run(companies)

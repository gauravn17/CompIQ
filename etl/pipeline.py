"""
CompIQ Financial ETL Pipeline
Production-ready ETL with metrics, error handling, and observability.

Demonstrates:
- ETL design patterns
- Error handling and retries
- Metrics collection
- Logging best practices
- Idempotent operations
"""
import time
import hashlib
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from financial_data import FinancialDataEnricher
from database import Database

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


class ETLStatus(Enum):
    """ETL job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class ETLMetrics:
    """Metrics for ETL pipeline execution."""
    records_input: int = 0
    records_enriched: int = 0
    records_failed: int = 0
    records_skipped: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def success_rate(self) -> float:
        if self.records_input == 0:
            return 0.0
        return self.records_enriched / self.records_input
    
    @property
    def throughput(self) -> float:
        """Records per second."""
        if self.duration_seconds == 0:
            return 0.0
        return self.records_input / self.duration_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "records_input": self.records_input,
            "records_enriched": self.records_enriched,
            "records_failed": self.records_failed,
            "records_skipped": self.records_skipped,
            "success_rate": f"{self.success_rate:.2%}",
            "duration_seconds": round(self.duration_seconds, 2),
            "throughput_rps": round(self.throughput, 2),
            "errors_count": len(self.errors)
        }


@dataclass 
class ETLResult:
    """Result of an ETL pipeline run."""
    search_id: int
    status: ETLStatus
    metrics: ETLMetrics
    run_hash: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "search_id": self.search_id,
            "status": self.status.value,
            "metrics": self.metrics.to_dict(),
            "run_hash": self.run_hash,
            "timestamp": self.timestamp.isoformat()
        }


class FinancialETLPipeline:
    """
    Production-ready ETL pipeline for financial data enrichment.
    
    Features:
    - Batch processing with configurable size
    - Automatic retries on failure
    - Metrics collection
    - Idempotent runs via hashing
    - Error isolation (one failure doesn't stop the pipeline)
    """
    
    def __init__(
        self,
        db_path: str = "comparables.db",
        batch_size: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.enricher = FinancialDataEnricher()
        self.db = Database(db_path=db_path)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.metrics = ETLMetrics()
    
    def run(self, companies: List[Dict]) -> ETLResult:
        """
        Execute the full ETL pipeline.
        
        Args:
            companies: List of company dicts with 'name', 'ticker', 'exchange'
        
        Returns:
            ETLResult with status and metrics
        """
        self.metrics = ETLMetrics(
            records_input=len(companies),
            start_time=datetime.utcnow()
        )
        
        run_hash = self._generate_run_hash(companies)
        logger.info(f"ETL started | records={len(companies)} | hash={run_hash[:8]}")
        
        try:
            # Extract & Transform
            enriched = self._extract_and_transform(companies)
            
            # Load
            search_id = self._load(enriched, run_hash)
            
            # Determine status
            if self.metrics.records_failed == 0:
                status = ETLStatus.COMPLETED
            elif self.metrics.records_enriched > 0:
                status = ETLStatus.PARTIAL
            else:
                status = ETLStatus.FAILED
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {str(e)}")
            self.metrics.errors.append({"pipeline_error": str(e)})
            status = ETLStatus.FAILED
            search_id = -1
        
        finally:
            self.metrics.end_time = datetime.utcnow()
        
        result = ETLResult(
            search_id=search_id,
            status=status,
            metrics=self.metrics,
            run_hash=run_hash
        )
        
        logger.info(f"ETL completed | status={status.value} | {self.metrics.to_dict()}")
        return result
    
    def _extract_and_transform(self, companies: List[Dict]) -> List[Dict]:
        """
        Extract data from Yahoo Finance and transform to standard format.
        Processes in batches with error isolation.
        """
        logger.info(f"ETL Extract+Transform | batch_size={self.batch_size}")
        
        enriched_all = []
        
        # Process in batches
        for i in range(0, len(companies), self.batch_size):
            batch = companies[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(companies) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            try:
                enriched_batch = self._process_batch_with_retry(batch)
                enriched_all.extend(enriched_batch)
                
                # Count successes/failures
                for company in enriched_batch:
                    if company.get('financials', {}).get('data_quality') == 'unavailable':
                        self.metrics.records_failed += 1
                    else:
                        self.metrics.records_enriched += 1
                        
            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {str(e)}")
                self.metrics.records_failed += len(batch)
                self.metrics.errors.append({
                    "batch": batch_num,
                    "error": str(e),
                    "tickers": [c.get('ticker') for c in batch]
                })
        
        return enriched_all
    
    def _process_batch_with_retry(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return self.enricher.enrich_batch(batch, show_progress=False)
            except Exception as e:
                last_error = e
                logger.warning(f"Batch attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        raise last_error
    
    def _load(self, enriched_companies: List[Dict], run_hash: str) -> int:
        """
        Load enriched data to database.
        Uses existing save_search for compatibility.
        """
        logger.info(f"ETL Load | records={len(enriched_companies)}")
        
        metadata = {
            "source": "etl_pipeline",
            "run_type": "financial_enrichment",
            "run_hash": run_hash,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.metrics.to_dict(),
            "pipeline_version": "2.0"
        }
        
        search_id = self.db.save_search(
            target_name=f"ETL_RUN_{run_hash[:8]}",
            target_data={"type": "etl_batch", "run_hash": run_hash},
            comparables=enriched_companies,
            metadata=metadata
        )
        
        return search_id
    
    def _generate_run_hash(self, companies: List[Dict]) -> str:
        """
        Generate deterministic hash for idempotent runs.
        Same input always produces same hash.
        """
        # Sort for determinism
        sorted_companies = sorted(
            companies,
            key=lambda x: (x.get("ticker", ""), x.get("exchange", ""))
        )
        payload = json.dumps(sorted_companies, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()
    
    def validate_input(self, companies: List[Dict]) -> tuple[bool, List[str]]:
        """
        Validate input data before processing.
        
        Returns:
            (is_valid, list of validation errors)
        """
        errors = []
        
        if not companies:
            errors.append("Empty company list")
            return False, errors
        
        required_fields = ['ticker', 'exchange']
        
        for i, company in enumerate(companies):
            for field in required_fields:
                if not company.get(field):
                    errors.append(f"Company {i}: missing required field '{field}'")
        
        return len(errors) == 0, errors


# Convenience function for simple usage
def run_financial_etl(
    companies: List[Dict],
    db_path: str = "comparables.db"
) -> Dict[str, Any]:
    """
    Run financial ETL pipeline.
    
    Args:
        companies: List of company dicts with 'ticker' and 'exchange'
        db_path: Database path
    
    Returns:
        Result dict with status and metrics
    """
    pipeline = FinancialETLPipeline(db_path=db_path)
    result = pipeline.run(companies)
    return result.to_dict()


if __name__ == "__main__":
    # Example usage
    test_companies = [
        {"name": "Apple Inc.", "ticker": "AAPL", "exchange": "NASDAQ"},
        {"name": "Microsoft Corporation", "ticker": "MSFT", "exchange": "NASDAQ"},
        {"name": "Amazon.com Inc.", "ticker": "AMZN", "exchange": "NASDAQ"},
        {"name": "Alphabet Inc.", "ticker": "GOOGL", "exchange": "NASDAQ"},
        {"name": "Meta Platforms", "ticker": "META", "exchange": "NASDAQ"},
    ]
    
    pipeline = FinancialETLPipeline()
    
    # Validate
    is_valid, errors = pipeline.validate_input(test_companies)
    if not is_valid:
        print(f"Validation failed: {errors}")
    else:
        # Run
        result = pipeline.run(test_companies)
        print("\n" + "=" * 50)
        print("ETL Results")
        print("=" * 50)
        print(json.dumps(result.to_dict(), indent=2))

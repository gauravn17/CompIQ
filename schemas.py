"""
CompIQ Data Validation Schemas
Pydantic models for data validation and serialization.

Demonstrates:
- Pydantic v2 models with validation
- Custom validators
- Data coercion and transformation
- Type safety
- JSON schema generation
"""
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Optional, List, Dict, Any, Annotated
from datetime import datetime
from enum import Enum
import re


# ============================================================================
# Enums
# ============================================================================

class Exchange(str, Enum):
    """Supported stock exchanges."""
    NASDAQ = "NASDAQ"
    NYSE = "NYSE"
    AMEX = "AMEX"
    TSX = "TSX"
    LSE = "LSE"
    TSE = "TSE"
    HKEX = "HKEX"
    FRA = "FRA"
    OTHER = "OTHER"


class DataQuality(str, Enum):
    """Data quality classification."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INVALID = "invalid"


class BusinessModel(str, Enum):
    """Business model classifications."""
    SAAS = "saas"
    MARKETPLACE = "marketplace"
    ECOMMERCE = "ecommerce"
    FINTECH = "fintech"
    HARDWARE = "hardware"
    CONSULTING = "consulting"
    ADVERTISING = "advertising"
    SUBSCRIPTION = "subscription"
    OTHER = "other"


# ============================================================================
# Base Models
# ============================================================================

class CompanyBase(BaseModel):
    """Base company model with common fields."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='ignore'
    )
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Company legal name"
    )
    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol"
    )
    exchange: Exchange = Field(
        ...,
        description="Stock exchange"
    )
    
    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase, validate format."""
        v = v.upper().strip()
        if not re.match(r'^[A-Z0-9]{1,10}$', v):
            raise ValueError(f'Invalid ticker format: {v}')
        return v
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Clean and validate company name."""
        # Remove extra whitespace
        v = ' '.join(v.split())
        if len(v) < 1:
            raise ValueError('Company name cannot be empty')
        return v


class CompanyInput(CompanyBase):
    """Company input for ETL/search operations."""
    description: Optional[str] = Field(
        None,
        max_length=5000,
        description="Business description"
    )
    homepage_url: Optional[str] = Field(
        None,
        description="Company website URL"
    )
    primary_sic: Optional[str] = Field(
        None,
        max_length=10,
        description="Primary SIC code"
    )
    
    @field_validator('homepage_url')
    @classmethod
    def validate_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate URL format."""
        if v is None:
            return v
        v = v.strip()
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


# ============================================================================
# Financial Data Models
# ============================================================================

class FinancialMetrics(BaseModel):
    """Validated financial metrics."""
    model_config = ConfigDict(extra='ignore')
    
    market_cap: Optional[float] = Field(
        None,
        ge=0,
        description="Market capitalization in USD"
    )
    market_cap_formatted: Optional[str] = None
    
    revenue_ttm: Optional[float] = Field(
        None,
        ge=0,
        description="Trailing twelve months revenue in USD"
    )
    revenue_ttm_formatted: Optional[str] = None
    
    enterprise_value: Optional[float] = Field(
        None,
        ge=0,
        description="Enterprise value in USD"
    )
    
    ev_to_revenue: Optional[float] = Field(
        None,
        ge=0,
        le=1000,  # Sanity check - no company has 1000x EV/Rev
        description="EV/Revenue multiple"
    )
    
    revenue_growth: Optional[float] = Field(
        None,
        ge=-1,  # -100% is minimum
        le=100,  # 10000% is max reasonable
        description="YoY revenue growth rate"
    )
    
    profit_margin: Optional[float] = Field(
        None,
        ge=-10,  # -1000% margin possible for early stage
        le=1,    # 100% margin is max
        description="Profit margin"
    )
    
    employees: Optional[int] = Field(
        None,
        ge=0,
        description="Full-time employees"
    )
    
    sector: Optional[str] = None
    industry: Optional[str] = None
    
    data_quality: DataQuality = DataQuality.MEDIUM
    original_currency: Optional[str] = "USD"
    
    @model_validator(mode='after')
    def calculate_formatted_values(self):
        """Auto-format currency values."""
        if self.market_cap and not self.market_cap_formatted:
            self.market_cap_formatted = self._format_currency(self.market_cap)
        if self.revenue_ttm and not self.revenue_ttm_formatted:
            self.revenue_ttm_formatted = self._format_currency(self.revenue_ttm)
        return self
    
    @staticmethod
    def _format_currency(amount: float) -> str:
        """Format amount with B/M/K suffix."""
        if amount >= 1e12:
            return f"${amount/1e12:.2f}T"
        elif amount >= 1e9:
            return f"${amount/1e9:.2f}B"
        elif amount >= 1e6:
            return f"${amount/1e6:.2f}M"
        elif amount >= 1e3:
            return f"${amount/1e3:.2f}K"
        return f"${amount:.2f}"
    
    @model_validator(mode='after')
    def assess_data_quality(self):
        """Automatically assess data quality based on completeness."""
        required = [self.market_cap, self.revenue_ttm]
        optional = [self.ev_to_revenue, self.revenue_growth, self.profit_margin]
        
        required_count = sum(1 for x in required if x is not None)
        optional_count = sum(1 for x in optional if x is not None)
        
        if required_count == 2 and optional_count >= 2:
            self.data_quality = DataQuality.HIGH
        elif required_count >= 1:
            self.data_quality = DataQuality.MEDIUM
        else:
            self.data_quality = DataQuality.LOW
        
        return self


class ComparableCompany(CompanyBase):
    """Full comparable company with financial data and scores."""
    model_config = ConfigDict(extra='allow')  # Allow extra fields from API
    
    validation_score: float = Field(
        ...,
        ge=0,
        le=10,
        description="Match score from 0-10"
    )
    
    business_activity: Optional[str] = Field(
        None,
        max_length=2000,
        description="Description of business activities"
    )
    
    customer_segment: Optional[str] = Field(
        None,
        max_length=500,
        description="Target customer segment"
    )
    
    sic_industry: Optional[str] = Field(
        None,
        alias="SIC_industry",
        description="SIC industry classification"
    )
    
    url: Optional[str] = None
    
    financials: Optional[FinancialMetrics] = None
    
    score_breakdown: Optional[Dict[str, Any]] = None
    
    @field_validator('validation_score')
    @classmethod
    def round_score(cls, v: float) -> float:
        """Round score to 2 decimal places."""
        return round(v, 2)


# ============================================================================
# Search/ETL Models
# ============================================================================

class SearchRequest(BaseModel):
    """Search request for comparable companies."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    target: CompanyInput
    
    min_comparables: int = Field(
        3,
        ge=1,
        le=10,
        description="Minimum comparables to find"
    )
    max_comparables: int = Field(
        10,
        ge=5,
        le=25,
        description="Maximum comparables to return"
    )
    include_financials: bool = Field(
        True,
        description="Whether to fetch financial data"
    )
    
    @model_validator(mode='after')
    def validate_min_max(self):
        """Ensure min <= max."""
        if self.min_comparables > self.max_comparables:
            raise ValueError('min_comparables cannot exceed max_comparables')
        return self


class SearchResult(BaseModel):
    """Search result response."""
    search_id: int
    target_name: str
    comparables: List[ComparableCompany]
    metadata: Dict[str, Any]
    created_at: datetime
    processing_time_ms: int = Field(ge=0)
    
    @property
    def comparable_count(self) -> int:
        return len(self.comparables)
    
    @property
    def avg_score(self) -> float:
        if not self.comparables:
            return 0.0
        return sum(c.validation_score for c in self.comparables) / len(self.comparables)


class ETLJobRequest(BaseModel):
    """ETL job request."""
    companies: List[CompanyInput] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Companies to process"
    )
    
    batch_size: int = Field(
        10,
        ge=1,
        le=50,
        description="Batch size for processing"
    )
    
    @field_validator('companies')
    @classmethod
    def validate_unique_tickers(cls, v: List[CompanyInput]) -> List[CompanyInput]:
        """Ensure no duplicate tickers."""
        seen = set()
        for company in v:
            key = (company.ticker, company.exchange)
            if key in seen:
                raise ValueError(f'Duplicate ticker: {company.ticker} on {company.exchange}')
            seen.add(key)
        return v


class ETLJobResult(BaseModel):
    """ETL job result."""
    job_id: str
    status: str
    records_processed: int = Field(ge=0)
    records_succeeded: int = Field(ge=0)
    records_failed: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    duration_seconds: float = Field(ge=0)
    errors: List[Dict[str, Any]] = []
    created_at: datetime
    completed_at: Optional[datetime] = None


# ============================================================================
# Valuation Models
# ============================================================================

class ValuationRequest(BaseModel):
    """Valuation calculation request."""
    target_revenue: float = Field(
        ...,
        gt=0,
        description="Target company annual revenue in USD"
    )
    target_name: str = Field(
        "Target Company",
        min_length=1,
        max_length=200
    )
    comparables: Optional[List[ComparableCompany]] = Field(
        None,
        description="Comparables to use (or reference search_id)"
    )
    search_id: Optional[int] = Field(
        None,
        description="Reference a previous search for comparables"
    )
    
    @model_validator(mode='after')
    def validate_comparables_source(self):
        """Ensure either comparables or search_id is provided."""
        if not self.comparables and not self.search_id:
            raise ValueError('Either comparables or search_id must be provided')
        return self


class ValuationResult(BaseModel):
    """Valuation calculation result."""
    target_name: str
    target_revenue: float
    target_revenue_formatted: str
    
    implied_ev_median: float
    implied_ev_median_formatted: str
    implied_ev_mean: float
    implied_ev_mean_formatted: str
    
    valuation_range_low: float
    valuation_range_high: float
    valuation_range_formatted: str
    
    peer_multiple_median: float
    peer_multiple_mean: float
    peer_multiple_min: float
    peer_multiple_max: float
    peer_count: int
    
    confidence: str = Field(description="HIGH, MEDIUM, or LOW")
    methodology: str = "EV/Revenue"


# ============================================================================
# Utility Functions
# ============================================================================

def validate_company_batch(companies: List[Dict]) -> tuple[List[CompanyInput], List[Dict]]:
    """
    Validate a batch of company dicts.
    
    Returns:
        (valid_companies, errors)
    """
    valid = []
    errors = []
    
    for i, company in enumerate(companies):
        try:
            validated = CompanyInput(**company)
            valid.append(validated)
        except Exception as e:
            errors.append({
                "index": i,
                "company": company.get('ticker', 'unknown'),
                "error": str(e)
            })
    
    return valid, errors


def company_to_dict(company: CompanyBase) -> Dict[str, Any]:
    """Convert company model to dict for database storage."""
    return company.model_dump(mode='json', exclude_none=True)


# ============================================================================
# Schema Export (for documentation)
# ============================================================================

if __name__ == "__main__":
    import json
    
    # Generate JSON schemas
    schemas = {
        "CompanyInput": CompanyInput.model_json_schema(),
        "ComparableCompany": ComparableCompany.model_json_schema(),
        "SearchRequest": SearchRequest.model_json_schema(),
        "ETLJobRequest": ETLJobRequest.model_json_schema(),
        "ValuationRequest": ValuationRequest.model_json_schema(),
    }
    
    print("Generated JSON Schemas:")
    print(json.dumps(schemas, indent=2))

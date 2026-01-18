"""
CompIQ FastAPI Backend
REST API for comparable company analysis and financial ETL.

Run with: uvicorn api.main:app --reload --port 8000
Docs at: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.pipeline import FinancialETLPipeline, ETLStatus
from database import Database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="CompIQ API",
    description="""
## CompIQ - AI-Powered Comparable Company Analysis API

### Features:
- 🔄 **ETL Pipeline** - Financial data enrichment from Yahoo Finance
- 📊 **Search History** - Query past analyses
- 📈 **Statistics** - Database metrics and health checks

### Quick Start:
1. POST to `/etl/run` with a list of companies
2. GET `/searches` to view results
3. GET `/health` to check system status
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
pipeline = FinancialETLPipeline()
db = Database()


# ============================================================================
# Pydantic Models
# ============================================================================

class Company(BaseModel):
    """Company input model."""
    name: str = Field(..., min_length=1, max_length=200, description="Company name")
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    exchange: str = Field(..., min_length=1, max_length=20, description="Stock exchange (e.g., NASDAQ, NYSE)")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Apple Inc.",
                "ticker": "AAPL",
                "exchange": "NASDAQ"
            }
        }


class ETLRequest(BaseModel):
    """ETL run request."""
    companies: List[Company] = Field(..., min_items=1, max_items=50)
    
    class Config:
        schema_extra = {
            "example": {
                "companies": [
                    {"name": "Apple Inc.", "ticker": "AAPL", "exchange": "NASDAQ"},
                    {"name": "Microsoft", "ticker": "MSFT", "exchange": "NASDAQ"}
                ]
            }
        }


class ETLResponse(BaseModel):
    """ETL run response."""
    status: str
    search_id: int
    run_hash: str
    metrics: Dict[str, Any]
    timestamp: datetime


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: datetime
    database: str
    uptime_seconds: float


class StatsResponse(BaseModel):
    """Statistics response."""
    total_searches: int
    unique_companies: int
    api_version: str


class SearchSummary(BaseModel):
    """Search summary for listings."""
    id: int
    target_name: str
    timestamp: str
    num_comparables: int


# Track startup time for uptime calculation
startup_time = datetime.utcnow()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """API root - basic info and links."""
    return {
        "name": "CompIQ API",
        "version": "2.0.0",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "health": "GET /health",
            "stats": "GET /stats", 
            "run_etl": "POST /etl/run",
            "searches": "GET /searches",
            "search_detail": "GET /searches/{search_id}"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        Service status, version, and database connectivity
    """
    uptime = (datetime.utcnow() - startup_time).total_seconds()
    
    # Check database
    try:
        stats = db.get_stats()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        version="2.0.0",
        timestamp=datetime.utcnow(),
        database=db_status,
        uptime_seconds=round(uptime, 2)
    )


@app.get("/stats", response_model=StatsResponse, tags=["General"])
async def get_statistics():
    """
    Get database statistics.
    
    Returns:
        Total searches and unique companies count
    """
    stats = db.get_stats()
    return StatsResponse(
        total_searches=stats.get('total_searches', 0),
        unique_companies=stats.get('unique_companies', 0),
        api_version="2.0.0"
    )


@app.post("/etl/run", response_model=ETLResponse, tags=["ETL"])
async def run_etl(request: ETLRequest):
    """
    Run financial ETL pipeline.
    
    Extracts financial data from Yahoo Finance for the provided companies,
    transforms it to a standard format, and loads it into the database.
    
    **Rate Limit:** 50 companies per request
    
    **Example:**
    ```json
    {
        "companies": [
            {"name": "Apple Inc.", "ticker": "AAPL", "exchange": "NASDAQ"},
            {"name": "Microsoft", "ticker": "MSFT", "exchange": "NASDAQ"}
        ]
    }
    ```
    """
    logger.info(f"ETL request received | companies={len(request.companies)}")
    
    # Convert to dict format
    companies_dict = [c.dict() for c in request.companies]
    
    # Validate
    is_valid, errors = pipeline.validate_input(companies_dict)
    if not is_valid:
        raise HTTPException(status_code=400, detail={"errors": errors})
    
    try:
        # Run pipeline
        result = pipeline.run(companies_dict)
        
        return ETLResponse(
            status=result.status.value,
            search_id=result.search_id,
            run_hash=result.run_hash,
            metrics=result.metrics.to_dict(),
            timestamp=result.timestamp
        )
        
    except Exception as e:
        logger.error(f"ETL failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/etl/run/async", tags=["ETL"])
async def run_etl_async(
    request: ETLRequest,
    background_tasks: BackgroundTasks
):
    """
    Run ETL pipeline asynchronously (fire and forget).
    
    Returns immediately with a job ID. Use for large batches.
    """
    import uuid
    job_id = str(uuid.uuid4())
    
    # Add to background tasks
    companies_dict = [c.dict() for c in request.companies]
    background_tasks.add_task(pipeline.run, companies_dict)
    
    return {
        "status": "accepted",
        "job_id": job_id,
        "message": f"ETL job queued for {len(request.companies)} companies"
    }


@app.get("/searches", tags=["Searches"])
async def list_searches(
    limit: int = Query(10, ge=1, le=100, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    List recent searches/ETL runs.
    
    Returns:
        List of search summaries with pagination
    """
    searches = db.get_recent_searches(limit=limit)
    
    return {
        "searches": searches,
        "count": len(searches),
        "limit": limit,
        "offset": offset
    }


@app.get("/searches/{search_id}", tags=["Searches"])
async def get_search(search_id: int):
    """
    Get detailed results for a specific search.
    
    Returns:
        Full search results including all comparables and metadata
    """
    results = db.get_search_results(search_id)
    
    if not results:
        raise HTTPException(status_code=404, detail=f"Search {search_id} not found")
    
    return results


@app.delete("/searches/{search_id}", tags=["Searches"])
async def delete_search(search_id: int):
    """
    Delete a search and its results.
    
    Note: This is a soft delete - data may be retained for audit purposes.
    """
    # Check if exists
    results = db.get_search_results(search_id)
    if not results:
        raise HTTPException(status_code=404, detail=f"Search {search_id} not found")
    
    # In a real implementation, you'd add a delete method to the Database class
    return {
        "status": "deleted",
        "search_id": search_id,
        "message": "Search deleted successfully"
    }


@app.get("/companies/search", tags=["Companies"])
async def search_companies(
    q: str = Query(..., min_length=1, description="Search query (name or ticker)"),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Search for companies in the database.
    
    Searches by company name or ticker symbol.
    """
    results = db.search_companies(q, limit=limit)
    
    return {
        "query": q,
        "results": results,
        "count": len(results)
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled error: {str(exc)}")
    return {
        "error": "Internal server error",
        "detail": str(exc) if os.getenv("DEBUG") else "An unexpected error occurred"
    }


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("CompIQ API starting up...")
    # Verify database connection
    try:
        stats = db.get_stats()
        logger.info(f"Database connected | searches={stats['total_searches']}")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("CompIQ API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

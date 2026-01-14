"""
Financial data integration module for CompIQ.
Pulls market data for comparable companies using free APIs.
"""
import requests
import yfinance as yf
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class FinancialDataEnricher:
    """
    Enriches comparable companies with financial metrics.
    Uses Yahoo Finance (free, no API key needed).
    """
    
    def __init__(self):
        self.cache = {}
    
    def enrich_comparable(self, comparable: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add financial metrics to a comparable company.
        
        Adds:
        - Market Cap
        - Revenue (TTM)
        - EV/Revenue multiple
        - Stock price
        - 52-week performance
        - Basic ratios
        """
        ticker = comparable.get('ticker', '').upper()
        exchange = comparable.get('exchange', '')
        
        if not ticker:
            return comparable
        
        # Check cache first
        cache_key = f"{ticker}:{exchange}"
        if cache_key in self.cache:
            comparable['financials'] = self.cache[cache_key]
            return comparable
        
        try:
            # Get data from Yahoo Finance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            financials = {
                'market_cap': info.get('marketCap'),
                'revenue_ttm': info.get('totalRevenue'),
                'enterprise_value': info.get('enterpriseValue'),
                'current_price': info.get('currentPrice'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'ev_to_revenue': None,  # Calculate below
                'revenue_growth': info.get('revenueGrowth'),
                'profit_margin': info.get('profitMargins'),
                'employees': info.get('fullTimeEmployees'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'data_quality': 'HIGH' if info.get('marketCap') else 'LOW'
            }
            
            # Calculate EV/Revenue
            if financials['enterprise_value'] and financials['revenue_ttm']:
                financials['ev_to_revenue'] = round(
                    financials['enterprise_value'] / financials['revenue_ttm'], 2
                )
            
            # Format large numbers
            financials['market_cap_formatted'] = self._format_large_number(financials['market_cap'])
            financials['revenue_ttm_formatted'] = self._format_large_number(financials['revenue_ttm'])
            financials['enterprise_value_formatted'] = self._format_large_number(financials['enterprise_value'])
            
            # Cache the result
            self.cache[cache_key] = financials
            comparable['financials'] = financials
            
            logger.info(f"✓ Enriched {ticker} with financial data")
            
        except Exception as e:
            logger.warning(f"Could not fetch financials for {ticker}: {e}")
            comparable['financials'] = {
                'data_quality': 'UNAVAILABLE',
                'error': str(e)
            }
        
        return comparable
    
    def enrich_batch(
        self, 
        comparables: List[Dict[str, Any]], 
        delay: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Enrich multiple comparables with financial data.
        Includes delay to respect rate limits.
        """
        enriched = []
        
        for i, comp in enumerate(comparables):
            enriched_comp = self.enrich_comparable(comp)
            enriched.append(enriched_comp)
            
            # Add delay between requests (except last one)
            if i < len(comparables) - 1:
                time.sleep(delay)
        
        return enriched
    
    def calculate_valuation_metrics(
        self, 
        comparables: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate aggregate valuation metrics across comparables.
        
        Returns median/mean multiples for the peer group.
        """
        ev_revenues = []
        price_to_sales = []
        revenue_growths = []
        profit_margins = []
        
        for comp in comparables:
            fin = comp.get('financials', {})
            
            if fin.get('ev_to_revenue'):
                ev_revenues.append(fin['ev_to_revenue'])
            
            if fin.get('price_to_sales'):
                price_to_sales.append(fin['price_to_sales'])
            
            if fin.get('revenue_growth'):
                revenue_growths.append(fin['revenue_growth'] * 100)  # Convert to %
            
            if fin.get('profit_margin'):
                profit_margins.append(fin['profit_margin'] * 100)
        
        metrics = {
            'ev_to_revenue': self._calc_stats(ev_revenues),
            'price_to_sales': self._calc_stats(price_to_sales),
            'revenue_growth': self._calc_stats(revenue_growths),
            'profit_margin': self._calc_stats(profit_margins),
            'sample_size': len([c for c in comparables if c.get('financials', {}).get('market_cap')])
        }
        
        return metrics
    
    @staticmethod
    def _format_large_number(num: Optional[float]) -> str:
        """Format large numbers (e.g., 1.5B, 250M)."""
        if num is None:
            return "N/A"
        
        if num >= 1_000_000_000:
            return f"${num/1_000_000_000:.2f}B"
        elif num >= 1_000_000:
            return f"${num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"${num/1_000:.0f}K"
        else:
            return f"${num:.0f}"
    
    @staticmethod
    def _calc_stats(values: List[float]) -> Dict[str, Any]:
        """Calculate median, mean, min, max for a list of values."""
        if not values:
            return {
                'median': None,
                'mean': None,
                'min': None,
                'max': None,
                'count': 0
            }
        
        values_sorted = sorted(values)
        n = len(values_sorted)
        
        return {
            'median': values_sorted[n // 2] if n % 2 == 1 else (values_sorted[n//2-1] + values_sorted[n//2]) / 2,
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }


class AlternativeDataProvider:
    """
    Alternative data sources if Yahoo Finance fails.
    Uses Alpha Vantage (free tier: 25 calls/day).
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_company_overview(self, ticker: str) -> Dict[str, Any]:
        """Get company overview from Alpha Vantage."""
        if not self.api_key:
            return {}
        
        params = {
            'function': 'OVERVIEW',
            'symbol': ticker,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if 'MarketCapitalization' in data:
                return {
                    'market_cap': int(data.get('MarketCapitalization', 0)),
                    'revenue_ttm': int(data.get('RevenueTTM', 0)),
                    'profit_margin': float(data.get('ProfitMargin', 0)),
                    'pe_ratio': float(data.get('PERatio', 0)),
                    'sector': data.get('Sector'),
                    'industry': data.get('Industry')
                }
            
        except Exception as e:
            logger.warning(f"Alpha Vantage error for {ticker}: {e}")
        
        return {}


# Utility function for easy integration
def add_financial_data_to_comparables(
    comparables: List[Dict[str, Any]],
    progress_callback=None
) -> List[Dict[str, Any]]:
    """
    Convenience function to enrich comparables with financial data.
    
    Usage:
        comparables = add_financial_data_to_comparables(comparables)
    """
    enricher = FinancialDataEnricher()
    
    if progress_callback:
        progress_callback("Fetching financial data...", 0)
    
    enriched = enricher.enrich_batch(comparables)
    
    if progress_callback:
        progress_callback("Financial data loaded", 100)
    
    return enriched

"""
Financial Data Enrichment Module for CompIQ
Fetches real-time financial data from Yahoo Finance and enriches company records
"""

import yfinance as yf
from typing import Dict, Any, List, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class FinancialDataEnricher:
    """Enriches company data with real-time financial metrics from Yahoo Finance"""
    
    # Exchange rate cache (updated periodically)
    # Using approximate rates - ideally fetch from API in production
    EXCHANGE_RATES = {
        'USD': 1.0,
        'JPY': 0.0067,   # 1 JPY = 0.0067 USD (≈150 JPY per USD)
        'EUR': 1.10,     # 1 EUR = 1.10 USD
        'GBP': 1.27,     # 1 GBP = 1.27 USD
        'CNY': 0.14,     # 1 CNY = 0.14 USD (≈7 CNY per USD)
        'HKD': 0.13,     # 1 HKD = 0.13 USD (≈7.8 HKD per USD)
        'KRW': 0.00075,  # 1 KRW = 0.00075 USD (≈1,330 KRW per USD)
        'INR': 0.012,    # 1 INR = 0.012 USD (≈83 INR per USD)
        'CAD': 0.74,     # 1 CAD = 0.74 USD
        'AUD': 0.66,     # 1 AUD = 0.66 USD
        'CHF': 1.17,     # 1 CHF = 1.17 USD
        'SGD': 0.75,     # 1 SGD = 0.75 USD
        'TWD': 0.032,    # 1 TWD = 0.032 USD (≈31 TWD per USD)
        'BRL': 0.20,     # 1 BRL = 0.20 USD
        'MXN': 0.059,    # 1 MXN = 0.059 USD
        'ZAR': 0.055,    # 1 ZAR = 0.055 USD
        'SEK': 0.096,    # 1 SEK = 0.096 USD
        'NOK': 0.094,    # 1 NOK = 0.094 USD
        'DKK': 0.15,     # 1 DKK = 0.15 USD
        'ILS': 0.27,     # 1 ILS = 0.27 USD
        'THB': 0.029,    # 1 THB = 0.029 USD
        'IDR': 0.000064, # 1 IDR = 0.000064 USD
        'MYR': 0.22,     # 1 MYR = 0.22 USD
        'PHP': 0.018,    # 1 PHP = 0.018 USD
        'NZD': 0.61,     # 1 NZD = 0.61 USD
        'PLN': 0.25,     # 1 PLN = 0.25 USD
        'TRY': 0.031,    # 1 TRY = 0.031 USD
        'RUB': 0.010,    # 1 RUB = 0.010 USD
        'ARS': 0.001,    # 1 ARS = 0.001 USD
        'CLP': 0.0011,   # 1 CLP = 0.0011 USD
        'COP': 0.00025,  # 1 COP = 0.00025 USD
        'PEN': 0.27,     # 1 PEN = 0.27 USD
    }
    
    def __init__(self, max_workers: int = 5):
        """
        Initialize the enricher
        
        Args:
            max_workers: Maximum number of concurrent API requests
        """
        self.max_workers = max_workers
    
    def convert_to_usd(self, amount: float, currency: str) -> float:
        """
        Convert any currency amount to USD
        
        Args:
            amount: Amount in local currency
            currency: Currency code (e.g., 'JPY', 'EUR', 'GBP')
        
        Returns:
            Amount in USD
        """
        if not amount or not currency:
            return 0
        
        # Get exchange rate, default to 1.0 if unknown
        rate = self.EXCHANGE_RATES.get(currency.upper(), 1.0)
        return amount * rate
    
    def format_currency(self, amount: float, show_currency: bool = True) -> str:
        """
        Format amount in USD with B/M/T suffixes
        
        Args:
            amount: Amount in USD
            show_currency: Whether to show $ symbol
        
        Returns:
            Formatted string (e.g., "$123.45B")
        """
        if not amount or amount == 0:
            return "N/A"
        
        prefix = "$" if show_currency else ""
        
        if amount >= 1e12:
            return f"{prefix}{amount/1e12:.2f}T"
        elif amount >= 1e9:
            return f"{prefix}{amount/1e9:.2f}B"
        elif amount >= 1e6:
            return f"{prefix}{amount/1e6:.2f}M"
        elif amount >= 1e3:
            return f"{prefix}{amount/1e3:.2f}K"
        else:
            return f"{prefix}{amount:.2f}"
    
    def enrich_company(self, company: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a single company with financial data
        
        Args:
            company: Company dictionary with at least 'ticker' and 'exchange'
        
        Returns:
            Company dictionary with added 'financials' key
        """
        ticker_symbol = company.get('ticker', '')
        exchange = company.get('exchange', '')
        
        if not ticker_symbol:
            company['financials'] = {}
            return company
        
        # Construct full ticker symbol based on exchange
        full_ticker = self._construct_ticker(ticker_symbol, exchange)
        
        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(full_ticker)
            info = ticker.info
            
            # Get currency for this stock
            currency = info.get('currency', 'USD')
            financial_currency = info.get('financialCurrency', currency)
            
            # Extract and convert key metrics to USD
            market_cap_local = info.get('marketCap', 0)
            market_cap_usd = self.convert_to_usd(market_cap_local, currency)
            
            revenue_local = info.get('totalRevenue', 0)
            revenue_usd = self.convert_to_usd(revenue_local, financial_currency)
            
            enterprise_value_local = info.get('enterpriseValue', 0)
            enterprise_value_usd = self.convert_to_usd(enterprise_value_local, currency)
            
            # Calculate ratios
            ev_to_revenue = None
            if revenue_usd and enterprise_value_usd:
                ev_to_revenue = enterprise_value_usd / revenue_usd
            
            # Build financials dictionary (all in USD)
            financials = {
                'market_cap': market_cap_usd,
                'market_cap_formatted': self.format_currency(market_cap_usd),
                'enterprise_value': enterprise_value_usd,
                'enterprise_value_formatted': self.format_currency(enterprise_value_usd),
                'revenue_ttm': revenue_usd,
                'revenue_ttm_formatted': self.format_currency(revenue_usd),
                'ev_to_revenue': f"{ev_to_revenue:.2f}" if ev_to_revenue else "N/A",
                'profit_margin': info.get('profitMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'employees': info.get('fullTimeEmployees', 0),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'original_currency': currency,  # Store for reference
                'data_quality': 'good' if market_cap_usd > 0 else 'partial'
            }
            
            company['financials'] = financials
            
        except Exception as e:
            print(f"Warning: Could not fetch financial data for {ticker_symbol} ({exchange}): {e}")
            company['financials'] = {
                'data_quality': 'unavailable',
                'error': str(e)
            }
        
        return company
    
    def _construct_ticker(self, ticker: str, exchange: str) -> str:
        """
        Construct full ticker symbol based on exchange
        
        Args:
            ticker: Base ticker symbol
            exchange: Exchange code
        
        Returns:
            Full ticker symbol for Yahoo Finance
        """
        # Exchange suffix mapping
        exchange_suffixes = {
            'NASDAQ': '',
            'NYSE': '',
            'AMEX': '',
            'TSX': '.TO',      # Toronto
            'LSE': '.L',       # London
            'FRA': '.F',       # Frankfurt
            'PAR': '.PA',      # Paris
            'AMS': '.AS',      # Amsterdam
            'SWX': '.SW',      # Swiss
            'HKG': '.HK',      # Hong Kong
            'HKEX': '.HK',     # Hong Kong
            'TSE': '.T',       # Tokyo
            'TYO': '.T',       # Tokyo
            'KRX': '.KS',      # Korea
            'KSC': '.KS',      # Korea
            'ASX': '.AX',      # Australia
            'BSE': '.BO',      # Bombay
            'NSE': '.NS',      # National Stock Exchange India
            'SSE': '.SS',      # Shanghai
            'SZSE': '.SZ',     # Shenzhen
            'TPE': '.TW',      # Taiwan
            'TWSE': '.TW',     # Taiwan
            'BMV': '.MX',      # Mexico
            'BOVESPA': '.SA',  # Brazil
            'JSE': '.JO',      # Johannesburg
            'OMX': '.ST',      # Stockholm
            'OSE': '.OL',      # Oslo
            'CPH': '.CO',      # Copenhagen
            'HEL': '.HE',      # Helsinki
            'MOEX': '.ME',     # Moscow
            'TSEC': '.TW',     # Taiwan
        }
        
        # Get suffix for exchange, default to no suffix (US exchanges)
        suffix = exchange_suffixes.get(exchange.upper(), '')
        
        # Return full ticker
        return f"{ticker}{suffix}"
    
    def enrich_batch(self, companies: List[Dict[str, Any]], 
                     show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Enrich multiple companies with financial data (parallel processing)
        
        Args:
            companies: List of company dictionaries
            show_progress: Whether to print progress
        
        Returns:
            List of enriched company dictionaries
        """
        enriched = []
        
        # Use ThreadPoolExecutor for parallel API calls
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_company = {
                executor.submit(self.enrich_company, company): company 
                for company in companies
            }
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_company), 1):
                try:
                    enriched_company = future.result()
                    enriched.append(enriched_company)
                    
                    if show_progress:
                        print(f"✓ Enriched {i}/{len(companies)}: {enriched_company['name']}")
                        
                except Exception as e:
                    # If enrichment fails, add the original company
                    company = future_to_company[future]
                    company['financials'] = {'data_quality': 'error', 'error': str(e)}
                    enriched.append(company)
                    
                    if show_progress:
                        print(f"✗ Failed {i}/{len(companies)}: {company['name']} - {e}")
        
        return enriched


# Example usage and testing
if __name__ == "__main__":
    # Test with some companies
    test_companies = [
        {"name": "Apple Inc.", "ticker": "AAPL", "exchange": "NASDAQ"},
        {"name": "Panasonic", "ticker": "6752", "exchange": "TSE"},
        {"name": "Samsung", "ticker": "005930", "exchange": "KRX"},
        {"name": "HSBC", "ticker": "HSBA", "exchange": "LSE"},
    ]
    
    enricher = FinancialDataEnricher()
    
    print("Testing Financial Data Enrichment with Currency Conversion:")
    print("=" * 70)
    
    for company in test_companies:
        enriched = enricher.enrich_company(company.copy())
        fin = enriched.get('financials', {})
        
        print(f"\n{company['name']} ({company['ticker']}.{company['exchange']})")
        print(f"  Original Currency: {fin.get('original_currency', 'N/A')}")
        print(f"  Market Cap (USD): {fin.get('market_cap_formatted', 'N/A')}")
        print(f"  Revenue (USD): {fin.get('revenue_ttm_formatted', 'N/A')}")
        print(f"  EV/Revenue: {fin.get('ev_to_revenue', 'N/A')}")

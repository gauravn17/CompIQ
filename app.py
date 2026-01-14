import streamlit as st
import pandas as pd
from datetime import datetime
import time
import os
from typing import Dict, Any, List
import json

from comps_agent import ComparablesAgent
from database import Database, SearchHistory

# Logo fetcher for company logos
try:
    from logo_fetcher import CompanyLogoFetcher, get_logo_html, add_logos_to_comparables
    LOGOS_ENABLED = True
except ImportError:
    LOGOS_ENABLED = False
    print("⚠️ Logo fetcher not available")

# Try to import v2.0 features (graceful degradation if not available)
try:
    from financial_data import FinancialDataEnricher
    from visualizations import CompIQVisualizer, render_financial_summary, render_comparison_matrix
    ENHANCED_FEATURES = True
except ImportError:
    ENHANCED_FEATURES = False
    print("⚠️ Enhanced features not available. Run: pip install yfinance plotly")

# Page configuration
st.set_page_config(
    page_title="CompIQ - AI Comparables Finder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .logo-container {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-bottom: 1rem;
    }
    .logo-image {
        width: 150px;
        height: 150px;
        filter: drop-shadow(0 2px 6px rgba(0,0,0,0.15));
    }
    .brand-title {
        font-size: 3.5rem;
        font-weight: bold;
        color: #000000;
        margin: 0;
        line-height: 1.2;
    }
    .brand-subtitle {
        font-size: 1.2rem;
        color: #666;
        margin: 0;
        margin-top: 0.5rem;
    }
    /* Add subtle shadow to images for better visibility */
    img {
        filter: drop-shadow(0 2px 6px rgba(0,0,0,0.15));
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .status-analyzing { border-color: #ff7f0e; background-color: #fff3e0; }
    .status-searching { border-color: #2ca02c; background-color: #e8f5e9; }
    .status-validating { border-color: #d62728; background-color: #ffebee; }
    .status-complete { border-color: #1f77b4; background-color: #e3f2fd; }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-card h2 {
        color: #1f77b4 !important;
        font-size: 2.5rem !important;
        margin: 0 !important;
        font-weight: bold !important;
    }
    .metric-card p {
        color: #333 !important;
        font-size: 0.9rem !important;
        margin: 0.5rem 0 0 0 !important;
        font-weight: 500 !important;
    }
    .company-card {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: white;
        transition: box-shadow 0.2s ease;
    }
    .company-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .company-card h3 {
        color: #1f77b4;
    }
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: bold;
        color: white;
    }
    .score-high { background-color: #2ca02c; }
    .score-medium { background-color: #ff7f0e; }
    .score-low { background-color: #d62728; }
    .v2-badge {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = Database()
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'show_enhanced' not in st.session_state:
    st.session_state.show_enhanced = ENHANCED_FEATURES

def load_search_history():
    """Load recent searches from database"""
    st.session_state.search_history = st.session_state.db.get_recent_searches(limit=10)

def get_score_class(score: float) -> str:
    """Get CSS class for score badge"""
    if score >= 5.0:
        return "score-high"
    elif score >= 3.0:
        return "score-medium"
    else:
        return "score-low"

def get_logo_url(comp: Dict[str, Any]) -> tuple:
    """
    Get company logo URL with smart fallback.
    Returns (primary_url, fallback_url)
    """
    # Extract domain from homepage URL
    homepage = comp.get('homepage_url', comp.get('url', ''))
    if homepage:
        domain = homepage.replace('https://', '').replace('http://', '').split('/')[0]
    else:
        # Try to construct from company name
        name = comp.get('name', '').lower().replace(' ', '').replace(',', '').replace('.', '')
        domain = f"{name}.com"
    
    ticker = comp.get('ticker', 'N/A')
    
    # Primary: Clearbit (free, high quality)
    clearbit_url = f"https://logo.clearbit.com/{domain}"
    
    # Fallback 1: Google Favicon (always works)
    google_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
    
    # Fallback 2: UI Avatars (generated from ticker)
    avatar_url = f"https://ui-avatars.com/api/?name={ticker}&size=64&background=1f77b4&color=fff&bold=true"
    
    return clearbit_url, google_url, avatar_url

def render_company_card(comp: Dict[str, Any], rank: int):
    """Render a single comparable company card - compact layout with logo"""
    score = comp.get('validation_score', 0)
    score_class = get_score_class(score)
    
    # Get logo URLs
    logo_primary, logo_fallback1, logo_fallback2 = get_logo_url(comp)
    
    # Get financial data if available
    fin = comp.get('financials', {})
    
    # Build the card HTML with logo
    card_html = f"""
    <div class="company-card">
        <div style="display: flex; gap: 1rem; align-items: center; margin-bottom: 0.75rem;">
            <img src="{logo_primary}" 
                 onerror="this.onerror=null; this.src='{logo_fallback1}';"
                 style="width: 56px; height: 56px; border-radius: 8px; object-fit: contain; background: #f8f9fa; padding: 4px; border: 1px solid #e0e0e0; flex-shrink: 0;"
                 alt="{comp['name']} logo">
            <div style="flex: 1; min-width: 0;">
                <h3 style="margin: 0; font-size: 1.3rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{rank}. {comp['name']}</h3>
                <p style="margin: 0.25rem 0 0 0; color: #666; font-size: 0.9rem;">
                    <strong>{comp['ticker']}</strong> • {comp['exchange']}
                </p>
            </div>
            <span class="score-badge {score_class}" style="font-size: 1.1rem; padding: 0.4rem 0.9rem; flex-shrink: 0;">{score:.2f}</span>
        </div>
        
        <p style="margin: 0.5rem 0 0.5rem 72px; font-size: 0.95rem; line-height: 1.4; color: #333;">
            {comp.get('business_activity', 'N/A')[:180]}...
        </p>
    """
    
    # Add financial metrics inline if available
    if ENHANCED_FEATURES and fin.get('market_cap_formatted'):
        card_html += f"""
        <div style="display: flex; gap: 1.5rem; margin: 0.75rem 0 0 72px; padding-top: 0.75rem; border-top: 1px solid #eee; font-size: 0.85rem; flex-wrap: wrap;">
            <span style="display: flex; align-items: center; gap: 0.25rem;">
                <span style="color: #666;">💰</span>
                <strong>Market Cap:</strong> {fin.get('market_cap_formatted', 'N/A')}
            </span>
            <span style="display: flex; align-items: center; gap: 0.25rem;">
                <span style="color: #666;">📊</span>
                <strong>Revenue:</strong> {fin.get('revenue_ttm_formatted', 'N/A')}
            </span>
            <span style="display: flex; align-items: center; gap: 0.25rem;">
                <span style="color: #666;">📈</span>
                <strong>EV/Rev:</strong> {fin.get('ev_to_revenue', 'N/A')}x
            </span>
        </div>
        """
    
    # Close the card div
    card_html += """
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Expandable details - more compact
    with st.expander("📊 View Details", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📍 Business Info:**")
            st.write(f"• Customer: {comp.get('customer_segment', 'N/A')}")
            st.write(f"• Industry: {comp.get('SIC_industry', 'N/A')}")
            st.write(f"• [Website]({comp.get('url', '#')})")
            
            # Show financial data if available
            if ENHANCED_FEATURES and comp.get('financials'):
                fin = comp['financials']
                st.markdown("**💰 Financials:**")
                if fin.get('revenue_growth'):
                    st.write(f"• Revenue Growth: {fin['revenue_growth']*100:.1f}%")
                if fin.get('profit_margin'):
                    st.write(f"• Profit Margin: {fin['profit_margin']*100:.1f}%")
                if fin.get('employees'):
                    st.write(f"• Employees: {fin['employees']:,}")
        
        with col2:
            st.markdown("**🎯 Score Breakdown:**")
            breakdown = comp.get('score_breakdown', {})
            for key, value in breakdown.items():
                display_key = key.replace('_', ' ').title()
                st.write(f"• {display_key}: {value}")
        
        if comp.get('_caveat'):
            st.warning(f"⚠️ {comp['_caveat']}")
        if comp.get('_needs_verification'):
            st.info(f"ℹ️ {comp.get('_verification_note', 'Manual verification recommended')}")

def main():
    # Header with logo
    col1, col2 = st.columns([1.2, 3.8])
    
    with col1:
        # Display logo - larger and more prominent
        try:
            st.image("compiq.png", width=150)
        except:
            st.markdown("🔍")
    
    with col2:
        st.markdown("""
        <div style="padding-top: 20px;">
            <h1 class="brand-title">CompIQ</h1>
            <p class="brand-subtitle">AI-Powered Comparable Company Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        if ENHANCED_FEATURES:
            st.markdown('<span class="v2-badge">v2.0 ENHANCED</span>', unsafe_allow_html=True)
    
    st.divider()
    
    # Sidebar
    with st.sidebar:
        # Logo in sidebar - centered and larger
        col1, col2, col3 = st.columns([0.5, 3, 0.5])
        with col2:
            try:
                st.image("compiq.png", width=120)
            except:
                pass
        
        st.markdown("<h3 style='text-align: center; margin-top: 10px;'>CompIQ</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666; margin-bottom: 20px;'>AI Comparables Finder</p>", unsafe_allow_html=True)
        
        st.divider()
        
        st.header("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value="",  # Never show existing key
            placeholder="sk-proj-...",
            help="Your OpenAI API key for running the analysis. Get one at: https://platform.openai.com/api-keys"
        )
        
        # Check if key exists in environment (from Streamlit secrets)
        env_key = os.getenv("OPENAI_API_KEY", "")
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("✅ API key configured")
        elif env_key:
            st.success("✅ API key loaded from environment")
        else:
            st.warning("⚠️ Please enter your OpenAI API key to use CompIQ")
            st.info("💡 Get your API key at: https://platform.openai.com/api-keys")
        
        st.divider()
        
        # Settings
        st.subheader("Search Settings")
        min_required = st.slider("Minimum Comparables", 1, 10, 3)
        max_allowed = st.slider("Maximum Comparables", 5, 20, 10)
        max_attempts = st.slider("Max Search Attempts", 1, 5, 3)
        
        # v2.0 Feature Toggle
        if ENHANCED_FEATURES:
            st.divider()
            st.subheader("🎛️ v2.0 Features")
            enable_financials = st.checkbox(
                "Financial Data", 
                value=True,
                help="Pull real-time market cap, revenue, multiples"
            )
            enable_charts = st.checkbox(
                "Interactive Charts",
                value=True,
                help="Show visualizations and comparisons"
            )
        else:
            enable_financials = False
            enable_charts = False
        
        st.divider()
        
        # Search History
        st.subheader("📊 Recent Searches")
        if st.button("🔄 Refresh History"):
            load_search_history()
        
        if st.session_state.search_history:
            for search in st.session_state.search_history[:5]:
                with st.expander(f"{search['target_name'][:30]}..."):
                    st.write(f"**Date:** {search['timestamp'][:10]}")
                    st.write(f"**Found:** {search['num_comparables']} companies")
                    if st.button(f"Load", key=f"load_{search['id']}"):
                        # Load previous search
                        results = st.session_state.db.get_search_results(search['id'])
                        if results:
                            st.session_state.search_results = {
                                'comparables': results['comparables'],
                                'metadata': results['metadata'],
                                'target': {'name': search['target_name']}
                            }
                            st.rerun()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 New Search", "📊 Results", "📚 Database"])
    
    with tab1:
        st.header("Find Comparable Companies")
        
        # Input form
        with st.form("target_company_form"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                company_name = st.text_input(
                    "Company Name",
                    placeholder="e.g., Palantir Technologies",
                    help="Name of the target company"
                )
                
                company_description = st.text_area(
                    "Business Description",
                    height=150,
                    placeholder="Detailed description of what the company does, its products/services, target markets, etc.",
                    help="The more detailed, the better the results"
                )
            
            with col2:
                homepage_url = st.text_input(
                    "Homepage URL",
                    placeholder="https://company.com"
                )
                
                primary_sic = st.text_input(
                    "Primary SIC (Optional)",
                    placeholder="e.g., Prepackaged Software"
                )
            
            submitted = st.form_submit_button("🚀 Find Comparables", use_container_width=True)
        
        # Process search
        if submitted and company_name and company_description:
            if not (api_key or env_key):
                st.error("⚠️ Please provide an OpenAI API key in the sidebar")
            else:
                # Create target company dict
                target = {
                    "name": company_name,
                    "description": company_description,
                    "homepage_url": homepage_url or "https://example.com",
                    "primary_sic": primary_sic or "Unknown"
                }
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_container = st.container()
                
                with status_container:
                    st.markdown('<div class="status-box status-analyzing">⏳ Analyzing target company...</div>', unsafe_allow_html=True)
                
                try:
                    # Initialize agent
                    agent = ComparablesAgent(
                        min_required=min_required,
                        max_allowed=max_allowed,
                        max_attempts=max_attempts
                    )
                    
                    # Run search with progress updates
                    progress_bar.progress(10)
                    status_container.markdown('<div class="status-box status-analyzing">🧠 Analyzing target company...</div>', unsafe_allow_html=True)
                    
                    results = agent.find_comparables(
                        target,
                        progress_callback=lambda step, progress: progress_bar.progress(progress)
                    )
                    
                    # Enrich with financial data if enabled
                    if ENHANCED_FEATURES and enable_financials:
                        progress_bar.progress(93)
                        status_container.markdown('<div class="status-box status-analyzing">💰 Loading financial data...</div>', unsafe_allow_html=True)
                        
                        enricher = FinancialDataEnricher()
                        results['comparables'] = enricher.enrich_batch(results['comparables'])
                    
                    # Fetch company logos
                    if LOGOS_ENABLED:
                        progress_bar.progress(97)
                        status_container.markdown('<div class="status-box status-analyzing">🎨 Loading company logos...</div>', unsafe_allow_html=True)
                        
                        results['comparables'] = add_logos_to_comparables(results['comparables'])
                    
                    progress_bar.progress(100)
                    status_container.markdown('<div class="status-box status-complete">✅ Search complete!</div>', unsafe_allow_html=True)
                    
                    # Store results
                    st.session_state.search_results = {
                        'comparables': results['comparables'],
                        'metadata': results['metadata'],
                        'target': target
                    }
                    
                    # Save to database
                    search_id = st.session_state.db.save_search(
                        target_name=company_name,
                        target_data=target,
                        comparables=results['comparables'],
                        metadata=results['metadata']
                    )
                    
                    st.success(f"✅ Found {len(results['comparables'])} comparable companies!")
                    st.balloons()
                    
                    # Auto-switch to results tab
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error during search: {str(e)}")
                    status_container.markdown(f'<div class="status-box status-validating">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
    
    with tab2:
        st.header("Search Results")
        
        if st.session_state.search_results:
            results = st.session_state.search_results
            comparables = results['comparables']
            metadata = results['metadata']
            target = results['target']
            
            # Summary metrics
            st.subheader(f"Target: {target['name']}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Comparables Found", len(comparables))
            with col2:
                avg_score = sum(c['validation_score'] for c in comparables) / len(comparables) if comparables else 0
                st.metric("Avg Score", f"{avg_score:.2f}")
            with col3:
                num_rejected = len(metadata.get('rejected_companies', []))
                st.metric("Rejected", num_rejected)
            with col4:
                specialization = metadata.get('analysis', {}).get('specialization_level', 0)
                st.metric("Specialization", f"{specialization:.2f}")
            
            st.divider()
            
            # Analysis insights
            with st.expander("📊 Analysis Insights", expanded=True):
                analysis = metadata.get('analysis', {})
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Focus Areas:**")
                    for area in analysis.get('core_focus_areas', [])[:5]:
                        st.write(f"- {area}")
                with col2:
                    st.write("**Business Model:**", analysis.get('business_model', 'N/A'))
                    st.write("**Key Differentiators:**")
                    for diff in analysis.get('key_differentiators', [])[:3]:
                        st.write(f"- {diff}")
            
            st.divider()
            
            # ===== v2.0 ENHANCED FEATURES =====
            if ENHANCED_FEATURES and enable_charts:
                try:
                    # Financial Summary
                    st.subheader("💰 Peer Group Valuation Metrics")
                    render_financial_summary(comparables)
                    
                    st.divider()
                    
                    # Visual Analysis
                    st.subheader("📊 Visual Analysis")
                    visualizer = CompIQVisualizer()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        score_fig = visualizer.create_score_distribution(comparables)
                        st.plotly_chart(score_fig, use_container_width=True)
                    
                    with col2:
                        val_fig = visualizer.create_valuation_comparison(comparables)
                        if val_fig:
                            st.plotly_chart(val_fig, use_container_width=True)
                        else:
                            st.info("💡 Valuation data not available for all companies")
                    
                    # Radar comparison
                    radar_fig = visualizer.create_radar_comparison(comparables, top_n=5)
                    st.plotly_chart(radar_fig, use_container_width=True)
                    
                    st.divider()
                    
                    # Comparison Matrix
                    render_comparison_matrix(comparables, top_n=5)
                    
                    st.divider()
                    
                    # Detailed Metrics Table
                    st.subheader("📋 Detailed Financial Metrics")
                    metrics_df = visualizer.create_peer_metrics_table(comparables)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    st.divider()
                    
                except Exception as e:
                    st.warning(f"Some enhanced features unavailable: {e}")
            
            # Comparable companies (basic cards)
            st.subheader("📋 Comparable Companies")
            
            for i, comp in enumerate(comparables, 1):
                render_company_card(comp, i)
            
            st.divider()
            
            # Rejected companies
            rejected = metadata.get('rejected_companies', [])
            if rejected:
                with st.expander(f"❌ Rejected Companies ({len(rejected)})"):
                    rejected_df = pd.DataFrame([{
                        'Name': r.get('company', {}).get('name', 'Unknown'),
                        'Ticker': r.get('company', {}).get('ticker', 'N/A'),
                        'Status': r.get('status', 'UNKNOWN'),
                        'Reason': r.get('reason', 'N/A')[:100],
                        'Acquirer': r.get('acquirer', 'N/A')
                    } for r in rejected[:20]])
                    st.dataframe(rejected_df, use_container_width=True)
            
            # Export options
            st.divider()
            st.subheader("📥 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV export
                if ENHANCED_FEATURES and comparables[0].get('financials'):
                    csv_data = pd.DataFrame([{
                        'Rank': i+1,
                        'Name': c['name'],
                        'Ticker': c['ticker'],
                        'Exchange': c['exchange'],
                        'Score': c['validation_score'],
                        'Market Cap': c.get('financials', {}).get('market_cap_formatted', 'N/A'),
                        'Revenue': c.get('financials', {}).get('revenue_ttm_formatted', 'N/A'),
                        'EV/Revenue': c.get('financials', {}).get('ev_to_revenue', 'N/A'),
                        'Business': c.get('business_activity', ''),
                        'URL': c.get('url', '')
                    } for i, c in enumerate(comparables)])
                else:
                    csv_data = pd.DataFrame([{
                        'Rank': i+1,
                        'Name': c['name'],
                        'Ticker': c['ticker'],
                        'Exchange': c['exchange'],
                        'Score': c['validation_score'],
                        'Business': c.get('business_activity', ''),
                        'Customer Segment': c.get('customer_segment', ''),
                        'SIC Industry': c.get('SIC_industry', ''),
                        'URL': c.get('url', '')
                    } for i, c in enumerate(comparables)])
                
                st.download_button(
                    "📄 Download CSV",
                    csv_data.to_csv(index=False),
                    "comparables.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # JSON export
                json_data = json.dumps({
                    'target': target,
                    'comparables': comparables,
                    'metadata': metadata
                }, indent=2)
                
                st.download_button(
                    "📋 Download JSON",
                    json_data,
                    "comparables.json",
                    "application/json",
                    use_container_width=True
                )
            
            with col3:
                # Excel export would go here
                st.button("📊 Generate Report", use_container_width=True, help="Coming soon!")
        
        else:
            st.info("👈 Run a new search to see results here")
    
    with tab3:
        st.header("Company Database")
        st.info("🚧 Database exploration features coming soon!")
        
        # Show stats
        stats = st.session_state.db.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Searches", stats['total_searches'])
        with col2:
            st.metric("Unique Companies", stats['unique_companies'])

if __name__ == "__main__":
    # Load history on startup
    load_search_history()
    main()


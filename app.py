import streamlit as st
import pandas as pd
from datetime import datetime
import time
import os
from typing import Dict, Any, List
import json

from comps_agent import ComparablesAgent
from database import Database, SearchHistory

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
    Returns (primary_url, fallback_url, fallback_url2)
    """
    # Manual mapping for common companies with tricky names
    DOMAIN_MAP = {
        'dell technologies': 'dell.com',
        'hewlett packard': 'hp.com',
        'hp inc': 'hp.com',
        'xiaomi corporation': 'xiaomi.com',
        'acer incorporated': 'acer.com',
        'acer inc': 'acer.com',
        'international business machines': 'ibm.com',
        'microsoft corporation': 'microsoft.com',
        'apple inc': 'apple.com',
        'alphabet inc': 'google.com',
        'meta platforms': 'meta.com',
        'amazon.com inc': 'amazon.com',
    }
    
    # Extract domain from homepage URL
    homepage = comp.get('homepage_url', comp.get('url', ''))
    if homepage:
        domain = homepage.replace('https://', '').replace('http://', '').split('/')[0]
    else:
        # Try manual mapping first
        name_lower = comp.get('name', '').lower()
        domain = None
        
        for key, mapped_domain in DOMAIN_MAP.items():
            if key in name_lower:
                domain = mapped_domain
                break
        
        # If no mapping found, construct from company name
        if not domain:
            name = comp.get('name', '').lower()
            # Remove common suffixes
            for suffix in [' inc.', ' inc', ' corporation', ' corp.', ' corp', ' ltd.', ' ltd', ' llc', ' technologies', ' group', ' company']:
                name = name.replace(suffix, '')
            name = name.strip().replace(' ', '').replace(',', '').replace('.', '')
            domain = f"{name}.com"
    
    ticker = comp.get('ticker', 'N/A')
    
    # Primary: Google Favicon (always works, reliable)
    google_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
    
    # Fallback 1: UI Avatars (generated from ticker)
    avatar_url = f"https://ui-avatars.com/api/?name={ticker}&size=64&background=667eea&color=fff&bold=true&font-size=0.5"
    
    # Fallback 2: DuckDuckGo icons (another reliable service)
    ddg_url = f"https://icons.duckduckgo.com/ip3/{domain}.ico"
    
    return google_url, avatar_url, ddg_url

def render_company_card(comp: Dict[str, Any], rank: int):
    """Render a single comparable company card - using Streamlit native components"""
    score = comp.get('validation_score', 0)
    score_class = get_score_class(score)
    
    # Get logo URLs
    logo_primary, logo_fallback1, logo_fallback2 = get_logo_url(comp)
    
    # Get financial data if available
    fin = comp.get('financials', {})
    
    # Use a container for better styling
    with st.container():
        # Create columns for logo, info, and score
        col_logo, col_info, col_score = st.columns([0.5, 4, 0.5])
        
        with col_logo:
            # Display logo with fallback to ticker badge only if logo fails
            logo_html = f"""
            <img src="{logo_primary}" 
                 onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';"
                 style="width: 56px; height: 56px; border-radius: 8px; object-fit: contain; background: #f8f9fa; padding: 4px; border: 1px solid #e0e0e0;">
            <div style="display: none; width: 56px; height: 56px; border-radius: 8px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); align-items: center; justify-content: center; font-size: 20px; color: white; font-weight: bold; border: 1px solid #e0e0e0;">
                {comp.get('ticker', '?')[:2]}
            </div>
            """
            st.markdown(logo_html, unsafe_allow_html=True)
        
        with col_info:
            st.markdown(f"### {rank}. {comp['name']}")
            st.markdown(f"**{comp['ticker']}** • {comp['exchange']}")
            st.markdown(f"{comp.get('business_activity', 'N/A')[:180]}...")
            
            # Add financial metrics if available
            if ENHANCED_FEATURES and fin.get('market_cap_formatted'):
                fin_cols = st.columns(3)
                with fin_cols[0]:
                    st.metric("Market Cap", fin.get('market_cap_formatted', 'N/A'))
                with fin_cols[1]:
                    st.metric("Revenue", fin.get('revenue_ttm_formatted', 'N/A'))
                with fin_cols[2]:
                    st.metric("EV/Rev", f"{fin.get('ev_to_revenue', 'N/A')}x")
        
        with col_score:
            score_color = "#2ca02c" if score >= 5.0 else "#ff7f0e" if score >= 3.0 else "#d62728"
            st.markdown(f'<div style="background-color: {score_color}; color: white; padding: 0.5rem; border-radius: 1rem; text-align: center; font-weight: bold;">{score:.2f}</div>', unsafe_allow_html=True)
        
        st.divider()
    
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
    # Professional header with perfect alignment
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 1.5rem; margin-bottom: 1rem; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 15px;">
        <div style="width: 80px; height: 80px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; display: flex; align-items: center; justify-content: center; font-size: 2.5rem; flex-shrink: 0; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);">
            🔍
        </div>
        <div style="flex: 1;">
            <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.2;">CompIQ</h1>
            <p style="margin: 0.25rem 0 0 0; font-size: 1.05rem; color: #aaa; font-weight: 400;">AI-Powered Comparable Company Analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if ENHANCED_FEATURES:
        st.markdown('<span class="v2-badge" style="margin-left: 95px;">v2.0 ENHANCED</span>', unsafe_allow_html=True)
    
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
            value="",
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
        # Stunning hero section
        st.markdown("""
        <div style="text-align: center; padding: 2.5rem 2rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%); border-radius: 15px; margin-bottom: 2rem;">
            <h1 style="font-size: 2.2rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 1rem; letter-spacing: -0.02em;">
                Find Your Perfect Comparables
            </h1>
            <p style="font-size: 1.1rem; color: #aaa; margin-bottom: 2rem; font-weight: 400; max-width: 600px; margin-left: auto; margin-right: auto;">
                AI-powered company analysis • Delivered in seconds
            </p>
            <div style="display: flex; justify-content: center; gap: 3rem; flex-wrap: wrap; max-width: 750px; margin: 0 auto;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <div style="width: 32px; height: 32px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                        <span style="color: white; font-weight: bold; font-size: 1.1rem;">✓</span>
                    </div>
                    <span style="color: #333; font-weight: 500; font-size: 1.05rem;">100,000+ Companies</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <div style="width: 32px; height: 32px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                        <span style="color: white; font-weight: bold; font-size: 1.1rem;">✓</span>
                    </div>
                    <span style="color: #333; font-weight: 500; font-size: 1.05rem;">Real-time Data</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <div style="width: 32px; height: 32px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                        <span style="color: white; font-weight: bold; font-size: 1.1rem;">✓</span>
                    </div>
                    <span style="color: #333; font-weight: 500; font-size: 1.05rem;">AI Matching</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Premium feature cards
        col1, col2, col3 = st.columns(3, gap="large")
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 1.75rem 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; height: 200px; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.25);">
                <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">🎯</div>
                <h3 style="margin: 0 0 0.75rem 0; color: white; font-size: 1.25rem; font-weight: 600;">Smart Analysis</h3>
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.95; line-height: 1.5;">AI understands your business model</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 1.75rem 1.5rem; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 15px; color: white; height: 200px; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 8px 20px rgba(245, 87, 108, 0.25);">
                <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">⚡</div>
                <h3 style="margin: 0 0 0.75rem 0; color: white; font-size: 1.25rem; font-weight: 600;">Lightning Fast</h3>
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.95; line-height: 1.5;">Results in under 30 seconds</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 1.75rem 1.5rem; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 15px; color: white; height: 200px; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 8px 20px rgba(79, 172, 254, 0.25);">
                <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">📊</div>
                <h3 style="margin: 0 0 0.75rem 0; color: white; font-size: 1.25rem; font-weight: 600;">Rich Insights</h3>
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.95; line-height: 1.5;">Financial data & visualizations</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Enhanced input form
        st.markdown("""
        <div style="background: rgba(255,255,255,0.05); padding: 1.5rem 2rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 1.5rem;">
            <h2 style="margin: 0 0 0.25rem 0; font-size: 1.5rem; color: #fff;">📝 Enter Company Details</h2>
            <p style="margin: 0; color: #aaa; font-size: 0.9rem;">The more detail you provide, the better your results</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("target_company_form"):
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("**Company Name** <span style='color: #f5576c;'>*</span>", unsafe_allow_html=True)
                company_name = st.text_input(
                    "Company Name",
                    placeholder="e.g., Apple Inc.",
                    help="Name of the target company",
                    label_visibility="collapsed"
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Business Description** <span style='color: #f5576c;'>*</span>", unsafe_allow_html=True)
                company_description = st.text_area(
                    "Business Description",
                    height=140,
                    placeholder="Example: Designs, develops, and sells consumer electronics, computer software, and online services. Products include iPhone, Mac, iPad, Apple Watch, and services like App Store, Apple Music, and iCloud.",
                    help="Provide a detailed description",
                    label_visibility="collapsed"
                )
            
            with col2:
                st.markdown("**Homepage URL**", unsafe_allow_html=True)
                homepage_url = st.text_input(
                    "Homepage URL",
                    placeholder="https://www.apple.com",
                    help="Optional: Company website",
                    label_visibility="collapsed"
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Primary Industry**", unsafe_allow_html=True)
                primary_sic = st.text_input(
                    "Primary SIC",
                    placeholder="e.g., Consumer Electronics",
                    help="Optional: Industry classification",
                    label_visibility="collapsed"
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Tips box
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.25rem; border-radius: 10px; color: white; margin-top: 1rem;">
                    <div style="font-weight: 600; margin-bottom: 0.5rem; font-size: 1rem;">💡 Pro Tips</div>
                    <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.9rem; line-height: 1.6;">
                        <li>Include key products/services</li>
                        <li>Mention target markets</li>
                        <li>Describe business model</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Submit button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submitted = st.form_submit_button(
                    "🚀 Find Comparables",
                    use_container_width=True,
                    type="primary"
                )
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Enhanced expandable sections
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("💼 Popular Use Cases", expanded=False):
                st.markdown("""
                **Investment Research**
                - Find peer companies for valuation
                - Compare financial metrics
                - Identify opportunities
                
                **M&A Analysis**
                - Identify acquisition targets
                - Benchmark valuations
                - Assess strategic fit
                
                **Competitive Intelligence**
                - Map competitive landscape
                - Track industry trends
                - Monitor positioning
                """)
        
        with col2:
            with st.expander("❓ How It Works", expanded=False):
                st.markdown("""
                **1. AI Analysis**  
                Our AI analyzes your company's business model and market position
                
                **2. Smart Search**  
                We search 100,000+ public companies globally
                
                **3. Validation**  
                Each match is scored on similarity, fit, and scale
                
                **4. Rich Data**  
                Get metrics, visualizations, and profiles
                
                ---
                *Powered by OpenAI GPT-4 & Yahoo Finance*
                """)
        

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


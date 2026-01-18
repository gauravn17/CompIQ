"""
Advanced visualization module for CompIQ.
Creates interactive charts and comparison views.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, List
import streamlit as st


class CompIQVisualizer:
    """Creates professional visualizations for comparable analysis."""
    
    @staticmethod
    def create_score_distribution(comparables: List[Dict[str, Any]]) -> go.Figure:
        """
        Create a histogram showing score distribution.
        """
        scores = [c.get('validation_score', 0) for c in comparables]
        names = [c.get('name', 'Unknown')[:20] for c in comparables]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=names,
            y=scores,
            marker=dict(
                color=scores,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Score")
            ),
            text=[f"{s:.2f}" for s in scores],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Comparable Scores Distribution",
            xaxis_title="Company",
            yaxis_title="Validation Score",
            height=400,
            showlegend=False,
            xaxis_tickangle=-45
        )
        
        return fig
    
    @staticmethod
    def create_valuation_comparison(comparables: List[Dict[str, Any]]) -> go.Figure:
        """
        Create bubble chart comparing market cap vs. revenue multiple.
        With jitter to prevent overlapping dots.
        """
        import numpy as np
        
        data = []
        
        for comp in comparables:
            fin = comp.get('financials', {})
            ev_to_rev = fin.get('ev_to_revenue')
            
            # Convert ev_to_revenue to float if it's a string
            if isinstance(ev_to_rev, str):
                try:
                    ev_to_rev = float(ev_to_rev)
                except (ValueError, TypeError):
                    ev_to_rev = None
            
            if fin.get('market_cap') and ev_to_rev:
                data.append({
                    'name': comp.get('name', 'Unknown')[:20],
                    'market_cap': fin['market_cap'],
                    'ev_revenue': ev_to_rev,
                    'score': comp.get('validation_score', 0),
                    'ticker': comp.get('ticker', '')
                })
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        
        # Add small random jitter to prevent exact overlaps (5% of value)
        np.random.seed(42)  # Consistent jitter
        df['market_cap_jitter'] = df['market_cap'] * (1 + np.random.uniform(-0.05, 0.05, len(df)))
        df['ev_revenue_jitter'] = df['ev_revenue'] * (1 + np.random.uniform(-0.02, 0.02, len(df)))
        
        fig = px.scatter(
            df,
            x='market_cap_jitter',
            y='ev_revenue_jitter',
            size='score',
            color='score',
            hover_data={'name': True, 'ticker': True, 
                       'market_cap_jitter': False, 'ev_revenue_jitter': False,
                       'market_cap': True, 'ev_revenue': True},
            text='ticker',
            color_continuous_scale='RdYlGn',
            labels={
                'market_cap_jitter': 'Market Cap ($)',
                'ev_revenue_jitter': 'EV/Revenue Multiple',
                'score': 'Match Score'
            }
        )
        
        fig.update_traces(
            textposition='top center',
            textfont_size=10,
            marker=dict(
                line=dict(width=1, color='white'),
                sizemode='diameter',
                sizemin=8
            )
        )
        
        fig.update_layout(
            title="Valuation Comparison (Market Cap vs EV/Revenue)",
            xaxis_type='log',
            height=500,
            showlegend=False,
            hovermode='closest'
        )
        
        return fig
    
    @staticmethod
    def create_radar_comparison(
        comparables: List[Dict[str, Any]], 
        top_n: int = 5
    ) -> go.Figure:
        """
        Create radar chart comparing top N companies across multiple dimensions.
        """
        # Take top N by score
        top_comps = sorted(
            comparables, 
            key=lambda x: x.get('validation_score', 0), 
            reverse=True
        )[:top_n]
        
        fig = go.Figure()
        
        categories = ['Match Score', 'Revenue Growth', 'Profit Margin', 'Market Position']
        
        for comp in top_comps:
            fin = comp.get('financials', {})
            score_breakdown = comp.get('score_breakdown', {})
            
            # Normalize values to 0-10 scale
            values = [
                min(comp.get('validation_score', 0), 10),
                (fin.get('revenue_growth', 0) * 100) if fin.get('revenue_growth') else 5,
                (fin.get('profit_margin', 0) * 100) if fin.get('profit_margin') else 5,
                7  # Placeholder for market position
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=comp.get('name', 'Unknown')[:20]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=True,
            title=f"Top {len(top_comps)} Comparables - Multidimensional View",
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_peer_metrics_table(comparables: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create formatted comparison table with key metrics.
        """
        rows = []
        
        for comp in comparables:
            fin = comp.get('financials', {})
            
            row = {
                'Company': comp.get('name', 'Unknown'),
                'Ticker': comp.get('ticker', 'N/A'),
                'Score': f"{comp.get('validation_score', 0):.2f}",
                'Market Cap': fin.get('market_cap_formatted', 'N/A'),
                'Revenue (TTM)': fin.get('revenue_ttm_formatted', 'N/A'),
                'EV/Revenue': f"{fin.get('ev_to_revenue', 0):.2f}x" if fin.get('ev_to_revenue') else 'N/A',
                'Revenue Growth': f"{fin.get('revenue_growth', 0) * 100:.1f}%" if fin.get('revenue_growth') else 'N/A',
                'Profit Margin': f"{fin.get('profit_margin', 0) * 100:.1f}%" if fin.get('profit_margin') else 'N/A',
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    @staticmethod
    def create_valuation_summary_card(
        comparables: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create summary statistics for the peer group.
        """
        ev_revenues = []
        market_caps = []
        revenue_growths = []
        
        for comp in comparables:
            fin = comp.get('financials', {})
            
            if fin.get('ev_to_revenue'):
                ev_revenues.append(fin['ev_to_revenue'])
            
            if fin.get('market_cap'):
                market_caps.append(fin['market_cap'])
            
            if fin.get('revenue_growth'):
                revenue_growths.append(fin['revenue_growth'] * 100)
        
        summary = {
            'median_ev_revenue': None,
            'median_market_cap': None,
            'median_revenue_growth': None,
            'sample_size': len([c for c in comparables if c.get('financials', {}).get('market_cap')])
        }
        
        if ev_revenues:
            summary['median_ev_revenue'] = sorted(ev_revenues)[len(ev_revenues) // 2]
        
        if market_caps:
            summary['median_market_cap'] = sorted(market_caps)[len(market_caps) // 2]
        
        if revenue_growths:
            summary['median_revenue_growth'] = sorted(revenue_growths)[len(revenue_growths) // 2]
        
        return summary
    
    @staticmethod
    def create_score_breakdown_chart(comparable: Dict[str, Any]) -> go.Figure:
        """
        Create waterfall chart showing how score was calculated.
        """
        breakdown = comparable.get('score_breakdown', {})
        
        components = []
        values = []
        
        for key, value in breakdown.items():
            if isinstance(value, (int, float)):
                components.append(key.replace('_', ' ').title())
                values.append(value)
            elif isinstance(value, str) and 'weighted' in value:
                # Extract numeric value from strings like "0.75 (weighted 4.5x)"
                try:
                    numeric = float(value.split()[0])
                    components.append(key.replace('_', ' ').title())
                    values.append(numeric)
                except:
                    pass
        
        if not values:
            return None
        
        fig = go.Figure(go.Waterfall(
            name="Score",
            orientation="v",
            measure=["relative"] * (len(values) - 1) + ["total"],
            x=components,
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=f"Score Breakdown: {comparable.get('name', 'Unknown')}",
            height=400,
            showlegend=False
        )
        
        return fig


def render_comparison_matrix(
    comparables: List[Dict[str, Any]], 
    top_n: int = 5
):
    """
    Render side-by-side comparison of top N companies.
    Streamlit component.
    """
    top_comps = sorted(
        comparables,
        key=lambda x: x.get('validation_score', 0),
        reverse=True
    )[:top_n]
    
    st.subheader(f"🔬 Top {len(top_comps)} Comparables - Detailed Comparison")
    
    # Create columns for each company
    cols = st.columns(len(top_comps))
    
    for idx, comp in enumerate(top_comps):
        with cols[idx]:
            fin = comp.get('financials', {})
            
            st.markdown(f"### {idx + 1}. {comp.get('name', 'Unknown')[:15]}")
            st.markdown(f"**{comp.get('ticker', 'N/A')}** • {comp.get('exchange', 'N/A')}")
            
            score = comp.get('validation_score', 0)
            score_color = "🟢" if score >= 5.0 else "🟡" if score >= 3.0 else "🔴"
            st.metric("Match Score", f"{score:.2f} {score_color}")
            
            st.markdown("---")
            
            # Financial metrics
            if fin.get('market_cap_formatted'):
                st.metric("Market Cap", fin['market_cap_formatted'])
            
            if fin.get('revenue_ttm_formatted'):
                st.metric("Revenue (TTM)", fin['revenue_ttm_formatted'])
            
            if fin.get('ev_to_revenue'):
                ev_val = fin['ev_to_revenue']
                if isinstance(ev_val, str):
                    st.metric("EV/Revenue", f"{ev_val}x")
                else:
                    st.metric("EV/Revenue", f"{ev_val:.2f}x")
            
            if fin.get('revenue_growth'):
                growth = fin['revenue_growth'] * 100
                st.metric(
                    "Revenue Growth", 
                    f"{growth:.1f}%",
                    delta=f"{growth:.1f}%" if growth > 0 else None
                )
            
            # Business info
            with st.expander("📊 Details"):
                st.write(f"**Sector:** {fin.get('sector', 'N/A')}")
                st.write(f"**Industry:** {fin.get('industry', 'N/A')}")
                st.write(f"**Employees:** {fin.get('employees', 'N/A'):,}" if fin.get('employees') else "**Employees:** N/A")
                st.write(f"**Website:** {comp.get('url', 'N/A')}")


def render_financial_summary(comparables: List[Dict[str, Any]]):
    """
    Render financial summary metrics for peer group.
    No header - app.py adds it.
    """
    visualizer = CompIQVisualizer()
    summary = visualizer.create_valuation_summary_card(comparables)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        median_ev = summary.get('median_ev_revenue')
        if median_ev:
            st.metric("Median EV/Revenue", f"{median_ev:.2f}x")
        else:
            st.metric("Median EV/Revenue", "N/A")
    
    with col2:
        median_cap = summary.get('median_market_cap')
        if median_cap:
            if median_cap >= 1_000_000_000:
                st.metric("Median Market Cap", f"${median_cap/1_000_000_000:.2f}B")
            else:
                st.metric("Median Market Cap", f"${median_cap/1_000_000:.0f}M")
        else:
            st.metric("Median Market Cap", "N/A")
    
    with col3:
        median_growth = summary.get('median_revenue_growth')
        if median_growth:
            st.metric("Median Revenue Growth", f"{median_growth:.1f}%")
        else:
            st.metric("Median Revenue Growth", "N/A")
    
    with col4:
        st.metric("Data Coverage", f"{summary.get('sample_size', 0)}/{len(comparables)}")

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# 1. Page Configuration
st.set_page_config(
    page_title="Uplift AI Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #999;
        margin-bottom: 2rem;
    }
    
    /* Enhanced metric styling for better visibility */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #667eea !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: #e0e0e0 !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.875rem !important;
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1.5rem !important;
        border-radius: 12px !important;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.6);
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🚀 Causal AI Marketing Intelligence Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Leverage Uplift Modeling to identify high-value persuadable customers and optimize campaign ROI</p>', unsafe_allow_html=True)

# Info expander
with st.expander("ℹ️ How This Works - Causal Machine Learning"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Traditional Approach (❌ Inefficient):**
        - Targets anyone likely to buy
        - Wastes budget on "Sure Things" who buy anyway
        - Annoys "Sleeping Dogs" who react negatively
        """)
    with col2:
        st.markdown("""
        **Uplift AI Approach (✅ Intelligent):**
        - Identifies **Persuadables**: buy *only if* contacted
        - Filters out negative responders
        - Maximizes incremental revenue per dollar spent
        """)

# 2. Load the Model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('uplift_model.pkl')
        return model
    except FileNotFoundError:
        return None

model = load_model()

# Sidebar Status
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/artificial-intelligence.png", width=150)
    st.header("⚙️ System Status")
    
    if model:
        st.success("✅ AI Model Loaded")
        st.info("**Model:** Class Transformation (Lai Method)")
        st.info("**Algorithm:** XGBoost Classifier")
    else:
        st.error("❌ Model not found! Please run train.py first.")
        st.stop()
    
    st.divider()
    
    # Advanced settings
    st.subheader("🎯 Targeting Settings")
    uplift_threshold = st.slider(
        "Minimum Uplift Score",
        min_value=-0.5,
        max_value=0.5,
        value=0.0,
        step=0.01,
        help="Only target customers above this uplift score"
    )
    
    st.divider()
    
    st.subheader("💰 Cost Parameters")
    email_cost = st.number_input(
        "Cost per Email ($)",
        min_value=0.0,
        max_value=10.0,
        value=0.10,
        step=0.01
    )
    
    avg_order_value = st.number_input(
        "Avg Order Value ($)",
        min_value=0.0,
        max_value=1000.0,
        value=50.0,
        step=5.0
    )
    
    st.divider()
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# 3. File Upload Section
uploaded_file = st.file_uploader("📂 Upload Customer CSV", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    
    st.write("### 🔍 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        st.metric("Data Features", len(df.columns))
    with col3:
        st.metric("Data Quality", f"{((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100):.1f}%")

    # Check if necessary columns exist
    required_cols = ['recency', 'history', 'zip_code', 'channel']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}")
        st.info("Required columns: recency, history, zip_code, channel")
    else:
        if st.button("🔮 Run Uplift Prediction", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing Customer Psychology with Causal AI..."):
                
                # 4. PREDICT
                uplift_scores = model.predict(df)
                
                # Add scores to dataframe
                df['Uplift_Score'] = uplift_scores
                
                # 5. DECISION LOGIC with threshold
                df['Recommendation'] = df['Uplift_Score'].apply(
                    lambda x: "✅ Send Email" if x > uplift_threshold else "⛔ Do Not Send"
                )
                
                # Add customer segment labels
                df['Segment'] = pd.cut(
                    df['Uplift_Score'],
                    bins=[-float('inf'), -0.1, 0, 0.1, float('inf')],
                    labels=['😴 Sleeping Dogs', '🤷 Lost Causes', '✨ Persuadables', '🎯 Sure Things']
                )
                
                # Estimate potential revenue
                df['Est_Revenue'] = df['Uplift_Score'] * avg_order_value
                df['Est_Revenue'] = df['Est_Revenue'].clip(lower=0)
                
                # 6. METRICS
                st.divider()
                st.subheader("📊 Campaign Intelligence Dashboard")
                
                target_count = df[df['Recommendation'] == "✅ Send Email"].shape[0]
                total_count = df.shape[0]
                pct_target = (target_count / total_count) * 100
                
                # Calculate ROI metrics
                email_costs = target_count * email_cost
                potential_revenue = df[df['Recommendation'] == "✅ Send Email"]['Est_Revenue'].sum()
                roi = ((potential_revenue - email_costs) / email_costs * 100) if email_costs > 0 else 0
                savings = (total_count - target_count) * email_cost
                
                # Display metrics in cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="🎯 Customers to Target",
                        value=f"{target_count:,}",
                        delta=f"{pct_target:.1f}% of total"
                    )
                with col2:
                    st.metric(
                        label="💰 Potential Revenue",
                        value=f"${potential_revenue:,.2f}",
                        delta="Incremental"
                    )
                with col3:
                    st.metric(
                        label="📉 Budget Saved",
                        value=f"${savings:,.2f}",
                        delta=f"{total_count - target_count:,} emails avoided"
                    )
                with col4:
                    st.metric(
                        label="📈 Campaign ROI",
                        value=f"{roi:.1f}%",
                        delta="Estimated"
                    )
                
                # 7. VISUALIZATIONS
                st.divider()
                
                tab1, tab2, tab3 = st.tabs(["📊 Score Distribution", "🎯 Customer Segments", "💎 Top Opportunities"])
                
                with tab1:
                    # Enhanced histogram
                    fig1 = go.Figure()
                    
                    # Add histogram for each recommendation
                    for rec in df['Recommendation'].unique():
                        filtered_df = df[df['Recommendation'] == rec]
                        fig1.add_trace(go.Histogram(
                            x=filtered_df['Uplift_Score'],
                            name=rec,
                            opacity=0.7,
                            nbinsx=50
                        ))
                    
                    fig1.update_layout(
                        title="Distribution of Uplift Scores",
                        xaxis_title="Uplift Score",
                        yaxis_title="Number of Customers",
                        barmode='overlay',
                        hovermode='x unified',
                        height=400
                    )
                    
                    # Add threshold line
                    fig1.add_vline(
                        x=uplift_threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Threshold: {uplift_threshold}"
                    )
                    
                    st.plotly_chart(fig1, use_container_width=True)
                
                with tab2:
                    # Segment breakdown
                    segment_counts = df['Segment'].value_counts()
                    
                    fig2 = go.Figure(data=[go.Pie(
                        labels=segment_counts.index,
                        values=segment_counts.values,
                        hole=0.4,
                        marker=dict(colors=['#FF6B6B', '#FFA07A', '#98D8C8', '#6BCF7F'])
                    )])
                    
                    fig2.update_layout(
                        title="Customer Segmentation",
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Segment details table
                    st.write("**Segment Breakdown:**")
                    segment_summary = df.groupby('Segment').agg({
                        'Uplift_Score': ['count', 'mean'],
                        'Est_Revenue': 'sum'
                    }).round(3)
                    segment_summary.columns = ['Count', 'Avg Uplift', 'Total Revenue Potential']
                    st.dataframe(segment_summary, use_container_width=True)
                
                with tab3:
                    # Top 10 customers by uplift score
                    st.write("**Top 10 High-Value Targets:**")
                    top_customers = df.nlargest(10, 'Uplift_Score')[
                        ['Uplift_Score', 'Est_Revenue', 'recency', 'history', 'channel', 'Recommendation']
                    ].reset_index(drop=True)
                    
                    # Style the dataframe
                    st.dataframe(
                        top_customers.style.background_gradient(cmap='Greens', subset=['Uplift_Score', 'Est_Revenue']),
                        use_container_width=True
                    )
                
                # 8. DOWNLOAD RESULTS
                st.divider()
                st.subheader("📥 Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Filter only the people we should email
                    target_list = df[df['Recommendation'] == "✅ Send Email"].copy()
                    target_list_sorted = target_list.sort_values('Uplift_Score', ascending=False)
                    csv_target = target_list_sorted.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="📧 Download Target List (High Priority)",
                        data=csv_target,
                        file_name=f"targeted_customers_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True
                    )
                    st.caption(f"Contains {len(target_list):,} customers to contact")
                
                with col2:
                    # Full results with all scores
                    csv_full = df.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="📊 Download Full Analysis",
                        data=csv_full,
                        file_name=f"full_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    st.caption(f"Contains all {len(df):,} customers with scores")
                
                # Success message
                st.success("✅ Analysis complete! Review the insights above and download your targeted customer list.")
else:
    # Landing page when no file is uploaded
    st.info("👆 Upload a customer CSV file to begin analysis")
    
    st.write("### 📋 Required Data Format")
    
    sample_data = pd.DataFrame({
        'recency': [1, 6, 3],
        'history': [142.44, 329.08, 180.65],
        'zip_code': ['Surburban', 'Rural', 'Urban'],
        'channel': ['Phone', 'Web', 'Multichannel']
    })
    
    st.dataframe(sample_data, use_container_width=True)
    
    st.write("**Column Descriptions:**")
    st.markdown("""
    - **recency**: Months since last purchase
    - **history**: Total historical spend ($)
    - **zip_code**: Customer location category (Urban/Suburban/Rural)
    - **channel**: Preferred purchase channel (Phone/Web/Multichannel)
    """)

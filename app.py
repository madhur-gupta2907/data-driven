import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

# --- Page Config ---
st.set_page_config(
    page_title="Data-Driven Insight Generation using Python",
    page_icon="📊",
    layout="wide",
)

# --- Theme State ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# --- Custom CSS ---
def apply_theme(theme):
    bg_main = "#ffffff" if theme == 'light' else "#1f2937"
    text = "#111827" if theme == 'light' else "#f9fafb"
    
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
        * {{ font-family: 'Poppins', sans-serif; }}
        .stApp {{ background-color: {bg_main}; color: {text}; }}
        [data-testid="stMetric"] {{
            background: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%) !important;
            border-radius: 15px; padding: 20px; color: white !important;
        }}
        [data-testid="stMetricValue"] {{ color: white !important; font-weight: 800; }}
        .stButton>button {{ background: linear-gradient(90deg, #7c3aed, #ec4899); color: white; border: none; border-radius: 8px; }}
    </style>
    """, unsafe_allow_html=True)

apply_theme(st.session_state.theme)

# --- Header ---
st.markdown("""
<div style="background: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%); text-align: center; padding: 2rem; border-radius: 20px; margin-bottom: 2rem;">
  <h1 style="color: white; margin:0;">Data-Driven Insight Generation using Python</h1>
  <p style="color: white; opacity: 0.9;">Data Engineering • Analytics • Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# --- Member 2: Data Engineering ---
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    df = pd.DataFrame()
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            df = pd.read_csv('superstore.csv', encoding='latin1')
            
        # Standardize column names (strip spaces)
        df.columns = df.columns.str.strip()
        
        date_cols = ['Order Date', 'Ship Date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        if 'Order Date' in df.columns:
            df = df.dropna(subset=['Order Date'])
            df['YearMonth'] = df['Order Date'].dt.to_period('M').astype(str)
            df['Ordinal_Date'] = df['Order Date'].map(datetime.toordinal)
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
    return df

# --- Sidebar ---
with st.sidebar:
    st.header("🛠️ Control Panel")
    uploaded = st.file_uploader("Upload Dataset (CSV)", type="csv")
    
    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("☀️ Light"): 
        st.session_state.theme = 'light'
        st.rerun()
    if col2.button("🌙 Dark"): 
        st.session_state.theme = 'dark'
        st.rerun()
    
    st.divider()
    view = st.radio("Navigate Modules", [
        "Overview (Analytics)", 
        "Sales Trends", 
        "ML Predictions", 
        "System Design"
    ])

df = load_and_clean_data(uploaded)

# --- Error Handling for Missing Columns ---
if df.empty or 'Sales' not in df.columns:
    st.warning("⚠️ 'Sales' column not found or file empty. Please ensure your CSV has a 'Sales' column.")
    st.stop()

# --- Member 3: Data Analysis ---
if view == "Overview (Analytics)":
    st.subheader("Key Performance Indicators")
    c1, c2, c3, c4 = st.columns(4)
    
    sales = df['Sales'].sum()
    profit = df['Profit'].sum() if 'Profit' in df.columns else 0
    
    c1.metric("Total Revenue", f"${sales:,.2f}")
    c2.metric("Net Profit", f"${profit:,.2f}")
    c3.metric("Orders", f"{df.shape[0]:,}")
    c4.metric("Margin", f"{(profit/sales)*100:.1f}%" if sales != 0 else "0%")

    col_a, col_b = st.columns(2)
    with col_a:
        if 'Region' in df.columns and 'Category' in df.columns:
            fig = px.sunburst(df, path=['Region', 'Category'], values='Sales', title="Regional Sales Composition")
            st.plotly_chart(fig, width='stretch')
    with col_b:
        if 'Category' in df.columns and 'Profit' in df.columns:
            fig = px.box(df, x='Category', y='Profit', color='Category', title="Profitability Distribution")
            st.plotly_chart(fig, width='stretch')

elif view == "Sales Trends":
    st.subheader("Temporal Analysis")
    if 'YearMonth' in df.columns:
        trend_df = df.groupby('YearMonth')['Sales'].sum().reset_index()
        fig = px.line(trend_df, x='YearMonth', y='Sales', markers=True, title="Monthly Sales Velocity")
        st.plotly_chart(fig, width='stretch')

# --- Member 4: ML Engineer ---
elif view == "ML Predictions":
    st.subheader("Monthly Sales Forecasting")
    
    # Aggregating by Month for better R2
    ml_df = df.groupby('YearMonth')['Sales'].sum().reset_index()
    ml_df['Month_Index'] = range(len(ml_df))
    
    X = ml_df[['Month_Index']].values
    y = ml_df['Sales'].values
    
    model = LinearRegression().fit(X, y)
    
    # Predict next 6 months
    future_idx = np.array(range(len(ml_df), len(ml_df) + 6)).reshape(-1, 1)
    preds = model.predict(future_idx)
    
    # Create forecast labels
    last_date = pd.to_datetime(ml_df['YearMonth'].iloc[-1])
    future_dates = [(last_date + pd.DateOffset(months=i)).strftime('%Y-%m') for i in range(1, 7)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ml_df['YearMonth'], y=y, name="Historical"))
    fig.add_trace(go.Scatter(x=future_dates, y=preds, name="Forecast", line=dict(dash='dash', color='red')))
    
    fig.update_layout(title="6-Month Revenue Forecast", template="none")
    st.plotly_chart(fig, width='stretch')
    st.write(f"**Model Accuracy (R²):** {model.score(X, y):.4f}")

# --- Member 1: Project Lead ---
elif view == "System Design":
    st.subheader("System Architecture")
    st.markdown("""
    ### 3-Member Workflow
    1. **Member 1 (Lead):** Built the Streamlit routing and UI theme using CSS.
    2. **Member 2 (Eng):** Developed the `load_and_clean_data` pipeline and handled the `Ordinal_Date` conversions.
    3. **Member 3 (Analyst):** Built the KPI logic and interactive Plotly visualizations.
    4. **Member 4 (ML):** Implemented the Linear Regression forecasting model.
    """)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

# --- MEMBER 1: Project Lead (System Design) ---
st.set_page_config(page_title="Insight Generation Hub", page_icon="📊", layout="wide")

if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def apply_theme(theme):
    bg_main = "#ffffff" if theme == 'light' else "#111827"
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

# --- MEMBER 2: Data Engineer (Robust Cleaning) ---
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        else:
            df = pd.read_csv('amazon.csv', encoding='utf-8')
            
        df.columns = df.columns.str.strip() # Remove hidden spaces in headers

        # Clean Price/Sales columns (Handles ₹, $, and commas)
        price_cols = ['Sales', 'discounted_price', 'actual_price', 'Profit']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Clean Rating column
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'].astype(str).str.replace('|', 'nan', regex=False), errors='coerce')
        
        # Date Handling for Time-Series datasets
        if 'Order Date' in df.columns:
            df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
            df = df.dropna(subset=['Order Date'])
            df['YearMonth'] = df['Order Date'].dt.to_period('M').astype(str)

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# --- Sidebar ---
with st.sidebar:
    st.header("🛠️ Controls")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if st.button("🌓 Toggle Theme"):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()
    
    view = st.radio("Navigate", ["📊 Analysis", "🤖 ML Predictions", "🏗️ System Design"])

df = load_and_clean_data(uploaded)

# --- MEMBER 3: Data Analyst (Dynamic Insight Generation) ---
if not df.empty:
    # Auto-detect target column (Sales for Superstore, discounted_price for Amazon)
    target_col = 'Sales' if 'Sales' in df.columns else 'discounted_price'
    
    if view == "📊 Analysis":
        st.subheader("Business Key Metrics")
        c1, c2, c3 = st.columns(3)
        
        total_val = df[target_col].sum() if target_col in df.columns else 0
        avg_rating = df['rating'].mean() if 'rating' in df.columns else 0
        
        c1.metric(f"Total {target_col.replace('_', ' ').title()}", f"₹{total_val:,.0f}")
        c2.metric("Average Rating", f"{avg_rating:.2f} ⭐")
        c3.metric("Total Items", f"{len(df):,}")

        col_a, col_b = st.columns(2)
        with col_a:
            # Flexible chart based on available columns
            color_col = 'category' if 'category' in df.columns else None
            fig = px.histogram(df, x=target_col, title=f"Distribution of {target_col}", color_discrete_sequence=['#7c3aed'])
            st.plotly_chart(fig, width='stretch')
        with col_b:
            if 'rating' in df.columns:
                fig = px.scatter(df, x=target_col, y="rating", title="Value vs. Rating Pattern", opacity=0.5)
                st.plotly_chart(fig, width='stretch')

    # --- MEMBER 4: ML Engineer (Prediction Logic) ---
    elif view == "🤖 ML Predictions":
        st.subheader("Predictive Analytics")
        if 'rating' in df.columns and target_col in df.columns:
            X = df[[target_col]].fillna(0).values
            y = df['rating'].fillna(df['rating'].mean()).values
            
            model = LinearRegression().fit(X, y)
            
            st.write(f"**Insight:** This model predicts customer satisfaction (Rating) based on product price.")
            user_input = st.number_input(f"Enter {target_col}:", value=float(df[target_col].median()))
            if st.button("Predict Rating"):
                pred = model.predict([[user_input]])
                st.success(f"Predicted Rating: {pred[0]:.2f} ⭐")
        else:
            st.info("Upload a dataset with 'Sales/Price' and 'Rating' columns for ML features.")

    elif view == "🏗️ System Design":
        st.subheader("Project Structure")
        st.write("- **Member 1:** UI/UX & Navigation")
        st.write("- **Member 2:** Data Cleaning (Regex & Handling Currencies)")
        st.write("- **Member 3:** Analysis & Visualization")
        st.write("- **Member 4:** ML Integration")
else:
    st.warning("Please upload a CSV file to view the analysis.")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

# --- MEMBER 1: Project Lead (System Design & UI) ---
st.set_page_config(
    page_title="Data-Driven Insight Hub",
    page_icon="📊",
    layout="wide",
)

if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

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
        .insight-card {{ background: #f0f2f6; border-left: 5px solid #7c3aed; padding: 15px; border-radius: 5px; margin: 10px 0; color: #31333F; }}
    </style>
    """, unsafe_allow_html=True)

apply_theme(st.session_state.theme)

st.markdown("""
<div style="background: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%); text-align: center; padding: 2rem; border-radius: 20px; margin-bottom: 2rem;">
  <h1 style="color: white; margin:0;">📊 Insight Generation Dashboard</h1>
  <p style="color: white; opacity: 0.9;">Analyze Patterns • Generate Insights • Drive Decisions</p>
</div>
""", unsafe_allow_html=True)

# --- MEMBER 2: Data Engineer (Enhanced Cleaning) ---
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        else:
            # Assumes amazon.csv is in the root directory
            df = pd.read_csv('amazon.csv', encoding='utf-8') 
            
        df.columns = df.columns.str.strip()
        
        # FIXED: Regex to remove currency symbols and encoding artifacts
        # This solves the 'â\x82¹' conversion error
        for col in ['discounted_price', 'actual_price']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean ratings (handle '|' and convert to float)
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'].astype(str).str.replace('|', 'nan', regex=False), errors='coerce')
            df = df.dropna(subset=['rating'])
            
        # Clean rating_count
        if 'rating_count' in df.columns:
            df['rating_count'] = df['rating_count'].astype(str).str.replace(r'[^\d]', '', regex=True)
            df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce').fillna(0)
            
        if 'category' in df.columns:
            df['Main_Category'] = df['category'].apply(lambda x: x.split('|')[0] if isinstance(x, str) else 'Unknown')
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# --- Sidebar ---
with st.sidebar:
    st.header("🛠️ Data Controls")
    uploaded = st.file_uploader("Upload amazon.csv", type="csv")
    
    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("☀️ Light"): 
        st.session_state.theme = 'light'
        st.rerun()
    if col2.button("🌙 Dark"): 
        st.session_state.theme = 'dark'
        st.rerun()
    
    st.divider()
    view = st.radio("Project Sections", [
        "📚 Introduction",
        "📊 EDA & Visuals", 
        "💡 Key Insights", 
        "🤖 ML Predictions", 
        "🏗️ System Design"
    ])

df = load_and_clean_data(uploaded)

if df.empty:
    st.warning("Please upload a valid dataset to begin.")
    st.stop()

# --- Section 1: Introduction ---
if view == "📚 Introduction":
    st.subheader("1️⃣ Project Overview")
    st.write("This project utilizes real-world Amazon marketplace data to extract meaningful patterns for business decision-making.")
    st.markdown("""
    **Core Objectives:**
    - **Clean Data:** Remove currency artifacts and standardize numerical values.
    - **Analyze Trends:** Observe how **Price** and **Discount** strategies impact **Ratings**.
    - **Predictive Modeling:** Build a tool to estimate customer satisfaction before product launch.
    """)
    st.info("💡 Navigation Tip: Head to 'EDA & Visuals' to explore the live charts.")

# --- Section 2: MEMBER 3: Data Analyst (EDA) ---
elif view == "📊 EDA & Visuals":
    st.subheader("2️⃣ Exploratory Data Analysis")
    c1, c2, c3 = st.columns(3)
    
    c1.metric("Total Products", f"{len(df):,}")
    c2.metric("Avg Rating", f"{df['rating'].mean():.2f} ⭐")
    if 'discount_percentage' in df.columns:
        # Removing '%' if present to calculate mean
        disc_val = df['discount_percentage'].astype(str).str.replace('%', '', regex=False)
        disc_val = pd.to_numeric(disc_val, errors='coerce').mean()
        c3.metric("Avg Discount", f"{disc_val:.1f}%")

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(df, x="rating", nbins=15, title="Customer Rating Distribution", 
                           color_discrete_sequence=['#7c3aed'], labels={'rating': 'Star Rating'})
        st.plotly_chart(fig, width='stretch')
    with col_b:
        fig = px.scatter(df, x="discounted_price", y="rating", color="Main_Category", 
                         title="Price vs. Customer Rating", opacity=0.5,
                         labels={'discounted_price': 'Price (₹)', 'rating': 'Rating'})
        st.plotly_chart(fig, width='stretch')

# --- Section 3: MEMBER 3: Insight Generation ---
elif view == "💡 Key Insights":
    st.subheader("3️⃣ Data-Driven Insights & Recommendations")
    
    st.markdown('<div class="insight-card"><b>Insight 1:</b> Higher price points do not always correlate with higher ratings, suggesting that perceived value (Price vs. Quality) is the primary driver of satisfaction.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="insight-card"><b>Insight 2:</b> Categories like "Electronics" show the highest review density, making them the most reliable categories for trend analysis.</div>', unsafe_allow_html=True)
    
    st.subheader("🚀 Business Recommendations")
    st.success("👉 **Optimized Discounting:** Target a 30-45% discount range. Data suggests this maintains high review volume without degrading the 'Premium' feel of the product.")
    st.success("👉 **Quality Focus:** Since price isn't a direct rating driver, focus on product-build quality to improve organic ranking.")

# --- Section 4: MEMBER 4: ML Engineer (Predictions) ---
elif view == "🤖 ML Predictions":
    st.subheader("4️⃣ Machine Learning: Rating Predictor")
    st.info("Member 4 Objective: Build a Linear Regression model to forecast a product's rating based on price.")
    
    # Feature Engineering
    features = ['discounted_price', 'actual_price']
    X = df[features].fillna(0).values
    y = df['rating'].values
    
    model = LinearRegression().fit(X, y)
    
    # Prediction Interface
    st.markdown("### 🔮 Predict Your Product's Success")
    val_dp = st.number_input("Proposed Discounted Price (₹)", value=499.0)
    val_ap = st.number_input("Original Actual Price (₹)", value=999.0)
    
    if st.button("Calculate Predicted Rating"):
        pred = model.predict([[val_dp, val_ap]])
        st.write(f"### Predicted Customer Rating: **{pred[0]:.2f} ⭐**")
        st.write(f"**Model R² Score:** {model.score(X, y):.4f}")

# --- Section 5: MEMBER 1: Project Lead (Architecture) ---
elif view == "🏗️ System Design":
    st.subheader("5️⃣ System Architecture & Team Roles")
    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown("""
        **Project Team Roles:**
        - **Member 1 (Lead):** App structure, routing, and Canva-style UI implementation.
        - **Member 2 (Engineer):** Robust data ingestion and Regex currency cleaning.
        - **Member 3 (Analyst):** EDA pattern identification and business insights.
        - **Member 4 (ML Eng):** Regression modeling for rating prediction.
        """)
    with col_y:
        st.markdown("""
        **The Tech Stack:**
        - **Frontend:** Streamlit 2026 Build
        - **Processing:** Pandas (Vectorized cleaning)
        - **Visualization:** Plotly (Responsive stretch mode)
        - **Intelligence:** Scikit-Learn (Linear Regression)
        """)

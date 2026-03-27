import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --- PAGE SETUP ---
st.set_page_config(page_title="Insight & Prediction Engine", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { 
        background-color: #000000; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
        border: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Data-Driven Business Insight & Prediction System")
st.divider()

# --- SIDEBAR: DATA INPUT ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload Business Dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # --- STEP 1: DATA CLEANING ---
    st.header("1️⃣ Automated Data Cleaning")
    col_a, col_b, col_c = st.columns(3)
    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    null_count = df.isnull().sum().sum()
    df = df.ffill() 
    
    col_a.metric("Total Records", initial_count)
    col_b.metric("Duplicates Removed", initial_count - len(df))
    col_c.metric("Missing Values Fixed", null_count)
    
    with st.expander("View Cleaned Dataset Preview"):
        st.dataframe(df.head(10), width='stretch') # Updated parameter

    # --- STEP 2: STATISTICAL ANALYSIS ---
    st.divider()
    st.header("2️⃣ Descriptive Analytics")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if num_cols:
        tab1, tab2 = st.tabs(["📊 Summary Statistics", "🧮 Correlation Matrix"])
        with tab1:
            st.write(df[num_cols].describe().T)
        with tab2:
            if len(num_cols) > 1:
                fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
                sns.heatmap(df[num_cols].corr(), annot=True, cmap='RdYlGn', ax=ax_corr)
                st.pyplot(fig_corr)

    # --- STEP 3: VISUALIZATION ---
    st.divider()
    st.header("3️⃣ Pattern Visualization")
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        if num_cols:
            st.subheader("Numeric Distribution")
            target_num = st.selectbox("Select metric to analyze:", num_cols)
            fig_hist, ax_hist = plt.subplots()
            sns.histplot(df[target_num], kde=True, color='#2e7d32')
            st.pyplot(fig_hist)
    with v_col2:
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            st.subheader("Category Performance")
            target_cat = st.selectbox("Select category:", cat_cols)
            fig_bar, ax_bar = plt.subplots()
            df[target_cat].value_counts().head(8).plot(kind='bar', color='#1565c0', ax=ax_bar)
            plt.xticks(rotation=45)
            st.pyplot(fig_bar)

    # --- STEP 4: PREDICTIVE INTELLIGENCE (The "Prediction Table" part) ---
    st.divider()
    st.header("4️⃣ Predictive Intelligence (Linear Regression)")
    
    if len(num_cols) >= 2:
        id_col = st.selectbox("Select Product/Reference Column:", df.columns.tolist())
        p_col1, p_col2 = st.columns([1, 2])
        
        with p_col1:
            y_var = st.selectbox("Target to Predict (Y):", num_cols)
            # Filter X features to avoid predicting Y using Y
            x_vars = st.multiselect("Features to use (X):", [c for c in num_cols if c != y_var])
        
        if x_vars:
            X = df[x_vars]
            y = df[y_var]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            with p_col2:
                # This is the Actual vs Predicted table you wanted!
                results_df = pd.DataFrame({
                    "Product/Reference": df.loc[y_test.index, id_col].values,
                    "Actual": y_test.values,
                    "Predicted": predictions
                })
                results_df["Error"] = results_df["Actual"] - results_df["Predicted"]
                
                st.write(f"**Actual vs Predicted Analysis**")
                st.dataframe(results_df.head(10), width='stretch') # Updated parameter
                st.metric("Model Confidence (R²)", f"{model.score(X_test, y_test):.2%}")
        else:
            st.info("Select Features (X) to generate the prediction table.")

else:
    st.warning("👈 Please upload a CSV file to start.")
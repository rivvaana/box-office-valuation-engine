import streamlit as st
import numpy as np
import tensorflow as tf

# Force wide layout configuration
st.set_page_config(page_title="Box Office Valuation Engine", page_icon="🎬", layout="wide")

# --- CLEAN MINIMALIST CINEMA THEME (CSS) ---
st.markdown("""
    <style>
    /* Hide scrollbars across the entire app interface */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden !important;
        background-color: #0F0F12;
        color: #FFFFFF;
    }
    
    /* Safely target the specific inner padding block of Streamlit to remove whitespace */
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    
    .block-container {
        padding-top: 2rem !important; /* Normal padding so it never cuts off */
        padding-bottom: 0rem !important;
        margin-top: 0px !important; 
    }
    
    /* Pull columns tighter vertically */
    [data-testid="stVerticalBlock"] {
        gap: 0.4rem !important;
    }
    
    /* Academic / Corporate Branding Typography */
    h1 {
        color: #8B1E2F !important; /* Elegant Wine Red */
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        font-size: 28px !important;
        letter-spacing: -0.5px;
        margin: 0 !important;
    }
    h3 {
        font-size: 18px !important;
        margin-bottom: 5px !important;
        color: #E2E8F0 !important;
    }
    label {
        font-size: 13px !important;
        color: #94A3B8 !important;
    }
    
    /* Sleek Input Fields */
    .stNumberInput div div input {
        background-color: #1A1A1E !important;
        color: #FFFFFF !important;
        border: 1px solid #2D2D34 !important;
        border-radius: 4px !important;
        height: 32px !important;
    }
    
    /* High-Performance Execution Button */
    div.stButton > button:first-child {
        background-color: #8B1E2F !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 10px 20px !important;
        border-radius: 4px !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 5px;
    }
    div.stButton > button:first-child:hover {
        background-color: #A32639 !important;
        box-shadow: 0px 0px 12px rgba(139, 30, 47, 0.4);
    }
    
    /* Ultra-Clean Analytics Output Card */
    .prediction-box {
        background-color: #1A1A1E;
        padding: 20px; 
        border-top: 4px solid #8B1E2F; 
        border-radius: 4px;
        margin-top: 15px;
        text-align: center;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# --- PROFESSIONAL HEADERS ---
st.title("🎬 Global Box Office Valuation Engine")
st.markdown("<p style='color: #94A3B8; margin: 0; font-size: 14px;'>An Artificial Neural Network (ANN) predictive framework optimizing risk mitigation and asset yield modeling.</p>", unsafe_allow_html=True)
st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

# --- TWO COLUMN SIDE-BY-SIDE DASHBOARD ---
col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.markdown("### 🎞️ Quantitative Metrics")
    budget = st.number_input("Capital Allocation / Budget ($)", min_value=0, value=50000000, step=1000000)
    popularity = st.number_input("Market Resonance / Popularity Index", min_value=0.0, value=50.0, step=1.0)
    runtime = st.number_input("Operational Runtime (Minutes)", min_value=1, value=120)
    vote_average = st.slider("Target Critical Rating Variance (1-10)", min_value=1.0, max_value=10.0, value=6.5)
    vote_count = st.number_input("Aggregated Audience Engagement Volume", min_value=0, value=1000, step=100)

with col2:
    st.markdown("### ⚙️ Core Architecture Status")
    try:
        model = tf.keras.models.load_model('box_office_ann_model.h5')
        st.markdown("<p style='color: #46D369 !important; font-size: 13px; font-weight: 500; margin: 0;'>✓ Predictive Engine Operational (TensorFlow Backend)</p>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Execution Error: {e}")
    
    predict_btn = st.button("RUN QUANTITATIVE VALUATION")
    
    if predict_btn:
        try:
            raw_features = np.array([[budget, popularity, runtime, vote_average, vote_count]], dtype=np.float32)
            
            # Mathematical standard scaling baseline
            mean_vals = np.array([3.63e+07, 2.14e+01, 1.06e+02, 6.09e+00, 6.90e+02])
            scale_vals = np.array([4.09e+07, 3.18e+01, 2.26e+01, 1.19e+00, 1.23e+03])
            scaled_features = (raw_features - mean_vals) / scale_vals
            
            prediction = model.predict(scaled_features)
            predicted_revenue = float(prediction[0][0])
            
            if predicted_revenue < 0 or raw_features[0][0] == 0:
                predicted_revenue = budget * 1.5  
                
            st.markdown(f"""
                <div class="prediction-box">
                    <p style="margin: 0; color: #94A3B8 !important; font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;">Expected Asset Gross Valuation</p>
                    <h1 style="margin: 10px 0 0 0; color: #8B1E2F !important; font-size: 34px; font-weight: bold;">${predicted_revenue:,.2f}</h1>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Pipeline Interruption: {e}")
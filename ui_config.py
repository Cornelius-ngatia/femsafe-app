import streamlit as st

def apply_femsafe_theme():
    st.set_page_config(
        page_title="FemSafe Risk Detector",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
        .stProgress > div > div {
            background-color: #ffcc00;
        }
        .stMarkdown h3 {
            color: #d6336c;
        }
        </style>
    """, unsafe_allow_html=True)

    # ✅ Web-hosted placeholder logo
    st.image("https://via.placeholder.com/120x120.png?text=FemSafe+Logo", width=120)
    st.markdown("### 🔍 **FemSafe Risk Detector**")
    st.markdown("_Built for high-recall GBV/femicide risk triage._")

def render_sidebar():
    with st.sidebar:
        st.header("📘 About FemSafe")
        st.markdown("""
        FemSafe is an AI-powered tool for detecting gender-based violence and femicide risk in text reports.

        **Features:**
        - Emoji-aware input
        - Recall-optimized predictions
        - Panic response trigger
        - SHAP-based transparency

        _Built for ethical triage and stakeholder trust._
        """)
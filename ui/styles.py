import streamlit as st

def apply_custom_css():
    st.markdown("""
    <style>
    /* Modern Premium UI Styling */
    
    /* Global Adjustments */
    .stApp {
        background-color: var(--background-color);
        font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Metrics / Status Badges */
    .metric-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 10px 0;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 1rem;
        font-weight: 500;
        color: var(--text-color);
        opacity: 0.8;
    }
    
    /* Custom Badges */
    .badge {
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-high { background-color: rgba(239, 68, 68, 0.1); color: #ef4444; border: 1px solid #ef4444; }
    .badge-medium { background-color: rgba(245, 158, 11, 0.1); color: #f59e0b; border: 1px solid #f59e0b; }
    .badge-low { background-color: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid #10b981; }
    
    /* Upload Section Enhancements */
    .stFileUploader > div > div {
        border-radius: 16px !important;
        border: 2px dashed rgba(139, 92, 246, 0.5) !important;
        background-color: rgba(139, 92, 246, 0.02) !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #8b5cf6 !important;
        background-color: rgba(139, 92, 246, 0.05) !important;
    }

    /* DataFrame styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

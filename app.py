import streamlit as st
st.set_page_config(
    page_title="EDA App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
#from chatbot import chatbot_ui
from extraction import extraction_ui
from image_analysis import render_image_analysis_section
from nlp import render_nlp_section
from sqlq import sqlq_ui
from time_series import render_time_series_section
import tracker
from visualizations import render_visualizations_section
import gc
import time
import os
import pandas as pd
import numpy as np

# Import modules with fixed implementations
from ml_integration import render_ml_section
from data_loading import home_ui
from data_cleaning import render_data_cleaning_section
# Import utilities for isolated session state management
from utils.session_state_manager import get_session_manager
from utils.memory_manager import MemoryManager

# Get session manager instance
session_manager = get_session_manager()

# Define a global rainbow color palette with more vibrant colors
RAINBOW_COLORS = [
    "#FF355E", "#FD5B78", "#FF6037", "#FF9966",
    "#FF9933", "#FFCC33", "#FFFF66", "#CCFF00",
    "#66FF66", "#50BFE6", "#FF6EFF", "#EE34D2"
]

# Initialize session state with comprehensive defaults
def initialize_session_state():
    """Initialize all global session state variables with proper defaults"""
    defaults = {
        'x_col': None,
        'y_col': None,
        'plot_type': "Scatter Plot",
        'pred_x_col': None,
        'pred_y_col': None,
        'pred_plot_type': "Line Plot",
        'is_structured': True,
        'current_section': "Home",
        'last_section': None,
        'transition_start': False,
        'df': None,
        'cleaned_df': None,
        'text_data': None,
        'pdf_text': None,
        'performance_mode': False,
        'uploaded_file_name': None,
        'uploaded_file': None,
        'file_uploader_key': 0,  # Used to force reset the file uploader
        'chat_history': [],
        'selected_section': "Home",
        'file_processed': False,
        'image_data': None,
        'pdf_images': None,
        'tab_states': {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

if 'tracker' not in st.session_state:
    st.session_state.tracker = tracker.AppTracker()
tracker = st.session_state.tracker

# Add advanced custom CSS with animations and effects
st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Montserrat:wght@400;600;700&display=swap');

    /* Global Styles */
    :root {
        --primary: #6B5B95;
        --secondary: #88B04B;
        --accent: #FF6B6B;
        --background: #f8f9fa;
        --card-bg: #ffffff;
        --text: #2d3436;
        --text-light: #636e72;
    }

    body {
        font-family: 'Poppins', sans-serif;
        background-color: var(--background);
        color: var(--text);
        margin: 0;
        padding: 0;
    }

    /* Header Box with Gradient Animation */
    .header-box {
        background: linear-gradient(135deg, #6B5B95, #88B04B, #FF6B6B, #6B5B95);
        background-size: 300% 300%;
        color: white;
        padding: 25px;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        border-radius: 15px;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
        font-family: 'Montserrat', sans-serif;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
        animation: gradientBG 15s ease infinite;
        transition: all 0.4s ease;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .header-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.15);
    }

    /* Section Transition Animation */
    .section-transition {
        animation: fadeIn 0.5s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Card Styling */
    .card {
        background-color: var(--card-bg);
        border-radius: 12px;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.08);
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        border-left: 4px solid var(--primary);
    }

    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.12);
        border-left: 4px solid var(--accent);
    }

    /* Button Styling */
    .stButton>button {
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
        background: linear-gradient(135deg, var(--secondary), var(--primary)) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header with improved design
st.markdown(
    """
    <div class="header-box">
        <div style="font-size: 1.2rem; margin-bottom: 5px; font-weight: 400;">✨ Automate Your EDA Workflow</div>
        <div style="font-size: 1.5rem;">Advanced Data Analysis Tool</div>
        <div style="font-size: 0.9rem; margin-top: 10px; font-weight: 300;">By <span style="color: #FFD700; font-weight: 600;">Yuvraj Kumar Gond</span></div>
    </div>
    """,
    unsafe_allow_html=True
)
# Performance optimization toggle
with st.sidebar.expander("🚀 Performance Mode"):
    st.session_state.performance_mode = st.toggle("Enable Performance Mode",
                                                  value=st.session_state.get('performance_mode', False))
    if st.session_state.performance_mode:
        st.info("Performance mode reduces animations for faster processing.")
with st.sidebar.expander("🔄 Refresh"):
    if st.button("Refresh", help="Refresh the entire application and clear session state"):
        session_manager.clear_all_data()  # Clear section-specific data
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if st.button("🧹 Clear All Data", help="Remove all uploaded files and reset the app"):
    # Clear all section data
        session_manager.clear_all_data()
    
    # Clear Streamlit cache
        st.cache_data.clear()
        st.cache_resource.clear()
    
    # Run garbage collection
        gc.collect()
        st.rerun()
        
    # Display memory usage information
    MemoryManager.display_memory_usage()

# Sidebar Navigation with improved design
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <style>
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #6B5B95;
        margin-bottom: 1px;
        text-align: center;
    }
    </style>
    <div class="sidebar-header">Navigation</div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

# Define the sections with icons (limited set for this fixed version)
sections = [
    {"name": "Home", "icon": "🏠"},
    {"name": "Data Cleaning", "icon": "🧹"},
    {"name": "Visualizations", "icon": "📈"},
    {"name": "Image Analysis", "icon": "🖼️"},
    {"name": "Time Series", "icon": "⏳"},
    {"name": "Machine Learning", "icon": "🤖"},
    {"name": "Chatbot", "icon": "💬"},
    {"name": "NLP", "icon": "🔤"},
    {"name": "SQL Query", "icon": "🗄️"},
    {"name": "Extraction", "icon": "📤"}    
]

# Create navigation buttons with icons
for section in sections:
    if st.sidebar.button(f"{section['icon']} {section['name']}", key=f"nav_{section['name']}"):
        st.session_state.last_section = st.session_state.selected_section
        st.session_state.selected_section = section["name"]
        st.session_state.transition_start = True

# Add a footer to the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="text-align: center; font-size: 0.9rem; color: #6c757d;">
        <p><strong>Advanced EDA Tool</strong></p>
        <p>Version: <b>2.0</b></p>
        <p>By</p>
        <p><b>Yuvraj Kumar Gond</b></p>
        <p>© 2025 All Rights Reserved</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Display the selected section with transition effect
section_container = st.container()

with section_container:
    if st.session_state.get('transition_start', False):
        st.markdown('<div class="new-section">', unsafe_allow_html=True)

    # Home Section
    if 'selected_section' in st.session_state and st.session_state.selected_section == "Home":
        tracker.log_section("Home")
        with st.container():
            home_ui()

    # Data Cleaning Section
    if 'selected_section' in st.session_state and st.session_state.selected_section == "Data Cleaning":
        tracker.log_section("Data Cleaning")
        with st.container():
            render_data_cleaning_section()
            
    # Visualizations Section
    if 'selected_section' in st.session_state and st.session_state.selected_section == "Visualizations":
        tracker.log_section("Visualizations")
        with st.container():
            if 'df' in st.session_state:
                df = st.session_state.df
                render_visualizations_section(df)
            else:
                st.info("Please upload and load a dataset first to use visualization features.")

    # Image Analysis Section
    if 'selected_section' in st.session_state and st.session_state.selected_section == "Image Analysis":
        tracker.log_section("Image Analysis")
        with st.container():
            render_image_analysis_section()
    

# Then in your section rendering logic (where you handle selected_section)
    if 'selected_section' in st.session_state and st.session_state.selected_section == "Machine Learning":
        tracker.log_section("Machine Learning")
        with st.container():
            render_ml_section()
           
    
    # NLP Section   
    if 'selected_section' in st.session_state and st.session_state.selected_section == "NLP":
        tracker.log_section("NLP")
        with st.container():
            render_nlp_section()
    
    # Chatbot Section        
    '''if 'selected_section' in st.session_state and st.session_state.selected_section == "Chatbot":
        with st.container():
            chatbot_ui() '''
    
    # SQL Section
    if 'selected_section' in st.session_state and st.session_state.selected_section == "SQL Query":
        tracker.log_section("SQL Query")
        with st.container():
            sqlq_ui()
            
    # Extraction Section
    if 'selected_section' in st.session_state and st.session_state.selected_section == "Extraction":
        tracker.log_section("Extraction")
        with st.container():
            extraction_msg = st.empty()
            extraction_ui()
    
    # Time Series Section
    if 'selected_section' in st.session_state and st.session_state.selected_section == "Time Series":
        tracker.log_section("Time Series")
        with st.container():
            render_time_series_section()   
    
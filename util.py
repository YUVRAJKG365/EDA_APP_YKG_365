import streamlit as st

def tab_manager(tab_labels: list, section_name: str):
    """Improved tab manager with better state persistence"""
    # Initialize tab state for this section
    if section_name not in st.session_state.tab_states:
        st.session_state.tab_states[section_name] = 0

    # Create tabs
    tabs = st.tabs(tab_labels)
    
    # Track active tab without using buttons
    active_index = st.session_state.tab_states[section_name]
    
    # Update session state if tabs are clicked
    if 'tab_clicked' in st.session_state and st.session_state.tab_clicked:
        st.session_state.tab_states[section_name] = st.session_state.tab_clicked
        st.session_state.tab_clicked = None
        st.rerun()
    
    return active_index, tabs
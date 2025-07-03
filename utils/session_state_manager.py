"""
Session State Manager - Handles isolated session state for different app sections
This utility prevents data leakage between different sections of the application
by maintaining section-specific session state variables.
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, Union
import uuid
import time
import gc

class SessionStateManager:
    """
    Manages isolated session states for different app sections.
    Prevents data leakage between sections by maintaining separate data stores
    for each section of the application.
    """
    
    def __init__(self):
        """Initialize session state manager and create section data stores"""
        # Initialize the main container if it doesn't exist
        if 'section_data_stores' not in st.session_state:
            st.session_state.section_data_stores = {}
            
        # Initialize section-specific store for each app section
        sections = [
            "Home", "Data Cleaning", "Visualizations", "Image Analysis",
            "Machine Learning", "Chatbot", "NLP", "SQL Query", 
            "Extraction", "Time Series", "Security", "Performance"
        ]
        
        for section in sections:
            if section not in st.session_state.section_data_stores:
                st.session_state.section_data_stores[section] = {
                    'df': None,
                    'cleaned_df': None,
                    'text_data': None,
                    'pdf_data': None,
                    'image_data': None,
                    'pdf_images': None,
                    'uploaded_file': None,
                    'uploaded_file_name': None,
                    'file_processed': False,
                    'file_uploader_key': str(uuid.uuid4()),
                    'metadata': {},
                    'last_access': time.time()
                }
                
    def get_section_store(self, section: str) -> Dict[str, Any]:
        """
        Get the data store for a specific section
        Args:
            section: The name of the section
        Returns:
            Dictionary containing section-specific data
        """
        if section not in st.session_state.section_data_stores:
            st.session_state.section_data_stores[section] = {
                'df': None,
                'cleaned_df': None,
                'text_data': None,
                'pdf_data': None,
                'image_data': None,
                'pdf_images': None,
                'uploaded_file': None,
                'uploaded_file_name': None,
                'file_processed': False,
                'file_uploader_key': str(uuid.uuid4()),
                'metadata': {},
                'last_access': time.time()
            }
            
        # Update last access time
        st.session_state.section_data_stores[section]['last_access'] = time.time()
        return st.session_state.section_data_stores[section]
    
    def set_data(self, section: str, key: str, value: Any) -> None:
        """
        Set data in a specific section's store
        Args:
            section: The name of the section
            key: The key for the data
            value: The value to store
        """
        section_store = self.get_section_store(section)
        section_store[key] = value
        
    def get_data(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get data from a specific section's store
        Args:
            section: The name of the section
            key: The key for the data
            default: Default value to return if key doesn't exist
        Returns:
            The value associated with the key or default if not found
        """
        section_store = self.get_section_store(section)
        return section_store.get(key, default)
    
    def has_data(self, section: str, key: str) -> bool:
        """
        Check if a key exists in a section's store and has a non-None value
        Args:
            section: The name of the section
            key: The key to check
        Returns:
            True if key exists and has a value, False otherwise
        """
        section_store = self.get_section_store(section)
        return key in section_store and section_store[key] is not None
    
    def clear_section_data(self, section: str) -> None:
        """
        Clear all data for a specific section
        Args:
            section: The name of the section to clear
        """
        if section in st.session_state.section_data_stores:
            # Generate a new file uploader key
            file_uploader_key = str(uuid.uuid4())
            
            # Reset the section store
            st.session_state.section_data_stores[section] = {
                'df': None,
                'cleaned_df': None,
                'text_data': None,
                'pdf_data': None,
                'image_data': None,
                'pdf_images': None,
                'uploaded_file': None,
                'uploaded_file_name': None,
                'file_processed': False,
                'file_uploader_key': file_uploader_key,
                'metadata': {},
                'last_access': time.time()
            }
            
            # Force garbage collection to free up memory
            gc.collect()
    
    def clear_all_data(self) -> None:
        """Clear all data from all section stores"""
        sections = list(st.session_state.section_data_stores.keys())
        for section in sections:
            self.clear_section_data(section)
    
    def get_file_uploader_key(self, section: str) -> str:
        """
        Get the file uploader key for a specific section
        This ensures each section has its own file uploader instance
        Args:
            section: The name of the section
        Returns:
            Unique file uploader key for the section
        """
        section_store = self.get_section_store(section)
        return section_store['file_uploader_key']
    
    def reset_file_uploader(self, section: str) -> None:
        """
        Reset the file uploader for a specific section
        This forces Streamlit to create a new file uploader instance
        Args:
            section: The name of the section
        """
        section_store = self.get_section_store(section)
        section_store['file_uploader_key'] = str(uuid.uuid4())
    
    def get_dataframe(self, section: str, use_cleaned: bool = False) -> Optional[pd.DataFrame]:
        """
        Get the dataframe for a specific section with option to use cleaned version
        Args:
            section: The name of the section
            use_cleaned: Whether to return the cleaned dataframe if available
        Returns:
            DataFrame or None if not available
        """
        section_store = self.get_section_store(section)
        
        if use_cleaned and section_store['cleaned_df'] is not None:
            return section_store['cleaned_df']
        
        return section_store['df']
    
    def get_text_data(self, section: str) -> Optional[str]:
        """
        Get the text data for a specific section
        Args:
            section: The name of the section
        Returns:
            Text data or None if not available
        """
        section_store = self.get_section_store(section)
        return section_store['text_data']
    
    def get_file_info(self, section: str) -> Dict[str, Any]:
        """
        Get file information for a specific section
        Args:
            section: The name of the section
        Returns:
            Dictionary with file information
        """
        section_store = self.get_section_store(section)
        return {
            'file_processed': section_store['file_processed'],
            'uploaded_file_name': section_store['uploaded_file_name'],
            'has_dataframe': section_store['df'] is not None,
            'has_text_data': section_store['text_data'] is not None,
            'has_image_data': section_store['image_data'] is not None,
            'has_pdf_data': section_store['pdf_data'] is not None
        }
    
    def update_metadata(self, section: str, key: str, value: Any) -> None:
        """
        Update metadata for a specific section
        Args:
            section: The name of the section
            key: Metadata key
            value: Metadata value
        """
        section_store = self.get_section_store(section)
        if 'metadata' not in section_store:
            section_store['metadata'] = {}
        section_store['metadata'][key] = value
    
    def get_metadata(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get metadata for a specific section
        Args:
            section: The name of the section
            key: Metadata key
            default: Default value if key not found
        Returns:
            Metadata value or default
        """
        section_store = self.get_section_store(section)
        if 'metadata' not in section_store:
            return default
        return section_store['metadata'].get(key, default)

# Create a singleton instance of the session state manager
def get_session_manager():
    """Get or create the SessionStateManager instance"""
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionStateManager()
    return st.session_state.session_manager
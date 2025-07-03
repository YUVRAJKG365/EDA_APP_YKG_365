"""
Memory Manager - Handles memory optimization and monitoring
for efficient processing of large datasets
"""

import streamlit as st
import pandas as pd
import numpy as np
import gc
import psutil
import time
from typing import Dict, Any, Optional, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os

class MemoryManager:
    """
    Manages memory usage optimization and monitoring
    to improve performance with large datasets and prevent OOM errors
    """
    
    @staticmethod
    def get_memory_usage():
        """
        Get current memory usage
        
        Returns:
            Dict containing memory usage statistics
        """
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024 * 1024 * 1024),
            "available_gb": mem.available / (1024 * 1024 * 1024),
            "used_gb": mem.used / (1024 * 1024 * 1024),
            "percent_used": mem.percent
        }
    
    @staticmethod
    def display_memory_usage():
        """Display memory usage information in the app"""
        mem_usage = MemoryManager.get_memory_usage()
        
        st.sidebar.markdown("### 💾 Memory Usage")
        
        # Format memory values
        total = f"{mem_usage['total_gb']:.1f} GB"
        used = f"{mem_usage['used_gb']:.1f} GB"
        available = f"{mem_usage['available_gb']:.1f} GB"
        
        # Create a progress bar for memory usage
        st.sidebar.progress(mem_usage["percent_used"] / 100)
        
        # Display memory statistics
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Used", used)
        col2.metric("Available", available)
        
        # Display a warning if memory usage is high
        if mem_usage["percent_used"] > 80:
            st.sidebar.warning("⚠️ High memory usage. Consider clearing unused data.")
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Memory-optimized DataFrame
        """
        start_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Optimize integer types
        for col in df.select_dtypes(include=['int']):
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Convert to smallest possible integer type
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        # Optimize float types
        for col in df.select_dtypes(include=['float']):
            # Check if column can be represented as float32 without losing precision
            if df[col].round(6).eq(df[col].astype('float32').round(6)).all():
                df[col] = df[col].astype(np.float32)
        
        # Optimize object types
        for col in df.select_dtypes(include=['object']):
            # Check if it's a candidate for categorical
            if df[col].nunique() / len(df) < 0.5:  # 50% unique threshold
                df[col] = df[col].astype('category')
        
        end_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
        reduction = 100 * (1 - end_mem/start_mem)
        
        st.info(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)")
        
        return df
    
    @staticmethod
    def force_garbage_collection():
        """Force garbage collection to free memory"""
        before = MemoryManager.get_memory_usage()
        gc.collect()
        after = MemoryManager.get_memory_usage()
        
        freed_mem = after["available_gb"] - before["available_gb"]
        if freed_mem > 0:
            st.success(f"Memory cleaned: {freed_mem:.2f} GB freed")
        
        return freed_mem
    
    @staticmethod
    @st.cache_data
    def process_dataframe_in_chunks(df: pd.DataFrame, chunk_size: int = 10000, 
                                  operation=None):
        """
        Process a large DataFrame in chunks to reduce memory usage
        
        Args:
            df: DataFrame to process
            chunk_size: Number of rows to process at once
            operation: Function to apply to each chunk
            
        Returns:
            Processed DataFrame or list of results
        """
        results = []
        n_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size > 0 else 0)
        
        # Create progress display
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(df))
                chunk = df.iloc[start_idx:end_idx].copy()
                
                # Apply operation if provided
                if operation is not None:
                    result = operation(chunk)
                    results.append(result)
                else:
                    results.append(chunk)
                
                # Update progress
                progress = (i + 1) / n_chunks
                progress_bar.progress(progress)
                status_text.text(f"Processing chunk {i+1}/{n_chunks} ({int(progress*100)}%)")
                
                # Force garbage collection between chunks
                if i % 5 == 0:  # Every 5 chunks
                    gc.collect()
            
            # Combine results based on their type
            if isinstance(results[0], pd.DataFrame):
                final_result = pd.concat(results, ignore_index=True)
                return final_result
            else:
                return results
                
        finally:
            progress_bar.empty()
            status_text.empty()
    
    @staticmethod
    def parallel_process(data, func, n_workers=None, use_threads=True):
        """
        Process data in parallel using multiple workers
        
        Args:
            data: Iterable data to process
            func: Function to apply to each data item
            n_workers: Number of workers (default: number of CPU cores)
            use_threads: Use ThreadPoolExecutor if True, else ProcessPoolExecutor
            
        Returns:
            List of results
        """
        if n_workers is None:
            n_workers = os.cpu_count()
        
        # Choose executor based on parameter
        Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        
        results = []
        with st.spinner(f"Processing with {n_workers} {'threads' if use_threads else 'processes'}..."):
            with Executor(max_workers=n_workers) as executor:
                results = list(executor.map(func, data))
                
        return results
    
    @staticmethod
    def monitor_resource_usage(interval=2.0, duration=None):
        """
        Monitor and display resource usage over time
        
        Args:
            interval: Seconds between measurements
            duration: Total monitoring duration (None for indefinite)
        """
        if 'monitoring_active' not in st.session_state:
            st.session_state.monitoring_active = False
            st.session_state.monitoring_data = {
                'time': [],
                'memory_percent': [],
                'cpu_percent': []
            }
        
        # Toggle monitoring
        if st.button('Start/Stop Monitoring'):
            st.session_state.monitoring_active = not st.session_state.monitoring_active
            
            if st.session_state.monitoring_active:
                st.success("Resource monitoring activated")
            else:
                st.info("Resource monitoring stopped")
        
        # If monitoring is active, collect data
        if st.session_state.monitoring_active:
            # Record current timestamp and resource usage
            st.session_state.monitoring_data['time'].append(time.time())
            st.session_state.monitoring_data['memory_percent'].append(
                psutil.virtual_memory().percent
            )
            st.session_state.monitoring_data['cpu_percent'].append(
                psutil.cpu_percent()
            )
            
            # Create DataFrame from monitoring data
            if len(st.session_state.monitoring_data['time']) > 1:
                # Convert raw times to relative seconds
                start_time = st.session_state.monitoring_data['time'][0]
                relative_times = [t - start_time for t in st.session_state.monitoring_data['time']]
                
                monitoring_df = pd.DataFrame({
                    'Time (s)': relative_times,
                    'Memory (%)': st.session_state.monitoring_data['memory_percent'],
                    'CPU (%)': st.session_state.monitoring_data['cpu_percent']
                })
                
                # Display line chart of resource usage
                st.line_chart(
                    monitoring_df.set_index('Time (s)')
                )
                
                # Display summary statistics
                if len(monitoring_df) > 5:  # Only show stats if we have enough data points
                    st.write("### Resource Usage Statistics")
                    
                    stats_df = pd.DataFrame({
                        'Metric': ['Memory (%)', 'CPU (%)'],
                        'Min': [monitoring_df['Memory (%)'].min(), monitoring_df['CPU (%)'].min()],
                        'Max': [monitoring_df['Memory (%)'].max(), monitoring_df['CPU (%)'].max()],
                        'Mean': [monitoring_df['Memory (%)'].mean(), monitoring_df['CPU (%)'].mean()],
                        'Last': [monitoring_df['Memory (%)'].iloc[-1], monitoring_df['CPU (%)'].iloc[-1]]
                    })
                    
                    st.dataframe(stats_df)
            
            # Schedule the next update
            if duration is None or time.time() - st.session_state.monitoring_data['time'][0] < duration:
                time.sleep(interval)
                st.rerun()
            else:
                st.session_state.monitoring_active = False
                st.info(f"Monitoring stopped after {duration} seconds")
    
    @staticmethod
    def get_dataframe_memory_details(df):
        """
        Get detailed memory usage information for a DataFrame
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            DataFrame with memory usage details by column
        """
        memory_usage = df.memory_usage(deep=True)
        memory_usage_mb = memory_usage / (1024 * 1024)
        
        total_memory = memory_usage.sum() / (1024 * 1024)
        
        # Create DataFrame with memory usage details
        memory_details = pd.DataFrame({
            'Column': list(memory_usage.index),
            'Memory (MB)': list(memory_usage_mb),
            'Percent': [100 * m / total_memory for m in memory_usage_mb],
            'Type': [str(df[col].dtype) if col != 'Index' else 'index' for col in memory_usage.index if col != 'Index']
        })
        
        # Sort by memory usage
        memory_details = memory_details.sort_values('Memory (MB)', ascending=False).reset_index(drop=True)
        
        return memory_details
    
    @staticmethod
    def display_memory_optimization_ui(df):
        """
        Display UI for memory optimization options
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame or None if cancelled
        """
        st.write("### Memory Optimization Options")
        
        original_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.info(f"Current memory usage: {original_mem:.2f} MB")
        
        # Display memory details
        with st.expander("Memory usage by column", expanded=False):
            memory_details = MemoryManager.get_dataframe_memory_details(df)
            st.dataframe(memory_details)
            
            # Display bar chart of top memory users
            top_columns = memory_details.head(10).copy()
            st.bar_chart(top_columns.set_index('Column')['Memory (MB)'])
        
        # Optimization options
        st.write("Select optimization techniques:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            optimize_ints = st.checkbox("Optimize integer types", value=True)
            optimize_floats = st.checkbox("Optimize float types", value=True)
        
        with col2:
            optimize_objects = st.checkbox("Convert objects to categories", value=True)
            threshold = st.slider("Unique value threshold (%)", 1, 100, 50)
        
        if st.button("Apply Optimization"):
            with st.spinner("Optimizing DataFrame..."):
                optimized_df = df.copy()
                
                # Optimize integer types
                if optimize_ints:
                    for col in optimized_df.select_dtypes(include=['int']):
                        c_min = optimized_df[col].min()
                        c_max = optimized_df[col].max()
                        
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            optimized_df[col] = optimized_df[col].astype(np.int8)
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            optimized_df[col] = optimized_df[col].astype(np.int16)
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            optimized_df[col] = optimized_df[col].astype(np.int32)
                
                # Optimize float types
                if optimize_floats:
                    for col in optimized_df.select_dtypes(include=['float']):
                        optimized_df[col] = optimized_df[col].astype(np.float32)
                
                # Optimize object types
                if optimize_objects:
                    for col in optimized_df.select_dtypes(include=['object']):
                        if optimized_df[col].nunique() / len(optimized_df) < threshold/100:
                            optimized_df[col] = optimized_df[col].astype('category')
                
                # Display results
                optimized_mem = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
                reduction = 100 * (1 - optimized_mem/original_mem)
                
                st.success(f"Memory usage reduced from {original_mem:.2f} MB to {optimized_mem:.2f} MB ({reduction:.1f}% reduction)")
                
                # Display before/after comparison
                comparison = pd.DataFrame({
                    'Original Types': df.dtypes.astype(str),
                    'Optimized Types': optimized_df.dtypes.astype(str),
                    'Original Memory (MB)': [df[col].memory_usage(deep=True)/(1024*1024) for col in df.columns],
                    'Optimized Memory (MB)': [optimized_df[col].memory_usage(deep=True)/(1024*1024) for col in optimized_df.columns],
                })
                
                comparison['Reduction (%)'] = 100 * (1 - comparison['Optimized Memory (MB)']/comparison['Original Memory (MB)'])
                
                with st.expander("Detailed comparison", expanded=True):
                    st.dataframe(comparison.sort_values('Reduction (%)', ascending=False))
                
                return optimized_df
                
        if st.button("Cancel"):
            return None
            
        return df  # Return original if no optimization applied
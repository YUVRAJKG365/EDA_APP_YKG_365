import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import streamlit as st
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import squarify
import missingno as msno
from wordcloud import WordCloud
from data_loading import*
from time_series import time_series_analysis
import tracker
from util import tab_manager
from data_loading import load_data
import concurrent.futures
import dask.dataframe as dd
import psutil
import gc
import os
import tempfile

# Performance configuration
MAX_ROWS_FOR_FULL_PROCESSING = 50000  # Threshold for full processing
SAMPLE_SIZE = 10000  # Default sample size for large datasets
MAX_COLS_FOR_FULL_PROCESSING = 30  # Threshold for column-heavy processing
MAX_CATEGORIES_FOR_SUNBURST = 1000  # Maximum categories to show in sunburst
MEMORY_THRESHOLD = 0.8  # Memory usage threshold for safety (80%)

# Enhanced color palette
PROFESSIONAL_PALETTE = {
    'primary': '#2C3E50',     # Dark blue
    'secondary': '#E74C3C',   # Red
    'accent': '#3498DB',      # Blue
    'background': '#F9F9F9',  # Light gray
    'text': '#333333',        # Dark gray
    'highlight': '#F1C40F',   # Yellow
    'success': '#27AE60',     # Green
    'diverging': ['#D62728', '#FF7F0E', '#2CA02C', '#1F77B4', '#9467BD']
}

# Set professional style
sns.set_theme(style="whitegrid")
sns.set_style("whitegrid", {
    'axes.edgecolor': PROFESSIONAL_PALETTE['primary'],
    'axes.labelcolor': PROFESSIONAL_PALETTE['text'],
    'text.color': PROFESSIONAL_PALETTE['text'],
    'xtick.color': PROFESSIONAL_PALETTE['text'],
    'ytick.color': PROFESSIONAL_PALETTE['text'],
    'axes.grid': True,
    'grid.alpha': 0.3
})

def get_safe_sample_size(df, default_size=10000):
    """Determine a safe sample size based on available memory"""
    try:
        # Get available memory
        available_mem = psutil.virtual_memory().available
        
        # Estimate memory needed for sample (assume 100 bytes per cell)
        sample_mem_estimate = default_size * len(df.columns) * 100
        
        # If available memory is less than 2x estimate, reduce sample size
        if available_mem < sample_mem_estimate * 2:
            safe_size = min(default_size, int(available_mem / (len(df.columns) * 100)))
            return max(1000, safe_size)  # Ensure minimum of 1000 rows
            
        return default_size
    except:
        return default_size

def get_downsampled_data(df, target_size=10000):
    """Downsample large datasets using intelligent strategies"""
    if len(df) <= target_size:
        return df
    
    # Check if dataset has a time index - SAFER METHOD
    if isinstance(df.index, pd.DatetimeIndex):  # Changed this line
        return df.resample('D').mean().ffill().sample(min(target_size, len(df)))
    
    # Check for categorical columns to stratify
    cat_cols = df.select_dtypes(include=['category', 'object']).columns
    if len(cat_cols) > 0:
        try:
            # Stratified sampling
            sample = pd.DataFrame()
            for _, group in df.groupby(cat_cols[0]):
                sample = pd.concat([sample, group.sample(int(target_size/len(df)*len(group)))])
            return sample.sample(min(target_size, len(sample)))
        except:
            pass
    
    # Simple random sampling as fallback
    return df.sample(min(target_size, len(df)))

def efficient_value_counts(df, col, max_categories=1000):
    """Memory-efficient value counts with support for single or multi-column aggregations"""
    # Handle multi-column case
    if isinstance(col, (list, tuple)):
        return df.groupby(col).size().nlargest(max_categories)

    # For categorical data, use built-in method
    if pd.api.types.is_categorical_dtype(df[col]):
        counts = df[col].value_counts()
        return counts.head(max_categories)
    
    # For object dtype, use efficient method
    if df[col].dtype == 'object':
        # Process in chunks for large datasets
        if len(df) > 1_000_000:
            unique_counts = {}
            chunk_size = 100_000
            for i in range(0, len(df), chunk_size):
                chunk = df[col].iloc[i:i+chunk_size]
                for val in chunk:
                    unique_counts[val] = unique_counts.get(val, 0) + 1
            counts = pd.Series(unique_counts).sort_values(ascending=False)
            return counts.head(max_categories)
        else:
            counts = df[col].value_counts()
            return counts.head(max_categories)

def plot_missing_values_enhanced(df):
    tracker = st.session_state.tracker
    """
    Enhanced missing values visualization with matrix, heatmap and bar chart
    """
    try:
        if df is None or not hasattr(df, "empty") or df.empty:
            st.info("Please upload and load a dataset first. No missing values to display.")
            return None

        # Create a tabbed interface for different missing data visualizations
        tab1, tab2, tab3 = st.tabs(["Matrix View", "Heatmap", "Bar Chart"])
        
        with tab1:
            tracker.log_subtab("Matrix View")
            st.subheader("Missing Values Matrix")
            try:
                # For large datasets, show a sample
                display_df = df.sample(min(5000, len(df))) if len(df) > 5000 else df
                
                fig, ax = plt.subplots(figsize=(12, 8))
                msno.matrix(display_df, ax=ax, color=(231/255, 76/255, 60/255))  # '#E74C3C' as RGB
                ax.set_title("Missing Data Pattern", pad=20, 
                             fontdict={'fontsize': 16, 'color': PROFESSIONAL_PALETTE['primary']})
                st.pyplot(fig)
                tracker.log_operation("Viewed Missing Values Matrix")
            except Exception as e:
                st.error(f"❌ Error in Matrix View: {e}")

        with tab2:
            tracker.log_subtab("Heatmap")
            st.subheader("Missing Values Correlation")
            try:
                # For large datasets, show a sample
                display_df = df.sample(min(5000, len(df))) if len(df) > 5000 else df
                
                fig, ax = plt.subplots(figsize=(10, 6))
                try:
                    msno.heatmap(display_df, ax=ax, cmap=PROFESSIONAL_PALETTE['diverging'])
                except Exception:
                    msno.heatmap(display_df, ax=ax)  # fallback to default
                ax.set_title("Missing Data Correlation", pad=20,
                            fontdict={'fontsize': 16, 'color': PROFESSIONAL_PALETTE['primary']})
                st.pyplot(fig)
                tracker.log_operation("Viewed Heatmap")
            except Exception as e:
                st.error(f"❌ Error in Heatmap View: {e}")

        with tab3:
            tracker.log_subtab("Bar Chart")
            st.subheader("Missing Values Count")
            try:
                # Use efficient computation for large datasets
                if len(df) > 1000000:
                    st.info("Large dataset: Computing missing values efficiently...")
                    missing_values = pd.Series({col: df[col].isnull().sum() for col in df.columns})
                else:
                    missing_values = df.isnull().sum()
                
                missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
                
                if missing_values.empty:
                    st.success("✅ No missing values found in the dataset!")
                    return None
                    
                fig = px.bar(missing_values, 
                             x=missing_values.index, 
                             y=missing_values.values,
                             color=missing_values.values,
                             color_continuous_scale=PROFESSIONAL_PALETTE['diverging'] if isinstance(PROFESSIONAL_PALETTE['diverging'], list) else 'Viridis',
                             labels={'x': 'Columns', 'y': 'Missing Values Count'},
                             title="Missing Values by Column")
                
                fig.update_layout(
                    plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                    paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                    font_color=PROFESSIONAL_PALETTE['text'],
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True)
                tracker.log_operation("Viewed Bar Chart")
            except Exception as e:
                st.error(f"❌ Error in Bar Chart View: {e}")
            
    except Exception as e:
        st.error(f"❌ Error generating missing values visualization: {str(e)}")
        return None

def plot_distribution_analysis(df):
    tracker = st.session_state.tracker
    """
    Comprehensive distribution analysis with histogram, KDE, and Q-Q plot
    """
    try:
        if df is None or not hasattr(df, "select_dtypes") or df.empty:
            st.info("Please upload and load a dataset first. No data available for distribution analysis.")
            return None

        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            st.info("No numeric columns found for distribution analysis.")
            return None
            
        # Create interactive selection
        selected_col = st.selectbox("Select column for distribution analysis:", numeric_df.columns)
        
        # Create tabs for different distribution views
        tab1, tab2, tab3 = st.tabs(["Histogram & KDE", "Q-Q Plot", "Box & Violin"])
        
        with tab1:
            tracker.log_subtab("Histogram & KDE")
            try:
                # Downsample large datasets
                plot_data = get_downsampled_data(numeric_df[selected_col].dropna(), 10000)
                
                fig = make_subplots(rows=1, cols=1)
                # Histogram
                fig.add_trace(
                    go.Histogram(
                        x=plot_data,
                        name="Histogram",
                        marker_color=PROFESSIONAL_PALETTE['accent'],
                        opacity=0.75,
                        nbinsx=50
                    ),
                    row=1, col=1
                )
                # KDE
                fig.add_trace(
                    go.Scatter(
                        x=np.linspace(plot_data.min(), plot_data.max(), 100),
                        y=stats.gaussian_kde(plot_data)(
                            np.linspace(plot_data.min(), plot_data.max(), 100)
                        ) * len(plot_data) * (
                            plot_data.max() - plot_data.min()
                        ) / 50,
                        name="KDE",
                        line=dict(color=PROFESSIONAL_PALETTE['secondary'], width=2)
                    ),
                    row=1, col=1
                )
                fig.update_layout(
                    title=f"Distribution of {selected_col}",
                    xaxis_title=selected_col,
                    yaxis_title="Count",
                    plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                    paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                    font_color=PROFESSIONAL_PALETTE['text'],
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                tracker.log_operation("Viewed Histogram & KDE")
            except Exception as e:
                st.error(f"Error in Histogram & KDE: {e}")
            
        with tab2:
            tracker.log_subtab("Q-Q Plot")
            try:
                # Downsample large datasets
                plot_data = get_downsampled_data(numeric_df[selected_col].dropna(), 10000)
                
                fig = go.Figure()
                qq = stats.probplot(plot_data, dist="norm")
                theoretical = [qq[0][0][0], qq[0][0][-1]]
                fig.add_trace(
                    go.Scatter(
                        x=theoretical,
                        y=theoretical,
                        name="Theoretical",
                        line=dict(color=PROFESSIONAL_PALETTE['primary'], width=2, dash='dash')
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=qq[0][0],
                        y=qq[0][1],
                        name="Sample",
                        mode='markers',
                        marker=dict(color=PROFESSIONAL_PALETTE['secondary'], size=8)
                    )
                )
                fig.update_layout(
                    title=f"Q-Q Plot for {selected_col}",
                    xaxis_title="Theoretical Quantiles",
                    yaxis_title="Sample Quantiles",
                    plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                    paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                    font_color=PROFESSIONAL_PALETTE['text'],
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                tracker.log_operation("Viewed Q-Q Plot")
            except Exception as e:
                st.error(f"Error in Q-Q Plot: {e}")
            
        with tab3:
            tracker.log_subtab("Box & Violin")
            try:
                # Downsample large datasets
                plot_data = get_downsampled_data(numeric_df[selected_col].dropna(), 10000)
                
                fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Box Plot", f"Violin Plot"))
                # Box plot
                fig.add_trace(
                    go.Box(
                        y=plot_data,
                        name="Box",
                        marker_color=PROFESSIONAL_PALETTE['accent'],
                        boxmean=True
                    ),
                    row=1, col=1
                )
                # Violin plot
                fig.add_trace(
                    go.Violin(
                        y=plot_data,
                        name="Violin",
                        marker_color=PROFESSIONAL_PALETTE['secondary'],
                        box_visible=True,
                        meanline_visible=True
                    ),
                    row=1, col=2
                )
                fig.update_layout(
                    title=f"Spread Analysis for {selected_col}",
                    plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                    paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                    font_color=PROFESSIONAL_PALETTE['text'],
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                tracker.log_operation("Viewed Box & Violin Plot")
            except Exception as e:
                st.error(f"Error in Box & Violin Plot: {e}")
            
    except Exception as e:
        st.error(f"Error in distribution analysis: {e}")
        return None

def plot_correlation_analysis(df):
    tracker = st.session_state.tracker
    """
    Enhanced correlation analysis with multiple visualization options
    """
    try:
        if df is None or not hasattr(df, "select_dtypes") or df.empty:
            st.info("Please upload and load a dataset first. No data available for correlation analysis.")
            return None

        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty or len(numeric_df.columns) < 2:
            st.info("Not enough numeric columns for correlation analysis.")
            return None
            
        # Create tabs for different correlation views
        tab1, tab2, tab3, tab4 = st.tabs(["Heatmap", "Scatter Matrix", "Pair Plot", "Correlation Network"])
        
        with tab1:
            tracker.log_subtab("Heatmap")
            try:
                # Efficient correlation calculation
                st.info("Calculating correlations efficiently...")
                if len(numeric_df) > 1000000:
                    # Use Dask for out-of-core computation
                    ddf = dd.from_pandas(numeric_df, npartitions=10)
                    corr_matrix = ddf.corr().compute()
                else:
                    corr_matrix = numeric_df.corr()
                
                rainbow_colorscale = [
                    [0.0, "#002147"],   # Dark Blue
                    [0.2, "#1976D2"],   # Light Blue
                    [0.4, "#00BFFF"],   # Sky Blue
                    [0.6, "#FF0000"],   # Red
                    [0.8, "#FFD700"],   # Gold/Yellow
                    [1.0, "#8B00FF"]    # Violet
                ]
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale=rainbow_colorscale,
                    zmin=-1,
                    zmax=1,
                    hoverongaps=False,
                    colorbar=dict(title="Correlation")
                ))
                fig.update_layout(
                    title="Correlation Heatmap",
                    xaxis_title="Features",
                    yaxis_title="Features",
                    plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                    paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                    font_color=PROFESSIONAL_PALETTE['text'],
                    width=800,
                    height=700
                )
                st.plotly_chart(fig, use_container_width=True)
                tracker.log_operation("Viewed Correlation Heatmap")
            except Exception as e:
                st.error(f"Error in Heatmap: {e}")
            
        with tab2:
            tracker.log_subtab("Scatter Matrix")
            try:
                # Limit to 5 columns for performance
                display_cols = numeric_df.columns[:min(5, len(numeric_df.columns))]
                plot_df = get_downsampled_data(numeric_df[display_cols], 5000)
                
                fig = px.scatter_matrix(
                    plot_df,
                    dimensions=display_cols,
                    color=display_cols[0] if len(display_cols) > 0 else None,
                    title="Scatter Matrix",
                    color_continuous_scale=PROFESSIONAL_PALETTE['diverging']
                )
                fig.update_layout(
                    plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                    paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                    font_color=PROFESSIONAL_PALETTE['text']
                )
                st.plotly_chart(fig, use_container_width=True)
                tracker.log_operation("Viewed Scatter Matrix")
            except Exception as e:
                st.error(f"Error in Scatter Matrix: {e}")
            
        with tab3:
            tracker.log_subtab("Pair Plot")
            try:
                # Limit to 3 columns for performance
                display_cols = numeric_df.columns[:min(3, len(numeric_df.columns))]
                plot_df = get_downsampled_data(numeric_df[display_cols], 5000)
                
                fig = px.scatter(
                    plot_df,
                    x=display_cols[0],
                    y=display_cols[1] if len(display_cols) > 1 else display_cols[0],
                    color=display_cols[2] if len(display_cols) > 2 else None,
                    marginal_x="histogram",
                    marginal_y="histogram",
                    trendline="ols",
                    title="Interactive Pair Plot",
                    color_continuous_scale=PROFESSIONAL_PALETTE['diverging']
                )
                fig.update_layout(
                    plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                    paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                    font_color=PROFESSIONAL_PALETTE['text']
                )
                st.plotly_chart(fig, use_container_width=True)
                tracker.log_operation("Viewed Pair Plot")
            except Exception as e:
                st.error(f"Error in Pair Plot: {e}")
            
        with tab4:
            tracker.log_subtab("Correlation Analysis")
            try:
                # Efficient correlation calculation
                if len(numeric_df) > 1000000:
                    # Use Dask for out-of-core computation
                    ddf = dd.from_pandas(numeric_df, npartitions=10)
                    corr_matrix = ddf.corr().compute()
                else:
                    corr_matrix = numeric_df.corr()
                    
                corr_threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.7, 0.05)
                edges = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i):
                        if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                            edges.append((
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                corr_matrix.iloc[i, j]
                            ))
                if not edges:
                    st.info(f"No correlations above threshold {corr_threshold}")
                    return
                edge_x = []
                edge_y = []
                edge_text = []
                node_x = []
                node_y = []
                node_text = []
                nodes = list(set([edge[0] for edge in edges] + [edge[1] for edge in edges]))
                angle = 2 * np.pi / len(nodes)
                for i, node in enumerate(nodes):
                    node_x.append(np.cos(i * angle))
                    node_y.append(np.sin(i * angle))
                    node_text.append(node)
                for edge in edges:
                    x0, y0 = node_x[nodes.index(edge[0])], node_y[nodes.index(edge[0])]
                    x1, y1 = node_x[nodes.index(edge[1])], node_y[nodes.index(edge[1])]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_text.append(f"Corr: {edge[2]:.2f}")
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=1, color='#888'),
                    hoverinfo='text',
                    text=edge_text,
                    mode='lines')
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=node_text,
                    textposition="top center",
                    marker=dict(
                        showscale=True,
                        colorscale=PROFESSIONAL_PALETTE['diverging'],
                        size=20,
                        color=[10] * len(nodes),  # All same color for now
                        colorbar=dict(
                            thickness=15,
                            title='Node Connections',
                            xanchor='left',
                            titleside='right'
                        ),
                        line_width=2))
                fig = go.Figure(data=[edge_trace, node_trace],
                             layout=go.Layout(
                                title=f'Correlation Network (Threshold: {corr_threshold})',
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                                paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                                font_color=PROFESSIONAL_PALETTE['text']
                            ))
                st.plotly_chart(fig, use_container_width=True)
                tracker.log_operation("Viewed Correlation Network")
            except Exception as e:
                st.error(f"Error in Correlation Network: {e}")
            
    except Exception as e:
        st.error(f"Error in correlation analysis: {e}")
        return None

def plot_categorical_analysis(df):
    tracker = st.session_state.tracker
    """
    Comprehensive categorical data analysis with multiple visualization types
    """
    try:
        if df is None or not hasattr(df, "select_dtypes") or df.empty:
            st.info("Please upload and load a dataset first. No data available for categorical analysis.")
            return None

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.any():
            st.info("No categorical columns found for analysis.")
            return None
            
        selected_col = st.selectbox("Select categorical column:", categorical_cols)
        
        # Create tabs for different categorical visualizations
        tab1, tab2, tab3 = st.tabs(["Bar Chart", "Pie Chart", "Treemap"])

        with tab1:
            tracker.log_subtab("Bar Chart")
            try:
                value_counts = efficient_value_counts(df, selected_col, 20)
                
                fig = px.bar(
                    value_counts,
                    x=value_counts.index,
                    y=value_counts.values,
                    color=value_counts.values,
                    color_continuous_scale=PROFESSIONAL_PALETTE['diverging'],
                    labels={'x': selected_col, 'y': 'Count'},
                    title=f"Distribution of {selected_col}"
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                    paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                    font_color=PROFESSIONAL_PALETTE['text']
                )
                st.plotly_chart(fig, use_container_width=True)
                tracker.log_operation("Viewed Bar Chart")
            except Exception as e:
                st.error(f"Error in Bar Chart: {e}")
            
        with tab2:
            tracker.log_subtab("Pie Chart")
            try:
                value_counts = efficient_value_counts(df, selected_col, 10)
                
                fig = px.pie(
                    value_counts,
                    names=value_counts.index,
                    values=value_counts.values,
                    title=f"Proportion of {selected_col}",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    pull=[0.1 if i == 0 else 0 for i in range(len(value_counts))]  # Pull out the largest slice
                )
                fig.update_layout(
                    plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                    paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                    font_color=PROFESSIONAL_PALETTE['text'],
                    uniformtext_minsize=12,
                    uniformtext_mode='hide'
                )
                st.plotly_chart(fig, use_container_width=True)
                tracker.log_operation("Viewed Pie Chart")
            except Exception as e:
                st.error(f"Error in Pie Chart: {e}")
            
        with tab3:
            tracker.log_subtab("Treemap")
            try:
                value_counts = efficient_value_counts(df, selected_col, 20)
                
                fig = go.Figure(go.Treemap(
                    labels=value_counts.index,
                    parents=[''] * len(value_counts),
                    values=value_counts.values,
                    marker_colors=value_counts.values,
                    textinfo="label+value+percent parent",
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.1%}',
                    pathbar={"visible": True}
                ))
                fig.update_layout(
                    title=f"Treemap of {selected_col}",
                    treemapcolorway=PROFESSIONAL_PALETTE['diverging'],
                    plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                    paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                    font_color=PROFESSIONAL_PALETTE['text'],
                    margin=dict(t=40, l=0, r=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                tracker.log_operation("Viewed Treemap")
            except Exception as e:
                st.error(f"Error in Treemap: {e}")
                
    except Exception as e:
        st.error(f"Error in categorical analysis: {e}")
        return None

def plot_advanced_visualizations(df):
    tracker = st.session_state.tracker
    """
    Advanced visualizations for deeper insights with memory optimizations
    """
    try:
        if df is None or not hasattr(df, "select_dtypes") or df.empty:
            st.info("Please upload and load a dataset first. No data available for advanced visualizations.")
            return

        # Identify column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Add partition selection UI for large datasets
        PARTITION_OPTIONS = ["None", "A (first 25%)", "B (25-50%)", "C (50-75%)", "D (75-100%)"]
        partition_choice = "None"
        
        if len(df) > 10000:  # Enable chunking for large datasets
            st.subheader("Data Partitioning for Large Datasets")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                partition_a = st.selectbox("A", PARTITION_OPTIONS, key="part_a")
            with col2:
                partition_b = st.selectbox("B", PARTITION_OPTIONS, key="part_b")
            with col3:
                partition_c = st.selectbox("C", PARTITION_OPTIONS, key="part_c")
            with col4:
                partition_d = st.selectbox("D", PARTITION_OPTIONS, key="part_d")
            
            # Determine selected partition
            if partition_a != "None":
                partition_choice = partition_a
            elif partition_b != "None":
                partition_choice = partition_b
            elif partition_c != "None":
                partition_choice = partition_c
            elif partition_d != "None":
                partition_choice = partition_d
        
        # Apply partition selection
        if partition_choice != "None":
            chunk_size = len(df) // 4
            part_index = PARTITION_OPTIONS.index(partition_choice) - 1
            start = part_index * chunk_size
            end = (part_index + 1) * chunk_size if part_index < 3 else len(df)
            df_vis = df.iloc[start:end].copy()
            st.info(f"Visualizing partition {partition_choice} ({len(df_vis)}/{len(df)} rows)")
        else:
            df_vis = df.copy()  # Use full dataset if no partition selected

        # Auto-sampling parameters
        MAX_POINTS = 100000  # Maximum points to show in scatter plots
        auto_sample = len(df_vis) > MAX_POINTS

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Parallel Coordinates",
            "Radar Chart",
            "Sunburst",
            "2D Bubble Chart",
            "3D Bubble Chart"
        ])

        # 1) Parallel Coordinates
        with tab1:
            tracker.log_subtab("Parallel Coordinates")
            if len(numeric_cols) <= 2:
                st.info("Need at least 2 numeric columns for parallel coordinates.")
            else:
                default_cols = numeric_cols[:min(5, len(numeric_cols))]
                selected_cols = st.multiselect(
                    "Select columns for parallel coordinates:",
                    numeric_cols,
                    default=default_cols
                )
                if len(selected_cols) >= 2:
                    # Apply sampling if needed
                    plot_df = df_vis
                    if len(df_vis) > MAX_POINTS:
                        plot_df = df_vis.sample(n=MAX_POINTS, random_state=42)
                        st.info(f"Showing random sample of {MAX_POINTS} points for performance")
                        
                    fig = px.parallel_coordinates(
                        plot_df,
                        dimensions=selected_cols,
                        color=selected_cols[0],
                        color_continuous_scale=PROFESSIONAL_PALETTE['diverging'],
                        title="Parallel Coordinates Plot"
                    )
                    fig.update_layout(
                        plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                        paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                        font_color=PROFESSIONAL_PALETTE['text']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    tracker.log_operation("Viewed Parallel Coordinates")

        # 2) Radar Chart
        with tab2:
            tracker.log_subtab("Radar Chart")
            if len(numeric_cols) < 3:
                st.info("Need at least 3 numeric columns for radar chart.")
            else:
                # Use entire dataset for accurate feature means
                means = df[numeric_cols].mean()
                min_val, max_val = means.min(), means.max()
                span = max_val - min_val if max_val != min_val else 1
                normalized = ((means - min_val) / span).reset_index()
                normalized.columns = ['feature', 'value']

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=normalized['value'],
                    theta=normalized['feature'],
                    fill='toself',
                    name="Features",
                    line_color=PROFESSIONAL_PALETTE['accent']
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Radar Chart of Feature Values",
                    plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                    paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                    font_color=PROFESSIONAL_PALETTE['text']
                )
                st.plotly_chart(fig, use_container_width=True)
                tracker.log_operation("Viewed Radar Chart")

        # 3) Sunburst Chart
        with tab3:
            tracker.log_subtab("Sunburst Chart")
            if len(categorical_cols) < 2:
                st.info("Need at least 2 categorical columns for sunburst chart.")
            else:
                default_cats = categorical_cols[:2]
                selected_cats = st.multiselect(
                    "Select hierarchical columns:",
                    categorical_cols,
                    default=default_cats
                )
                if len(selected_cats) >= 2:
                    fig = px.sunburst(
                        df_vis,
                        path=selected_cats,
                        title="Hierarchical Sunburst Chart",
                        color_discrete_sequence=PROFESSIONAL_PALETTE['diverging']
                    )
                    fig.update_layout(
                        plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                        paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                        font_color=PROFESSIONAL_PALETTE['text']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    tracker.log_operation("Viewed Sunburst Chart")

        # 4) Bubble Chart - Optimized RGB generation
        with tab4:
            tracker.log_subtab("2D Bubble Chart")
            if len(numeric_cols) < 2:
                st.info("Need at least 2 numeric columns for bubble chart.")
            else:
                # First select x and y axes
                x_col = st.selectbox("X-axis:", numeric_cols, index=0)
                remaining_cols = [col for col in numeric_cols if col not in [x_col]]
                y_col = st.selectbox("Y-axis:", remaining_cols, index=0)
                
                # Update remaining columns for size selection (exclude x and y)
                remaining_cols_size = [col for col in numeric_cols if col not in [x_col, y_col]]
                size_col = st.selectbox(
                    "Bubble size:", remaining_cols_size,
                    index=0
                )
                
                # Update remaining columns for color selection (exclude x, y, and size)
                remaining_cols_color = [col for col in numeric_cols if col not in [x_col, y_col, size_col]]
                color_col = st.selectbox(
                    "Bubble color:", remaining_cols_color,
                    index=0
                )

                # Apply sampling if needed
                plot_df = df_vis
                if len(df_vis) > MAX_POINTS:
                    plot_df = df_vis.sample(n=MAX_POINTS, random_state=42)
                    st.info(f"Showing random sample of {MAX_POINTS} points for performance")

                # Compute absolute values and scale to [10, 60]
                vals = plot_df[size_col].abs().fillna(0)
                if vals.max() == vals.min():
                    scaled = pd.Series(20, index=plot_df.index)
                else:
                    min_s, max_s = 10, 60
                    scaled = min_s + (vals - vals.min()) * (max_s - min_s) / (vals.max() - vals.min())
                plot_df = plot_df.assign(_bubble_size=scaled)

                # Optimized RGB color generation
                norm_color = plot_df[color_col].to_numpy()
                
                # Handle normalization with vectorized operations
                if norm_color.max() <= 1.0:
                    norm_color = (norm_color * 255).astype(np.uint8)
                elif norm_color.min() < 0 or norm_color.max() > 255:
                    norm_min = norm_color.min()
                    norm_max = norm_color.max()
                    norm_color = ((norm_color - norm_min) / (norm_max - norm_min) * 255).astype(np.uint8)
                else:
                    norm_color = norm_color.astype(np.uint8)
                
                # Vectorized RGB generation
                r = norm_color
                g = 255 - r
                b = np.random.randint(0, 256, size=len(plot_df), dtype=np.uint8)
                rgb_colors = np.stack([r, g, b], axis=1)
                rgb_strings = [f'rgb({c[0]},{c[1]},{c[2]})' for c in rgb_colors]

                # Plot with optimized coloring
                fig = px.scatter(
                    plot_df,
                    x=x_col,
                    y=y_col,
                    size="_bubble_size",
                    color=rgb_strings,
                    hover_name=plot_df.index if plot_df.index.name else None,
                    title="Interactive Bubble Chart with RGB Coloring"
                )
                fig.update_layout(
                    plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                    paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                    font_color=PROFESSIONAL_PALETTE['text'],
                    xaxis_title=x_col,
                    yaxis_title=y_col
                )
                st.plotly_chart(fig, use_container_width=True)
                tracker.log_operation("Viewed 2D Bubble Chart")

        # 5) Enhanced 3D Bubble Chart
        with tab5:
            tracker.log_subtab("Viewed 3D Bubble Chart")
            st.subheader("3D Bubble Chart")
            
            if df_vis.empty:
                st.info("No data available for 3D visualization.")
            else:
                all_cols = df_vis.columns.tolist()
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-Axis", all_cols, index=0)
                    remaining_cols_y = [col for col in all_cols if col != x_col]
                    y_col = st.selectbox("Y-Axis", remaining_cols_y, index=0)
                with col2:
                    remaining_cols_z = [col for col in all_cols if col not in [x_col, y_col]]
                    z_col = st.selectbox("Z-Axis", remaining_cols_z, index=0)
                    remaining_cols_color = [col for col in all_cols if col not in [x_col, y_col, z_col]]
                    color_col = st.selectbox("Color", remaining_cols_color, index=0)

                # Apply sampling if needed
                plot_df = df_vis
                if len(df_vis) > MAX_POINTS:
                    plot_df = df_vis.sample(n=MAX_POINTS, random_state=42)
                    st.info(f"Showing random sample of {MAX_POINTS} points for performance")
                    
                plot_df = plot_df[[x_col, y_col, z_col, color_col]].copy()
                color_data = plot_df[color_col]
                
                # Optimized color handling
                if pd.api.types.is_numeric_dtype(color_data):
                    norm_color = color_data.to_numpy()
                    if norm_color.max() <= 1.0:
                        norm_color = (norm_color * 255).astype(np.uint8)
                    elif norm_color.min() < 0 or norm_color.max() > 255:
                        norm_min = norm_color.min()
                        norm_max = norm_color.max()
                        norm_color = ((norm_color - norm_min) / (norm_max - norm_min) * 255).astype(np.uint8)
                    else:
                        norm_color = norm_color.astype(np.uint8)
                else:
                    # Categorical columns
                    codes = pd.factorize(color_data)[0]
                    norm_min = codes.min()
                    norm_max = codes.max()
                    span = norm_max - norm_min if norm_max != norm_min else 1
                    norm_color = ((codes - norm_min) / span * 255).astype(np.uint8)
                
                # Vectorized RGB generation
                r = norm_color
                g = 255 - r
                b = np.random.randint(0, 256, size=len(plot_df), dtype=np.uint8)
                rgb_colors = np.stack([r, g, b], axis=1)
                rgb_strings = [f'rgb({c[0]},{c[1]},{c[2]})' for c in rgb_colors]

                # Create 3D plot
                fig = px.scatter_3d(
                    plot_df,
                    x=x_col,
                    y=y_col,
                    z=z_col,
                    color=rgb_strings,
                    title="3D Bubble Chart with RGB Coloring"
                )
                fig.update_layout(
                    scene=dict(
                        xaxis_title=x_col,
                        yaxis_title=y_col,
                        zaxis_title=z_col
                    ),
                    plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                    paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                    font_color=PROFESSIONAL_PALETTE['text'],
                    height=700
                )
                st.plotly_chart(fig, use_container_width=True)
                tracker.log_operation("Viewed 3D Bubble Chart")

    except Exception as e:
        st.error(f"Error in advanced visualizations: {e}")

            
def load_data(uploaded_file):
    """Load data from uploaded file, supporting tabular formats with memory optimization"""
    try:
        # For large files, use memory mapping
        if uploaded_file.size > 100 * 1024 * 1024:  # >100MB
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
                
            if uploaded_file.type == 'text/csv' or uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(tmp_path, low_memory=False, memory_map=True)
            elif uploaded_file.type in ['application/vnd.ms-excel', 
                                      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'] or \
                 uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(tmp_path)
            else:
                raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
                
            os.unlink(tmp_path)
            return df
        
        # For smaller files, use standard loading
        if uploaded_file.type == 'text/csv' or uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.type in ['application/vnd.ms-excel', 
                                  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'] or \
             uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

def render_visualizations_section(df):
    tracker = st.session_state.tracker
    tracker.log_section("Visualizations")
    st.markdown("## 📈 Advanced Visualizations")
    st.markdown("Explore your dataset through comprehensive visualizations.")
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df

    tabs = [
        "🔍 Missing Values",
        "📈 Distributions",
        "🔗 Correlations",
        "📝 Categories",
        "✨ Advanced",
    ]

    choice = st.radio(
        label="",
        options=tabs,
        key="viz_tab_choice",
        horizontal=True,
    )

    # 2) Convert that string to an integer index
    active = tabs.index(choice)

    if active == 0:
        tracker.log_operation("Viewed Missing Values Analysis")
        st.markdown("### Missing Values Analysis")
        plot_missing_values_enhanced(df)

    elif active == 1:
        tracker.log_operation("Viewed Distribution Analysis")
        st.markdown("### Distribution Analysis")
        plot_distribution_analysis(df)

    elif active == 2:
        tracker.log_operation("Viewed Correlation Analysis")
        st.markdown("### Correlation Analysis")
        plot_correlation_analysis(df)

    elif active == 3:
        tracker.log_operation("Viewed Categorical Data Analysis")
        st.markdown("### Categorical Data Analysis")
        plot_categorical_analysis(df)

    elif active == 4:
        tracker.log_operation("Viewed Advanced Visualizations")
        st.markdown("### Advanced Visualizations")
        plot_advanced_visualizations(df)

        
           
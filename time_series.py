import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.express as px
from statsmodels.tsa.stattools import acf, pacf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from pmdarima import auto_arima
from prophet import Prophet
import seaborn as sns
import warnings
from dask import dataframe as dd
import gc
from tqdm import tqdm

from utils.data_loader import display_file_info, handle_file_upload
from utils.session_state_manager import get_session_manager
warnings.filterwarnings('ignore')

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

def optimize_dataframe(df):
    """Optimize dataframe memory usage"""
    # Convert to datetime efficiently
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='ignore')
        except:
            pass
    
    # Downcast numeric columns
    for col in df.select_dtypes(include=['integer']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

def preprocess_time_series(df, value_col):
    """Preprocess time series data with resampling, filling missing values, etc."""
    try:
        # Create a copy and optimize memory
        df = optimize_dataframe(df.copy())
        
        # Ensure we have the value column
        if value_col not in df.columns:
            st.error(f"Value column '{value_col}' not found in dataframe")
            return df
        
        # Container for all preprocessing options
        with st.expander("⚙️ Data Preprocessing Options", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                # Resampling options
                st.markdown("#### Resampling")
                resample_freq = st.selectbox(
                    "Resampling Frequency",
                    ['Raw', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly'],
                    index=0,
                    key='resample_freq'
                )
                
                if resample_freq != 'Raw':
                    freq_map = {
                        'Daily': 'D',
                        'Weekly': 'W',
                        'Monthly': 'M',
                        'Quarterly': 'Q',
                        'Yearly': 'Y'
                    }
                    # Use efficient resampling with mean()
                    df = df.resample(freq_map[resample_freq]).mean()
            
            with col2:
                # Handle missing values
                st.markdown("#### Missing Values")
                missing_method = st.selectbox(
                    "Handle Missing Values",
                    ['None', 'Forward Fill', 'Backward Fill', 'Linear Interpolation', 'Mean', 'Median'],
                    index=0,
                    key='missing_method'
                )
                
                if missing_method == 'Forward Fill':
                    df[value_col] = df[value_col].ffill()
                elif missing_method == 'Backward Fill':
                    df[value_col] = df[value_col].bfill()
                elif missing_method == 'Linear Interpolation':
                    df[value_col] = df[value_col].interpolate(method='linear')
                elif missing_method == 'Mean':
                    df[value_col] = df[value_col].fillna(df[value_col].mean())
                elif missing_method == 'Median':
                    df[value_col] = df[value_col].fillna(df[value_col].median())
        
            # Outlier detection - only if dataset isn't too large
            if len(df) < 100000:  # Only run outlier detection on smaller datasets
                st.markdown("#### Outlier Detection")
                outlier_col1, outlier_col2 = st.columns(2)
                
                with outlier_col1:
                    remove_outliers = st.checkbox("Remove Outliers", False, key='remove_outliers')
                
                if remove_outliers:
                    with outlier_col2:
                        method = st.selectbox(
                            "Outlier Detection Method",
                            ['Z-Score', 'IQR', 'Isolation Forest'],
                            index=0,
                            key='outlier_method'
                        )
                    
                    if method == 'Z-Score':
                        threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, key='z_threshold')
                        z_scores = (df[value_col] - df[value_col].mean()) / df[value_col].std()
                        df = df[np.abs(z_scores) < threshold]
                    elif method == 'IQR':
                        q1 = df[value_col].quantile(0.25)
                        q3 = df[value_col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - (1.5 * iqr)
                        upper_bound = q3 + (1.5 * iqr)
                        df = df[(df[value_col] >= lower_bound) & (df[value_col] <= upper_bound)]
                    elif method == 'Isolation Forest':
                        contamination = st.slider("Contamination", 0.01, 0.5, 0.05, key='contamination')
                        model = IsolationForest(contamination=contamination, n_jobs=-1)
                        preds = model.fit_predict(df[[value_col]])
                        df = df[preds == 1]
        
        return df
    
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return df

def large_data_decomposition(series, model='additive', period=12):
    """Efficient decomposition for large datasets"""
    try:
        # Use a sample if the series is too large
        if len(series) > 10000:
            sample_size = min(10000, len(series))
            step = len(series) // sample_size
            series_sample = series.iloc[::step].copy()
        else:
            series_sample = series.copy()
        
        # Handle missing values in the sample
        series_sample = series_sample.dropna()
        
        # Check if multiplicative can be used
        if model == 'multiplicative' and (series_sample <= 0).any():
            st.warning("Multiplicative decomposition not suitable for series with ≤0 values. Switching to additive.")
            model = 'additive'
        
        # Perform decomposition
        decomposition = seasonal_decompose(
            series_sample,
            model=model,
            period=period,
            extrapolate_trend='freq'
        )
        
        return decomposition
    
    except Exception as e:
        st.error(f"Error in decomposition: {e}")
        return None

def efficient_rolling_stats(df, value_col, window):
    """Calculate rolling statistics efficiently for large datasets"""
    try:
        # Calculate rolling stats in chunks if dataset is large
        if len(df) > 10000:
            chunk_size = 10000
            chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
            
            rolling_mean = pd.concat([
                chunk[value_col].rolling(window=window).mean() 
                for chunk in tqdm(chunks, desc="Calculating rolling mean")
            ])
            
            rolling_std = pd.concat([
                chunk[value_col].rolling(window=window).std() 
                for chunk in tqdm(chunks, desc="Calculating rolling std")
            ])
            
            rolling_min = pd.concat([
                chunk[value_col].rolling(window=window).min() 
                for chunk in tqdm(chunks, desc="Calculating rolling min")
            ])
            
            rolling_max = pd.concat([
                chunk[value_col].rolling(window=window).max() 
                for chunk in tqdm(chunks, desc="Calculating rolling max")
            ])
        else:
            rolling_mean = df[value_col].rolling(window=window).mean()
            rolling_std = df[value_col].rolling(window=window).std()
            rolling_min = df[value_col].rolling(window=window).min()
            rolling_max = df[value_col].rolling(window=window).max()
        
        return rolling_mean, rolling_std, rolling_min, rolling_max
    
    except Exception as e:
        st.error(f"Error calculating rolling stats: {e}")
        return None, None, None, None

def plot_large_data(df, x_col, y_col, title):
    """Efficient plotting for large datasets"""
    try:
        # Sample data if too large
        if len(df) > 10000:
            sample_size = min(10000, len(df))
            step = len(df) // sample_size
            plot_df = df.iloc[::step].copy()
        else:
            plot_df = df.copy()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df[x_col],
            y=plot_df[y_col],
            mode='lines',
            line=dict(color=PROFESSIONAL_PALETTE['accent'])
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            plot_bgcolor=PROFESSIONAL_PALETTE['background'],
            paper_bgcolor=PROFESSIONAL_PALETTE['background'],
            font_color=PROFESSIONAL_PALETTE['text']
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error plotting data: {e}")
        return None

def time_series_analysis(df, date_col, value_col):
    tracker = st.session_state.tracker
    """Enhanced time series analysis optimized for large datasets"""
    try:
        # Make a copy of the original dataframe to avoid modifying the session state
        df = df.copy()
        
        # Initial preprocessing to ensure proper datetime index
        with st.spinner("Initializing time series data..."):
            try:
                # Convert date column to datetime
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=[date_col])
                
                # Sort by date and set index
                df = df.sort_values(date_col)
                df.set_index(date_col, inplace=True)
                
                # Select only the value column to reduce memory usage
                if value_col in df.columns:
                    df = df[[value_col]]
                else:
                    st.error(f"Value column '{value_col}' not found in dataframe")
                    return df
            except Exception as e:
                st.error(f"Initial preprocessing error: {e}")
                return df
        
        # Data Overview Section
            st.markdown("📊 Data Overview")
            st.metric("Data Points", len(df))
            st.metric("Date Range", f"{df.index.min().date()} to {df.index.max().date()}")
            st.metric("Missing Values", df.isnull().sum().sum())

        with st.expander("📋 Data Overview", expanded=True):
            st.dataframe(df.head(100), use_container_width=True)
        
        # Apply preprocessing
        df = preprocess_time_series(df, value_col)
        
        st.markdown("### 📈 Time Series Analysis")
        st.write(f"**Analyzing:** `{value_col}` over `{date_col}`")
        if not df.empty:
            st.write(f"**Data Points:** {len(df):,} | **Time Range:** {df.index.min().date()} to {df.index.max().date()}")
            
        # Create radio navigation for different analysis sections
        analysis_section = st.radio(
            "Select Analysis Section:",
            ["📊 Overview", "🔍 Decomposition", "📈 Forecasting", "⚠️ Anomaly Detection", "🛠 Feature Engineering"],
            horizontal=True,
            label_visibility="collapsed"
        )
        tracker.log_tab(analysis_section)
        
        if analysis_section == "📊 Overview":
            tracker.log_operation("Viewed Time Series Overview")
            st.markdown("#### Basic Statistics")
            stats = df[value_col].describe().to_frame('Statistics')
            st.write(stats)
            
            # Rolling statistics with progress
            st.markdown("#### Rolling Statistics")
            rolling_window = st.slider(
                "Select Rolling Window Size", 
                min_value=1, 
                max_value=min(365, len(df)), 
                value=7,
                key='rolling_window_overview'
            )
            
            with st.spinner("Calculating rolling statistics..."):
                rolling_mean, rolling_std, rolling_min, rolling_max = efficient_rolling_stats(
                    df, value_col, rolling_window
                )
                
                if rolling_mean is not None:
                    df[f'{value_col}_Rolling_Avg'] = rolling_mean
                    df[f'{value_col}_Rolling_Std'] = rolling_std
                    df[f'{value_col}_Rolling_Min'] = rolling_min
                    df[f'{value_col}_Rolling_Max'] = rolling_max
            
            # Plot original and rolling average
            with st.spinner("Generating plot..."):
                fig1 = plot_large_data(
                    df.reset_index(), 
                    date_col, 
                    value_col, 
                    "Time Series with Rolling Statistics"
                )
                
                if fig1 is not None:
                    # Add rolling average trace
                    fig1.add_trace(go.Scatter(
                        x=df.index,
                        y=df[f'{value_col}_Rolling_Avg'],
                        name=f"{rolling_window}-period Rolling Avg",
                        line=dict(color=PROFESSIONAL_PALETTE['secondary'])
                    ))
                    
                    # Add std deviation bands
                    fig1.add_trace(go.Scatter(
                        x=df.index, 
                        y=df[f'{value_col}_Rolling_Avg'] + df[f'{value_col}_Rolling_Std'],
                        name="Upper Bound",
                        line=dict(width=0),
                        showlegend=False
                    ))
                    fig1.add_trace(go.Scatter(
                        x=df.index, 
                        y=df[f'{value_col}_Rolling_Avg'] - df[f'{value_col}_Rolling_Std'],
                        name="Lower Bound",
                        line=dict(width=0),
                        fillcolor='rgba(231, 76, 60, 0.2)',
                        fill='tonexty',
                        showlegend=False
                    ))
                    
                    fig1.update_layout(showlegend=True)
                    st.plotly_chart(fig1, use_container_width=True)
            
            # Stationarity Test (ADF)
            st.markdown("#### Stationarity Test (ADF)")
            with st.spinner("Performing ADF test..."):
                adf_result = adfuller(df[value_col].dropna())
                st.write(f"**ADF Statistic:** {adf_result[0]:.4f}")
                st.write(f"**p-value:** {adf_result[1]:.4f}")
                if adf_result[1] < 0.05:
                    st.success("The series is likely stationary (p < 0.05).")
                else:
                    st.warning("The series is likely non-stationary (p >= 0.05). Consider differencing or detrending.")
            
            # Differencing
            if st.checkbox("Show Differenced Series", key='show_diff_series'):
                diff_order = st.slider(
                    "Differencing Order", 
                    1, 3, 1,
                    key='diff_order_overview'
                )
                
                with st.spinner("Calculating differenced series..."):
                    diff_series = df[value_col].diff(periods=diff_order).dropna()
                    
                    fig_diff = plot_large_data(
                        diff_series.reset_index(),
                        date_col,
                        value_col,
                        f"{diff_order}-Order Differenced Series"
                    )
                    
                    if fig_diff is not None:
                        fig_diff.update_layout(yaxis_title=f"{value_col} (Differenced)")
                        st.plotly_chart(fig_diff, use_container_width=True)
                    
                    # Test stationarity again
                    adf_diff_result = adfuller(diff_series)
                    st.write(f"**ADF Statistic (Differenced):** {adf_diff_result[0]:.4f}")
                    st.write(f"**p-value (Differenced):** {adf_diff_result[1]:.4f}")

        elif analysis_section == "🔍 Decomposition":
            tracker.log_operation("Viewed Time Series Decomposition")
            # Decomposition
            st.markdown("#### Seasonal Decomposition")
            
            # Check if dataset is too large for full decomposition
            if len(df) > 10000:
                st.warning("Dataset is large. Using a sample for decomposition to improve performance.")
            
            decomp_model = st.radio(
                "Decomposition Model", 
                ['Additive', 'Multiplicative'], 
                index=0,
                key='decomp_model'
            )
            
            period = st.slider(
                "Seasonal Period", 
                2, 365, 12,
                key='seasonal_period'
            )
            
            with st.spinner("Performing decomposition..."):
                decomposition = large_data_decomposition(
                    df[value_col],
                    model='additive' if decomp_model == 'Additive' else 'multiplicative',
                    period=period
                )
                
                if decomposition is not None:
                    fig2 = make_subplots(
                        rows=4, cols=1,
                        subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
                    )

                    # Observed
                    fig2.add_trace(
                        go.Scatter(
                            x=decomposition.observed.index,
                            y=decomposition.observed,
                            name="Observed",
                            line=dict(color=PROFESSIONAL_PALETTE['accent'])
                        ),
                        row=1, col=1
                    )

                    # Trend
                    fig2.add_trace(
                        go.Scatter(
                            x=decomposition.trend.index,
                            y=decomposition.trend,
                            name="Trend",
                            line=dict(color=PROFESSIONAL_PALETTE['secondary'])
                        ),
                        row=2, col=1
                    )

                    # Seasonal
                    fig2.add_trace(
                        go.Scatter(
                            x=decomposition.seasonal.index,
                            y=decomposition.seasonal,
                            name="Seasonal",
                            line=dict(color=PROFESSIONAL_PALETTE['highlight'])
                        ),
                        row=3, col=1
                    )

                    # Residual
                    fig2.add_trace(
                        go.Scatter(
                            x=decomposition.resid.index,
                            y=decomposition.resid,
                            name="Residual",
                            line=dict(color=PROFESSIONAL_PALETTE['primary'])
                        ),
                        row=4, col=1
                    )

                    fig2.update_layout(
                        height=800,
                        title_text=f"Time Series Decomposition ({decomp_model} Model)",
                        plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                        paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                        font_color=PROFESSIONAL_PALETTE['text'],
                        showlegend=False
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            # Autocorrelation and Partial Autocorrelation
            st.markdown("#### Autocorrelation & Partial Autocorrelation")
            max_lags = min(60, len(df) // 2)
            lags = st.slider(
                "Number of Lags", 
                5, max_lags, min(30, max_lags),
                key='acf_lags'
            )
            
            with st.spinner("Calculating ACF/PACF..."):
                # Sample data if too large
                if len(df) > 10000:
                    sample_size = min(10000, len(df))
                    step = len(df) // sample_size
                    sample_data = df[value_col].iloc[::step].dropna()
                else:
                    sample_data = df[value_col].dropna()
                
                # Calculate ACF/PACF values
                acf_values = acf(sample_data, nlags=lags, fft=True)  # Using FFT for efficiency
                pacf_values = pacf(sample_data, nlags=lags, method='ywm')

                # Create ACF plot
                acf_fig = go.Figure()
                acf_fig.add_trace(go.Bar(
                    x=list(range(1, lags + 1)),
                    y=acf_values[1:],
                    name="ACF",
                    marker_color=PROFESSIONAL_PALETTE['accent']
                ))
                acf_fig.update_layout(
                    title="Autocorrelation (ACF)",
                    xaxis_title="Lag",
                    yaxis_title="Correlation",
                    plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                    paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                    font_color=PROFESSIONAL_PALETTE['text']
                )
                st.plotly_chart(acf_fig, use_container_width=True)

                # Create PACF plot
                pacf_fig = go.Figure()
                pacf_fig.add_trace(go.Bar(
                    x=list(range(1, lags + 1)),
                    y=pacf_values[1:],
                    name="PACF",
                    marker_color=PROFESSIONAL_PALETTE['secondary']
                ))
                pacf_fig.update_layout(
                    title="Partial Autocorrelation (PACF)",
                    xaxis_title="Lag",
                    yaxis_title="Correlation",
                    plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                    paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                    font_color=PROFESSIONAL_PALETTE['text']
                )
                st.plotly_chart(pacf_fig, use_container_width=True)
        
        elif analysis_section == "📈 Forecasting":
            tracker.log_operation("Viewed Time Series Forecasting")
            # Forecasting
            st.markdown("#### Time Series Forecasting")
            
            # Limit data size for forecasting models
            if len(df) > 10000:
                st.warning("Large dataset detected. Using a sample for forecasting to improve performance.")
                sample_size = min(10000, len(df))
                step = len(df) // sample_size
                forecast_df = df.iloc[::step].copy()
            else:
                forecast_df = df.copy()
            
            forecast_method = st.selectbox(
                "Select Forecasting Method",
                ['Naive', 'Moving Average', 'Exponential Smoothing', 'ARIMA', 'SARIMA', 'Prophet'],
                index=0,
                key='forecast_method'
            )
            
            forecast_periods = st.slider(
                "Forecast periods (future steps)", 
                min_value=1, 
                max_value=365, 
                value=30,
                key='forecast_periods'
            )
            
            test_size = st.slider(
                "Test Size (%) for Evaluation", 
                10, 50, 20,
                key='test_size'
            )
            
            # Split data into train and test
            train_size = int(len(forecast_df) * (1 - test_size/100))
            train, test = forecast_df.iloc[:train_size], forecast_df.iloc[train_size:]
            
            with st.spinner(f"Training {forecast_method} model..."):
                if forecast_method == 'Naive':
                    last_value = train[value_col].iloc[-1]
                    forecast = np.repeat(last_value, forecast_periods)
                    model_name = "Naive Forecast"
                    
                elif forecast_method == 'Moving Average':
                    window = st.slider(
                        "Moving Average Window", 
                        1, 30, 3,
                        key='ma_window'
                    )
                    forecast = train[value_col].rolling(window=window).mean().iloc[-forecast_periods:]
                    model_name = f"{window}-period Moving Average"
                    
                elif forecast_method == 'Exponential Smoothing':
                    try:
                        model = ExponentialSmoothing(
                            train[value_col],
                            trend='add',
                            seasonal='add',
                            seasonal_periods=12
                        ).fit()
                        forecast = model.forecast(forecast_periods)
                        model_name = "Holt-Winters Exponential Smoothing"
                    except Exception as e:
                        st.error(f"Error in Exponential Smoothing: {e}")
                        forecast = np.repeat(train[value_col].mean(), forecast_periods)
                        model_name = "Fallback Mean Forecast"

                elif forecast_method in ['ARIMA', 'SARIMA']:
                    try:
                        order_p = st.slider(
                            "AR Order (p)", 
                            0, 5, 1,
                            key='order_p'
                        )
                        order_d = st.slider(
                            "Difference Order (d)", 
                            0, 2, 1,
                            key='order_d'
                        )
                        order_q = st.slider(
                            "MA Order (q)", 
                            0, 5, 1,
                            key='order_q'
                        )

                        if forecast_method == 'ARIMA':
                            model = ARIMA(
                                train[value_col],
                                order=(order_p, order_d, order_q)
                            ).fit()
                            model_name = f"ARIMA({order_p},{order_d},{order_q})"
                        else:  # SARIMA
                            seasonal_p = st.slider(
                                "Seasonal AR Order (P)", 
                                0, 2, 0,
                                key='seasonal_p'
                            )
                            seasonal_d = st.slider(
                                "Seasonal Difference (D)", 
                                0, 1, 0,
                                key='seasonal_d'
                            )
                            seasonal_q = st.slider(
                                "Seasonal MA Order (Q)", 
                                0, 2, 0,
                                key='seasonal_q'
                            )
                            seasonal_period = st.slider(
                                "Seasonal Period (s)", 
                                4, 24, 12,
                                key='seasonal_period_sarima'
                            )

                            model = SARIMAX(
                                train[value_col],
                                order=(order_p, order_d, order_q),
                                seasonal_order=(seasonal_p, seasonal_d, seasonal_q, seasonal_period)
                            ).fit()
                            model_name = f"SARIMA({order_p},{order_d},{order_q})({seasonal_p},{seasonal_d},{seasonal_q})[{seasonal_period}]"

                        forecast = model.forecast(forecast_periods)

                    except Exception as e:
                        st.error(f"Error in {forecast_method} model: {e}")
                        forecast = np.repeat(train[value_col].mean(), forecast_periods)
                        model_name = "Fallback Mean Forecast"

                elif forecast_method == 'Prophet':
                    try:
                        prophet_df = train[value_col].reset_index()
                        prophet_df.columns = ['ds', 'y']
                        
                        model = Prophet(
                            yearly_seasonality=True,
                            weekly_seasonality=True,
                            daily_seasonality=False
                        )
                        model.fit(prophet_df)
                        
                        future = model.make_future_dataframe(periods=forecast_periods)
                        forecast_result = model.predict(future)
                        forecast = forecast_result['yhat'].iloc[-forecast_periods:]
                        model_name = "Facebook Prophet"
                        
                    except Exception as e:
                        st.error(f"Error in Prophet model: {e}")
                        forecast = np.repeat(train[value_col].mean(), forecast_periods)
                        model_name = "Fallback Mean Forecast"
            
            # Generate future dates
            freq = pd.infer_freq(forecast_df.index)
            future_dates = pd.date_range(forecast_df.index[-1], periods=forecast_periods+1, freq=freq)[1:]
            
            # Plot forecast
            with st.spinner("Generating forecast plot..."):
                fig_forecast = go.Figure()
                
                # Training data
                fig_forecast.add_trace(go.Scatter(
                    x=train.index,
                    y=train[value_col],
                    name="Training Data",
                    line=dict(color=PROFESSIONAL_PALETTE['accent'])
                ))
                
                # Test data (if available)
                if len(test) > 0:
                    fig_forecast.add_trace(go.Scatter(
                        x=test.index,
                        y=test[value_col],
                        name="Actual Test Data",
                        line=dict(color=PROFESSIONAL_PALETTE['primary'])
                    ))
                
                # Forecast
                fig_forecast.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast,
                    name=f"{model_name} Forecast",
                    line=dict(color=PROFESSIONAL_PALETTE['secondary'], dash='dot')
                ))
                
                # Confidence interval (placeholder)
                fig_forecast.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast * 1.1,
                    name="Upper Bound",
                    line=dict(width=0),
                    showlegend=False
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast * 0.9,
                    name="Lower Bound",
                    line=dict(width=0),
                    fillcolor='rgba(231, 76, 60, 0.2)',
                    fill='tonexty',
                    showlegend=False
                ))
                
                fig_forecast.update_layout(
                    title=f"Time Series Forecast - {model_name}",
                    xaxis_title="Date",
                    yaxis_title=value_col,
                    plot_bgcolor=PROFESSIONAL_PALETTE['background'],
                    paper_bgcolor=PROFESSIONAL_PALETTE['background'],
                    font_color=PROFESSIONAL_PALETTE['text']
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
                st.markdown("---")
                st.markdown("### Export Forecast Results")
                
                # Export Forecast Data
                if 'forecast' in locals():
                    forecast_export_df = pd.DataFrame({
                        'Date': future_dates,
                        'Forecast': forecast
                    })
                    
                    st.download_button(
                        label="📥 Download Forecast as CSV",
                        data=forecast_export_df.to_csv(index=False),
                        file_name=f"{value_col}_forecast.csv",
                        mime="text/csv",
                        key='forecast_download'
                    )
                
                # Export Forecast Plot
                if st.button("💾 Download Forecast Plot as PNG"):
                    try:
                        # Create a temporary file in memory
                        img_bytes = fig_forecast.to_image(format="png")
                        
                        # Trigger download immediately
                        st.download_button(
                            label="Download",  # This won't be shown
                            data=img_bytes,
                            file_name=f"{value_col}_forecast.png",
                            mime="image/png",
                            key="instant_download"  # Important for immediate trigger
                        )
                        
                        # Show success message
                        st.success("Plot downloaded successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating plot: {e}")
            
            # Calculate metrics if test data is available
            if len(test) > 0 and len(test) >= forecast_periods:
                actual = test[value_col].iloc[:forecast_periods]
                
                mae = mean_absolute_error(actual, forecast)
                rmse = np.sqrt(mean_squared_error(actual, forecast))
                mape = np.mean(np.abs((actual - forecast) / actual)) * 100
                
                st.markdown("#### Forecast Evaluation Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"{mae:.2f}")
                col2.metric("RMSE", f"{rmse:.2f}")
                col3.metric("MAPE", f"{mape:.2f}%")
        
        elif analysis_section == "⚠️ Anomaly Detection":
            tracker.log_operation("Viewed Time Series Anomaly Detection")
            # Anomaly Detection
            st.markdown("#### Anomaly Detection")
            
            # Check if dataset is too large for anomaly detection
            if len(df) > 10000:
                st.warning("Large dataset detected. Using a sample for anomaly detection.")
                sample_size = min(10000, len(df))
                step = len(df) // sample_size
                anomaly_df = df.iloc[::step].copy()
            else:
                anomaly_df = df.copy()
            
            anomaly_method = st.selectbox(
                "Select Anomaly Detection Method",
                ['Z-Score', 'IQR', 'Isolation Forest', 'Moving Average Deviation'],
                index=0,
                key='anomaly_method'
            )
            
            with st.spinner("Detecting anomalies..."):
                if anomaly_method == 'Z-Score':
                    threshold = st.slider(
                        "Z-Score Threshold", 
                        1.0, 5.0, 3.0,
                        key='z_score_threshold'
                    )
                    mean = anomaly_df[value_col].mean()
                    std = anomaly_df[value_col].std()
                    anomaly_df['Z-Score'] = (anomaly_df[value_col] - mean) / std
                    anomalies = anomaly_df[np.abs(anomaly_df['Z-Score']) > threshold]
                    
                elif anomaly_method == 'IQR':
                    q1 = anomaly_df[value_col].quantile(0.25)
                    q3 = anomaly_df[value_col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - (1.5 * iqr)
                    upper_bound = q3 + (1.5 * iqr)
                    anomalies = anomaly_df[(anomaly_df[value_col] < lower_bound) | (anomaly_df[value_col] > upper_bound)]
                    
                elif anomaly_method == 'Isolation Forest':
                    contamination = st.slider(
                        "Expected Anomaly Fraction", 
                        0.01, 0.5, 0.05,
                        key='contamination_anomaly'
                    )
                    model = IsolationForest(contamination=contamination, n_jobs=-1)
                    preds = model.fit_predict(anomaly_df[[value_col]])
                    anomalies = anomaly_df[preds == -1]
                    
                elif anomaly_method == 'Moving Average Deviation':
                    window = st.slider(
                        "Moving Average Window", 
                        1, 30, 7,
                        key='ma_window_anomaly'
                    )
                    threshold = st.slider(
                        "Deviation Threshold (STD)", 
                        1.0, 5.0, 2.0,
                        key='deviation_threshold'
                    )
                    
                    rolling_mean, rolling_std, _, _ = efficient_rolling_stats(
                        anomaly_df, value_col, window
                    )
                    
                    if rolling_mean is not None:
                        anomaly_df['Moving_Avg'] = rolling_mean
                        anomaly_df['Moving_Std'] = rolling_std
                        anomaly_df['Upper_Bound'] = anomaly_df['Moving_Avg'] + (threshold * anomaly_df['Moving_Std'])
                        anomaly_df['Lower_Bound'] = anomaly_df['Moving_Avg'] - (threshold * anomaly_df['Moving_Std'])
                        
                        anomalies = anomaly_df[
                            (anomaly_df[value_col] > anomaly_df['Upper_Bound']) | 
                            (anomaly_df[value_col] < anomaly_df['Lower_Bound'])
                        ]
            
            # Plot anomalies
            with st.spinner("Generating anomaly plot..."):
                fig_anomalies = plot_large_data(
                    anomaly_df.reset_index(),
                    date_col,
                    value_col,
                    f"Anomaly Detection - {anomaly_method}"
                )
                
                if fig_anomalies is not None:
                    # Add anomalies
                    fig_anomalies.add_trace(go.Scatter(
                        x=anomalies.index,
                        y=anomalies[value_col],
                        mode='markers',
                        name="Anomalies",
                        marker=dict(
                            color=PROFESSIONAL_PALETTE['secondary'],
                            size=8,
                            line=dict(width=1, color='DarkSlateGrey')
                        )
                    ))
                    
                    # Add bounds if using moving average method
                    if anomaly_method == 'Moving Average Deviation' and 'Upper_Bound' in anomaly_df:
                        fig_anomalies.add_trace(go.Scatter(
                            x=anomaly_df.index,
                            y=anomaly_df['Upper_Bound'],
                            name="Upper Bound",
                            line=dict(color=PROFESSIONAL_PALETTE['highlight'], dash='dash')
                        ))
                        fig_anomalies.add_trace(go.Scatter(
                            x=anomaly_df.index,
                            y=anomaly_df['Lower_Bound'],
                            name="Lower Bound",
                            line=dict(color=PROFESSIONAL_PALETTE['highlight'], dash='dash')
                        ))
                    
                    st.plotly_chart(fig_anomalies, use_container_width=True)
            
            # Show anomalies table
            st.markdown("#### Detected Anomalies")
            st.write(anomalies[[value_col]])
        
        elif analysis_section == "🛠 Feature Engineering":
            # Feature Engineering
            st.markdown("#### Time Series Feature Engineering")
            
            # Initialize a list to track created features
            new_features = []
            
            # Date-based features
            if st.checkbox("Extract Date Features", key='extract_date_features'):
                with st.spinner("Adding date features..."):
                    date_features = ['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Quarter', 'IsWeekend']
                    df['Year'] = df.index.year
                    df['Month'] = df.index.month
                    df['Day'] = df.index.day
                    df['DayOfWeek'] = df.index.dayofweek
                    df['DayOfYear'] = df.index.dayofyear
                    df['Quarter'] = df.index.quarter
                    df['IsWeekend'] = df.index.dayofweek >= 5
                    new_features.extend(date_features)

                    # Visualization
                    fig_date = px.histogram(df, x='Month', y=value_col, 
                                          histfunc='avg', title='Monthly Averages')
                    st.plotly_chart(fig_date, use_container_width=True)
                    
                    # Download
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download Date Features (CSV)",
                            data=df[date_features].to_csv(),
                            file_name="date_features.csv",
                            mime="text/csv"
                        )
                    with col2:
                        st.download_button(
                            label="📥 Download Monthly Averages Plot (PNG)",
                            data=fig_date.to_image(format="png"),
                            file_name="monthly_averages.png",
                            mime="image/png"
                        )

            # Lag features
            if st.checkbox("Create Lag Features", key='create_lag_features'):
                n_lags = st.slider("Number of Lag Features", 1, 10, 3, key="lag_features_slider")
                
                with st.spinner(f"Creating {n_lags} lag features..."):
                    lag_features = [f'Lag_{i}' for i in range(1, n_lags+1)]
                    for i in range(1, n_lags+1):
                        df[f'Lag_{i}'] = df[value_col].shift(i)
                    new_features.extend(lag_features)

                    # Visualization
                    fig_lag = go.Figure()
                    for i in range(1, min(4, n_lags+1)):  # Show first 3 lags for clarity
                        fig_lag.add_trace(go.Scatter(
                            x=df.index,
                            y=df[f'Lag_{i}'],
                            name=f'Lag {i}'
                        ))
                    fig_lag.update_layout(title='Lag Features Visualization')
                    st.plotly_chart(fig_lag, use_container_width=True)
                    
                    # Download
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download Lag Features (CSV)",
                            data=df[lag_features].to_csv(),
                            file_name="lag_features.csv",
                            mime="text/csv"
                        )
                    with col2:
                        st.download_button(
                            label="📥 Download Lag Plot (PNG)",
                            data=fig_lag.to_image(format="png"),
                            file_name="lag_visualization.png",
                            mime="image/png"
                        )

            # Rolling statistics
            if st.checkbox("Create Rolling Statistics", key='create_rolling_stats'):
                window = st.slider("Rolling Window Size", 2, 30, 7, key="rolling_window_slider")
                
                with st.spinner("Calculating rolling statistics..."):
                    rolling_mean, rolling_std, rolling_min, rolling_max = efficient_rolling_stats(
                        df, value_col, window
                    )
                    
                    if rolling_mean is not None:
                        roll_features = [
                            f'Rolling_Mean_{window}',
                            f'Rolling_Std_{window}',
                            f'Rolling_Min_{window}',
                            f'Rolling_Max_{window}'
                        ]
                        df[f'Rolling_Mean_{window}'] = rolling_mean
                        df[f'Rolling_Std_{window}'] = rolling_std
                        df[f'Rolling_Min_{window}'] = rolling_min
                        df[f'Rolling_Max_{window}'] = rolling_max
                        new_features.extend(roll_features)

                        # Visualization
                        fig_roll = go.Figure()
                        fig_roll.add_trace(go.Scatter(
                            x=df.index,
                            y=df[value_col],
                            name='Original',
                            line=dict(color='blue')
                        ))
                        fig_roll.add_trace(go.Scatter(
                            x=df.index,
                            y=df[f'Rolling_Mean_{window}'],
                            name=f'{window}-period Rolling Mean',
                            line=dict(color='red')
                        ))
                        fig_roll.update_layout(title='Rolling Statistics Visualization')
                        st.plotly_chart(fig_roll, use_container_width=True)
                        
                        # Download
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="📥 Download Rolling Stats (CSV)",
                                data=df[roll_features].to_csv(),
                                file_name="rolling_stats.csv",
                                mime="text/csv"
                            )
                        with col2:
                            st.download_button(
                                label="📥 Download Rolling Plot (PNG)",
                                data=fig_roll.to_image(format="png"),
                                file_name="rolling_visualization.png",
                                mime="image/png"
                            )

            # Differencing
            if st.checkbox("Create Differenced Series", key='create_diff_series'):
                diff_order = st.slider("Differencing Order", 1, 3, 1, key="diff_order_slider")
                
                with st.spinner("Creating differenced series..."):
                    diff_feature = f'Diff_{diff_order}'
                    df[diff_feature] = df[value_col].diff(periods=diff_order)
                    new_features.append(diff_feature)

                    # Visualization
                    fig_diff = px.line(df, y=diff_feature, 
                                     title=f'{diff_order}-Order Differenced Series')
                    st.plotly_chart(fig_diff, use_container_width=True)
                    
                    # Download
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download Differenced Data (CSV)",
                            data=df[[diff_feature]].to_csv(),
                            file_name="differenced_series.csv",
                            mime="text/csv"
                        )
                    with col2:
                        st.download_button(
                            label="📥 Download Differenced Plot (PNG)",
                            data=fig_diff.to_image(format="png"),
                            file_name="differenced_series.png",
                            mime="image/png"
                        )

            # Final download option for all engineered features
            if new_features:
                st.markdown("---")
                st.markdown("### Export All Engineered Features")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download All Features (CSV)",
                        data=df[new_features].to_csv(),
                        file_name="all_engineered_features.csv",
                        mime="text/csv"
                    )
                with col2:
                    st.download_button(
                        label="📥 Download Complete Dataset (CSV)",
                        data=df.to_csv(),
                        file_name="complete_dataset.csv",
                        mime="text/csv"
                    )


        # Clean up memory
        gc.collect()
        return df
    
    except Exception as e:
        st.error(f"Error performing time series analysis: {e}")
        return df

def render_time_series_section():
    tracker = st.session_state.tracker  # Ensure tracker is available
    session_manager = get_session_manager()  # Get session manager instance
    section = "Time Series"  # Define section name

    tracker.log_section(section)
    st.markdown("## ⏳ Advanced Time Series Analysis")
    st.markdown("Analyze, forecast, and visualize trends, patterns, seasonality, and anomalies in time-based data.")

    # --- File uploader for CSV/Excel only ---
    with st.expander("📁 Upload Time Series Data (CSV/Excel only)", expanded=True):
        # Use the session manager's file upload handler
        uploaded_file = handle_file_upload(
            section=section,
            file_types=['csv', 'xlsx', 'xls'],
            title="Upload a CSV or Excel file",
            help_text="Only structured CSV or Excel files with rows and columns are supported."
        )

        # --- Clear Data & Operations Button ---
        if st.button("🗑️ Clear Data & Operations", key="clear_ts_data"):
            # Use session manager to clear section data
            session_manager.clear_section_data(section)
            # Clear tracker logs
            if hasattr(tracker, 'logs'):
                tracker.logs.clear()
            # Stay in Time Series section
            st.session_state.selected_section = section
            st.rerun()

        # Display file info if processed
        if session_manager.get_data(section, 'file_processed', False):
            display_file_info(section)

    # --- Proceed if data is loaded ---
    df = session_manager.get_dataframe(section)
    if df is not None:
        # Find datetime columns
        datetime_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns

        # Try to parse date if no datetime columns found
        if len(datetime_cols) == 0:
            for col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    datetime_cols = [col]
                    break
                except Exception:
                    continue
            # Update session state with parsed datetime
            session_manager.set_data(section, 'df', df)

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns

        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            with st.expander("📅 Time Series Setup", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    date_col = st.selectbox("Select Date Column", datetime_cols, key="date_col")
                with col2:
                    value_col = st.selectbox("Select Value Column", numeric_cols, key="value_col")
                tracker.log_operation("Selected date and value columns for time series")

            # Show dataset size warning if too large
            if len(df) > 10000 or len(df.columns) > 50:
                st.warning(f"Large dataset detected ({len(df):,} rows, {len(df.columns)} columns). Some operations may be slower.")
                tracker.log_operation("Displayed large dataset warning")

            # Run analysis with progress
            with st.spinner("Performing time series analysis..."):
                time_series_analysis(df, date_col, value_col)

        else:
            st.info("The dataset requires at least one datetime column and one numeric column for time series analysis.")
            tracker.log_operation("Not enough columns for time series analysis")
    else:
        st.info("Please upload and load a dataset first to use time series analysis features.")
        tracker.log_operation("Prompted user to upload data for time series")
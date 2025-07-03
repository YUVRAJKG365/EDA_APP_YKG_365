import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from fuzzywuzzy import process, fuzz

import tracker

def clean_data(df, cleaning_method, fill_value, drop_axis, remove_duplicates=False, 
              remove_outliers_col=None, convert_dtype_col=None, convert_dtype_target=None,
              trim_whitespace=False, text_case=None, scaling_method=None, encoding_method=None,
              parse_dates=False, handle_inconsistent_categories=False, string_operations=None):
    """
    Comprehensive data cleaning and preprocessing function
    """
    # Handle missing values
    if cleaning_method == "Drop Missing Values":
        if drop_axis == "rows":
            df = df.dropna(axis=0)
        else:
            df = df.dropna(axis=1)
    elif cleaning_method == "Fill with Value":
        df = df.fillna(fill_value)
    elif cleaning_method == "Fill with Mean":
        df = df.fillna(df.select_dtypes(include=np.number).mean())
    elif cleaning_method == "Fill with Median":
        df = df.fillna(df.select_dtypes(include=np.number).median())
    elif cleaning_method == "Fill with Mode":
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else np.nan)

    # Remove duplicates
    if remove_duplicates:
        df = df.drop_duplicates()

    # Remove outliers (IQR method)
    if remove_outliers_col and remove_outliers_col in df.columns and pd.api.types.is_numeric_dtype(df[remove_outliers_col]):
        Q1 = df[remove_outliers_col].quantile(0.25)
        Q3 = df[remove_outliers_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[remove_outliers_col] >= lower_bound) & (df[remove_outliers_col] <= upper_bound)]

    # Convert data types
    if convert_dtype_col and convert_dtype_target:
        try:
            if convert_dtype_target == "int":
                df[convert_dtype_col] = pd.to_numeric(df[convert_dtype_col], errors='coerce').astype('Int64')
            elif convert_dtype_target == "float":
                df[convert_dtype_col] = pd.to_numeric(df[convert_dtype_col], errors='coerce')
            elif convert_dtype_target == "str":
                df[convert_dtype_col] = df[convert_dtype_col].astype(str)
            elif convert_dtype_target == "category":
                df[convert_dtype_col] = df[convert_dtype_col].astype('category')
        except Exception as e:
            st.error(f"Error converting column type: {e}")

    # Trim whitespace
    if trim_whitespace:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()

    # Text case standardization
    if text_case:
        for col in df.select_dtypes(include=['object']).columns:
            if text_case == "lower":
                df[col] = df[col].str.lower()
            elif text_case == "upper":
                df[col] = df[col].str.upper()
            elif text_case == "title":
                df[col] = df[col].str.title()
                
    # Handle inconsistent categories
    if handle_inconsistent_categories:
        for col in df.select_dtypes(include=['object', 'category']).columns:
            # Get unique values and find fuzzy matches
            unique_vals = df[col].unique()
            clusters = {}
            
            for val in unique_vals:
                if pd.isna(val):
                    continue
                    
                matched = False
                for cluster_key in clusters.keys():
                    if fuzz.token_sort_ratio(str(val), str(cluster_key)) > 85:  # 85% similarity threshold
                        clusters[cluster_key].append(val)
                        matched = True
                        break
                
                if not matched:
                    clusters[val] = [val]
            
            # Create mapping for replacement
            mapping = {}
            for cluster_key, cluster_values in clusters.items():
                for value in cluster_values:
                    mapping[value] = cluster_key
                    
            df[col] = df[col].map(mapping).fillna(df[col])

    # Advanced string operations
    if string_operations:
        for col in df.select_dtypes(include=['object']).columns:
            if 'remove_special_chars' in string_operations:
                df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)) if pd.notna(x) else x)
            if 'remove_numbers' in string_operations:
                df[col] = df[col].apply(lambda x: re.sub(r'\d+', '', str(x)) if pd.notna(x) else x)
            if 'remove_extra_spaces' in string_operations:
                df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip() if pd.notna(x) else x)
    
    # Parse dates and extract features
    if parse_dates:
        for col in df.select_dtypes(include=['object', 'datetime64']).columns:
            try:
                # Try to convert to datetime
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Extract datetime features
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_minute'] = df[col].dt.minute
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df[f'{col}_quarter'] = df[col].dt.quarter
            except:
                continue

    # Feature scaling
    if scaling_method and scaling_method != "None":
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols:
            if scaling_method == "StandardScaler":
                scaler = StandardScaler()
                df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            elif scaling_method == "MinMaxScaler":
                scaler = MinMaxScaler()
                df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Categorical encoding
    if encoding_method and encoding_method != "None":
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            if encoding_method == "One-Hot Encoding":
                df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
            elif encoding_method == "Label Encoding":
                le = LabelEncoder()
                for col in categorical_cols:
                    df[col] = le.fit_transform(df[col].astype(str))

    return df

def detect_anomalies(df, method, columns=None, contamination=0.05):
    """
    Detect anomalies using various methods
    Returns: Tuple (anomaly_mask, summary_df)
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    if not columns:
        return pd.DataFrame(), pd.DataFrame()
    
    # Create a copy for numerical columns only
    df_numeric = df[columns].copy()
    
    # Initialize anomaly mask
    anomaly_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    summary_data = []
    
    if method == "IQR":
        for col in columns:
            if col in df_numeric.columns:
                Q1 = df_numeric[col].quantile(0.25)
                Q3 = df_numeric[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                col_mask = (df_numeric[col] < lower_bound) | (df_numeric[col] > upper_bound)
                anomaly_mask[col] = col_mask
                
                # Add to summary
                num_anomalies = col_mask.sum()
                if num_anomalies > 0:
                    summary_data.append({
                        'Column': col,
                        'Method': 'IQR',
                        'Anomalies': num_anomalies,
                        'Percentage': f"{num_anomalies/len(df)*100:.2f}%",
                        'Lower Bound': f"{lower_bound:.4f}",
                        'Upper Bound': f"{upper_bound:.4f}"
                    })
    
    elif method == "Z-score":
        for col in columns:
            if col in df_numeric.columns:
                z_scores = np.abs((df_numeric[col] - df_numeric[col].mean()) / df_numeric[col].std())
                col_mask = z_scores > 3
                anomaly_mask[col] = col_mask
                
                # Add to summary
                num_anomalies = col_mask.sum()
                if num_anomalies > 0:
                    summary_data.append({
                        'Column': col,
                        'Method': 'Z-score',
                        'Anomalies': num_anomalies,
                        'Percentage': f"{num_anomalies/len(df)*100:.2f}%",
                        'Mean': f"{df_numeric[col].mean():.4f}",
                        'Std Dev': f"{df_numeric[col].std():.4f}"
                    })
    
    elif method == "Isolation Forest":
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_numeric)
        
        # Train model
        clf = IsolationForest(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(scaled_data)
        
        # Create mask
        row_mask = predictions == -1
        for col in columns:
            anomaly_mask[col] = row_mask
        
        # Add to summary
        num_anomalies = row_mask.sum()
        if num_anomalies > 0:
            summary_data.append({
                'Column': 'Multiple',
                'Method': 'Isolation Forest',
                'Anomalies': num_anomalies,
                'Percentage': f"{num_anomalies/len(df)*100:.2f}%",
                'Contamination': f"{contamination*100}%",
                'Features': ', '.join(columns)
            })
    
    summary_df = pd.DataFrame(summary_data)
    return anomaly_mask, summary_df

def highlight_anomalies(val, anomaly_mask):
    """
    Highlight anomalies in DataFrame display
    """
    if val.name in anomaly_mask.columns and val.index in anomaly_mask.index:
        if anomaly_mask.at[val.index, val.name]:
            return 'background-color: yellow'
    return ''

def render_data_cleaning_section():
    tracker = st.session_state.tracker
    tracker.log_section("Data Cleaning")
    cleaning_msg = st.empty()
    st.markdown("## 🧹 Data Cleaning")
    st.markdown("Clean your dataset by handling missing values, outliers, and anomalies.")

    if (
        'df' in st.session_state
        and st.session_state.df is not None
        and st.session_state.is_structured
    ):
        df = st.session_state.df

        with st.expander("🔍 Data Preview", expanded=True):
            tracker.log_tab("Data Preview")
            st.dataframe(df)
            tracker.log_operation("Displayed data preview")

        with st.expander("🧭 Missing Value Locations", expanded=True):
            tracker.log_tab("Missing Value Locations")
            try:
                missing_locs = df.isnull()
                if missing_locs.values.any():
                    missing_indices = []
                    for col in df.columns:
                        rows_with_na = missing_locs.index[missing_locs[col]].tolist()
                        for row in rows_with_na:
                            missing_indices.append({'Row': row, 'Column': col})
                    if missing_indices:
                        st.dataframe(pd.DataFrame(missing_indices))
                    else:
                        st.success("No missing values found in the dataset.")
                else:
                    st.success("No missing values found in the dataset.")
                tracker.log_operation("Displayed missing value matrix and summary")
            except Exception as e:
                st.error(f"Error displaying missing value locations: {e}")

        # ======================
        # Anomaly Detection
        # ======================
        st.markdown("---")
        with st.expander("🛡️ Professional Anomaly Prevention Guide", expanded=False):
            tracker.log_tab("Professional Anomaly Prevention Guide")
            st.markdown("""
            ## 🛡️ Advanced Anomaly Prevention Framework
            **Level 1: Basic Prevention**
            - **Data Validation Rules:** Enforce data type and range constraints    
            - **Automated Checks:** Implement basic statistical threshold alerts
            - **Data Profiling:** Regular distribution analysis (Pandas Profiling)
            - **Schema Enforcement:** Use JSON Schema or SQL DDL constraints
            **Level 2: Intermediate Safeguards**
            ```python
            # Example: Automated data quality check     
            def data_quality_check(df):
                report = {
                    "missing_values": df.isnull().sum().to_dict(),
                    "value_ranges": {col: (df[col].min(), df[col].max()) for col in df.select_dtypes(include=np.number)},
                    "unique_counts": df.nunique().to_dict()
                }
                return report
            ```
            - **Anomaly Detection Pipelines:** Scheduled batch monitoring
            - **Data Lineage Tracking:** Track data origins and transformations
            - **Statistical Process Control:** Implement control charts for key metrics
            **Level 3: Advanced Protection**
            - **Real-time Monitoring:** Streaming anomaly detection (Apache Kafka + ML)
            - **ML-based Anomaly Detection:** 
                - Time-series forecasting (Prophet, LSTM)
                - Clustering approaches (DBSCAN, HDBSCAN)
            - **Data Contracts:** Enforce producer-consumer agreements
            - **Data Observability Platforms:** Monte Carlo, Anomalo, BigEye
            **Enterprise Tools Comparison:**
            | Tool | Type | Key Features | Best For |
            |------|------|--------------|----------|
            | **Great Expectations** | OSS | Validation, Data Docs | Python-based teams |   
            | **Monte Carlo** | Commercial | E2E observability, Column-level lineage | Cloud data stacks |
            | **Anomalo** | Commercial | Automated anomaly detection | Large enterprises |
            | **AWS Deequ** | OSS | Scalable data validation | AWS environments |
            | **Datafold** | Commercial | Data diffing, Regression detection | CI/CD for data |
            **Best Practices:**
            1. Implement automated data quality gates in pipelines  
            2. Establish data quality SLAs with stakeholders
            3. Use metadata management for anomaly context
            4. Create feedback loops between data producers and consumers
            5. Conduct regular data health assessments
            """)
            tracker.log_operation("Viewed Professional Anomaly Prevention Guide")
        st.markdown("---")
            
        st.markdown("## 🚨 Anomaly Detection")
        st.info("Identify unusual patterns or outliers in your data that may indicate errors or significant events")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            anomaly_method = st.selectbox(
                "Detection Method:",
                ["IQR", "Z-score", "Isolation Forest"],
                key="anomaly_method"
            )
            
            if anomaly_method == "Isolation Forest":
                contamination = st.slider(
                    "Expected Anomaly Fraction:",
                    0.01, 0.5, 0.05, 0.01,
                    key="contamination",
                    help="Expected proportion of outliers in the data"
                )
        
        with col2:
            numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numerical_cols:
                anomaly_cols = st.multiselect(
                    "Select columns for analysis:",
                    df.columns,
                    default=numerical_cols,
                    key="anomaly_cols"
                )
            else:
                st.warning("No numerical columns available for anomaly detection")
                anomaly_cols = []
        
        detect_btn = st.button("Detect Anomalies", key="detect_anomalies")
        
        if detect_btn and anomaly_cols:
            try:
                with st.spinner("Analyzing data for anomalies..."):
                    anomaly_mask, summary_df = detect_anomalies(
                        df, 
                        method=anomaly_method,
                        columns=anomaly_cols,
                        contamination=contamination if anomaly_method == "Isolation Forest" else 0.05
                    )
                
                total_anomalies = anomaly_mask.sum().sum()
                
                if total_anomalies > 0:
                    st.warning(f"🚩 Detected {total_anomalies} anomalies in the dataset")
                    
                    # Show summary table
                    st.markdown("### 📊 Anomaly Summary")
                    st.dataframe(summary_df)
                    
                    # Show highlighted dataframe
                    st.markdown("### 🟡 Anomaly Locations (Highlighted)")
                    st.dataframe(df.style.apply(lambda _: anomaly_mask.applymap(
                        lambda x: 'background-color: yellow' if x else ''), axis=None))
                    
                    # Show anomaly prevention tips
                    st.markdown("### 🛡️ Anomaly Prevention Tips")
                    st.markdown("""
                    **Professional Anomaly Prevention Strategies:**
                    
                    | Technique | Tools | Implementation Level |
                    |-----------|-------|----------------------|
                    | **Data Validation** | Great Expectations, Pandera | Advanced |
                    | **Automated Monitoring** | Monte Carlo, Anomalo | Enterprise |
                    | **Statistical Thresholds** | Custom Python scripts | Intermediate |
                    | **Data Profiling** | Pandas Profiling, Sweetviz | Basic |
                    | **Schema Enforcement** | SQL Constraints, Pydantic | Advanced |
                    
                    **Best Practices:**
                    1. Implement data quality checks at ingestion points
                    2. Set automated anomaly detection alerts
                    3. Establish data quality SLAs
                    4. Use version control for data pipelines
                    5. Regularly profile data distributions
                    """)
                else:
                    st.success("✅ No anomalies detected in the selected columns")
                    tracker.log_operation("Ran anomaly detection")
            except Exception as e:
                st.error(f"Anomaly detection failed: {str(e)}")
        elif detect_btn:
            st.warning("Please select at least one column for anomaly detection")

        st.markdown("## 🧼 Cleaning Options")
        cleaning_method = st.selectbox(
            "Choose a method to handle missing values:",
            ["None", "Drop Missing Values", "Fill with Value", "Fill with Mean", "Fill with Median", "Fill with Mode"],
            key="cleaning_method"
        )

        fill_value = None
        drop_axis = "rows"
        if cleaning_method == "Drop Missing Values":
            drop_axis = st.selectbox(
                "Drop missing values from:",
                ["rows", "columns"],
                key="drop_axis"
            )
        if cleaning_method == "Fill with Value":
            fill_value = st.text_input("Enter the value to fill missing values with:", key="fill_value")

        # ======================
        # Enhanced Cleaning Features
        # ======================
        with st.expander("✨ Advanced Cleaning Options", expanded=False):
            tracker.log_tab("Advanced Cleaning Options")
            st.markdown("**Data Type & Structure**")
            col1, col2 = st.columns(2)
            with col1:
                remove_duplicates = st.checkbox("Remove duplicate rows", value=False)
                convert_types = st.checkbox("Convert column data types", value=False)
                parse_dates = st.checkbox("Parse dates & extract features", value=False)
                
            with col2:
                trim_whitespace = st.checkbox("Trim whitespace from strings", value=False)
                text_case = st.selectbox("Standardize text case:", 
                                         ["None", "lower", "upper", "title"], 
                                         key="text_case")
                handle_inconsistent = st.checkbox("Fix inconsistent categories", value=False)
            
            if convert_types:
                convert_col, convert_target = st.columns(2)
                with convert_col:
                    convert_dtype_col = st.selectbox("Select column:", df.columns, key="dtype_col")
                with convert_target:
                    convert_dtype_target = st.selectbox("Convert to:", 
                                                      ["int", "float", "str", "category"], 
                                                      key="dtype_target")
            
            st.markdown("**String Processing**")
            string_operations = st.multiselect(
                "Apply text cleaning:",
                ["remove_special_chars", "remove_numbers", "remove_extra_spaces"],
                key="string_ops"
            )
            
            outlier_cols = df.select_dtypes(include=['number']).columns.tolist()
            remove_outliers = st.checkbox("Remove outliers (IQR method)", value=False)
            remove_outliers_col = None
            if remove_outliers and outlier_cols:
                remove_outliers_col = st.selectbox("Select column for outlier removal:", 
                                                  outlier_cols, key="outlier_col")
            tracker.log_operation("Viewed advanced cleaning options")
        
        # ======================
        # Data Refining Options
        # ======================
        with st.expander("⚙️ Data Refining", expanded=False):
            tracker.log_tab("Data Refining")
            st.markdown("**Feature Engineering**")
            
            col1, col2 = st.columns(2)
            with col1:
                scaling_method = st.selectbox("Feature Scaling:", 
                                             ["None", "StandardScaler", "MinMaxScaler"], 
                                             key="scaling")
            with col2:
                encoding_method = st.selectbox("Categorical Encoding:", 
                                             ["None", "One-Hot Encoding", "Label Encoding"], 
                                             key="encoding")
            tracker.log_operation("Viewed data refining options")
        
        # --- Single clean button ---
        clean_clicked = st.button("✨ Clean & Refine Data", key="clean_data_btn", type="primary")

        if clean_clicked:
            try:
                with st.spinner("Cleaning and refining data..."):
                    df_cleaned = clean_data(
                        df,
                        cleaning_method,
                        fill_value,
                        drop_axis,
                        remove_duplicates=remove_duplicates,
                        remove_outliers_col=remove_outliers_col,
                        convert_dtype_col=convert_dtype_col if convert_types else None,
                        convert_dtype_target=convert_dtype_target if convert_types else None,
                        trim_whitespace=trim_whitespace,
                        text_case=text_case if text_case != "None" else None,
                        scaling_method=scaling_method,
                        encoding_method=encoding_method,
                        parse_dates=parse_dates,
                        handle_inconsistent_categories=handle_inconsistent,
                        string_operations=string_operations
                    )
                    st.session_state.cleaned_df = df_cleaned
                    st.success("✅ Data cleaned and refined successfully!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 🧹 Before Cleaning")
                        st.write(f"Missing values: {df.isnull().sum().sum()}")
                        st.write(f"Duplicates: {df.duplicated().sum()}")
                        st.dataframe(df.head())
                    with col2:
                        st.markdown("#### ✨ After Cleaning")
                        st.write(f"Missing values: {df_cleaned.isnull().sum().sum()}")
                        st.write(f"Duplicates: {df_cleaned.duplicated().sum()}")
                        st.dataframe(df_cleaned.head())

                    # Download buttons
                    csv = df_cleaned.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Cleaned Data as CSV",
                        data=csv,
                        file_name='cleaned_data.csv',
                        mime='text/csv',
                        key="download_cleaned_csv"
                    )
                    import io
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        df_cleaned.to_excel(writer, index=False, sheet_name='Cleaned Data')
                    excel_buffer.seek(0)
                    st.download_button(
                        label="📥 Download Cleaned Data as Excel",
                        data=excel_buffer,
                        file_name='cleaned_data.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        key="download_cleaned_excel"
                    )
                tracker.log_operation("Cleaned and refined data")
            except Exception as e:
                st.error(f"❌ Error cleaning data: {e}")
    else:
        st.warning("⚠️ Please upload a CSV or Excel file to use Data Cleaning features. Data cleaning is only available for structured tabular data (CSV/Excel).")


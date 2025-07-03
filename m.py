import os
import shutil
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from io import BytesIO
from sklearn.model_selection import KFold, train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                            mean_squared_error, r2_score, precision_score, 
                            recall_score, mean_absolute_error)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                             VotingClassifier, StackingClassifier)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder, CountEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import eli5
from eli5.sklearn import PermutationImportance
import datetime

from xgboost import XGBRegressor

# Enhanced memory optimization with string conversion support
def optimize_memory(df):
    """Optimize DataFrame memory usage using efficient datatypes and convert strings to numeric where possible"""
    for col in df.columns:
        col_type = df[col].dtype

        # Try to convert string columns to numeric if possible
        if col_type == object:
            try:
                converted = pd.to_numeric(df[col], errors='ignore')
                if converted.dtype != object:
                    df[col] = converted
                    col_type = df[col].dtype
                else:
                    df[col] = df[col].astype('category')
            except:
                df[col] = df[col].astype('category')

        if col_type in ['int8', 'int16', 'int32', 'int64']:
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif col_type in ['float8', 'float16', 'float32', 'float64']:
            # Pandas does not have float8, so treat float8 as float16
            df[col] = df[col].astype(np.float16 if col_type in ['float8', 'float16'] else np.float32)

    return df


# Smart data preprocessing
class SmartPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, target_col=None, problem_type=None):
        self.target_col = target_col
        self.problem_type = problem_type
        self.preprocessor = None
        self.feature_names = None
        
    def fit(self, X, y=None):
        # Identify feature types
        numeric_features = X.select_dtypes(include=['int', 'float']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Store feature names
        self.feature_names = list(numeric_features) + list(categorical_features)
        
        # Create transformers based on data characteristics
        numeric_transformer = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler()
        )
        
        # Determine best categorical encoding strategy
        if len(categorical_features) > 0:
            if self.problem_type == 'Classification':
                categorical_transformer = make_pipeline(
                    SimpleImputer(strategy='most_frequent'),
                    OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                )
            else:
                # For regression, use target encoding if target is available
                if y is not None and self.target_col:
                    categorical_transformer = make_pipeline(
                        SimpleImputer(strategy='most_frequent'),
                        TargetEncoder()
                    )
                else:
                    categorical_transformer = make_pipeline(
                        SimpleImputer(strategy='most_frequent'),
                        CountEncoder()
                    )
        
        # Create column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        self.preprocessor.fit(X, y)
        return self
    
    def transform(self, X):
        if self.preprocessor is None:
            raise RuntimeError("Must fit transformer before transforming data")
        return self.preprocessor.transform(X)
    
    def get_feature_names(self):
        if self.preprocessor is None:
            return self.feature_names
        
        # Get feature names from the preprocessor
        numeric_features = self.preprocessor.named_transformers_['num'].steps[-1][1].get_feature_names_out()
        categorical_features = self.preprocessor.named_transformers_['cat'].steps[-1][1].get_feature_names_out()
        return list(numeric_features) + list(categorical_features)

# Large file loader with progress
@st.cache_data(show_spinner=False)
def load_large_file(uploaded_fileml, sample_threshold=100000):
    """Load large files efficiently with sampling option and universal encoding support"""
    try:
        if uploaded_fileml.name.endswith('.csv'):
            # Try utf-8 first, then fallback to latin1, then to ISO-8859-1
            encodings_to_try = ['utf-8', 'latin1', 'ISO-8859-1']
            for encoding in encodings_to_try:
                try:
                    # First pass to count lines
                    line_count = 0
                    for _ in pd.read_csv(uploaded_fileml, chunksize=10000, encoding=encoding):
                        line_count += 10000
                    uploaded_fileml.seek(0)
                    
                    if line_count > sample_threshold:
                        st.warning(f"Large dataset detected (~{line_count} rows). Consider sampling for faster processing.")
                        sample_size = st.slider("Sampling percentage", 1, 100, 10, 
                                              key="sample_slider_" + uploaded_fileml.name)
                        sample_rows = max(1, int(line_count * sample_size / 100))
                        df = pd.read_csv(uploaded_fileml, nrows=sample_rows, encoding=encoding)
                        st.success(f"Loaded {len(df)} sampled rows")
                    else:
                        if line_count > 50000:
                            chunks = []
                            chunk_size = 10000
                            progress_bar = st.progress(0)
                            for i, chunk in enumerate(pd.read_csv(uploaded_fileml, chunksize=chunk_size, encoding=encoding)):
                                chunks.append(chunk)
                                progress = min(100, int((i * chunk_size) / line_count * 100))
                                progress_bar.progress(progress)
                            df = pd.concat(chunks, ignore_index=True)
                            progress_bar.progress(100)
                        else:
                            df = pd.read_csv(uploaded_fileml, encoding=encoding)
                    break  # If successful, break out of encoding loop
                except UnicodeDecodeError:
                    uploaded_fileml.seek(0)
                    continue
            else:
                st.error("Could not decode CSV file with common encodings (utf-8, latin1, ISO-8859-1). Please check your file encoding.")
                return None
        elif uploaded_fileml.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_fileml)
        else:
            st.error("Unsupported file format")
            return None
        
        return optimize_memory(df)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None
    
# Enhanced Model Recommendation System
def get_model_recommendations(df, target_col, problem_type):
    """Generate smart model recommendations based on dataset characteristics"""
    num_rows = len(df)
    num_features = len(df.columns) - 1  # exclude target
    recommendations = {
        "Classification": {
            "Highly Recommended": [],
            "Recommended": [],
            "Optional": []
        },
        "Regression": {
            "Highly Recommended": [],
            "Recommended": [],
            "Optional": []
        }
    }
    
    # Classification recommendations
    if problem_type == "Classification":
        # Check for class imbalance
        class_balance = df[target_col].value_counts(normalize=True)
        is_imbalanced = any(prob < 0.2 for prob in class_balance)
        
        if num_rows < 1000:
            recommendations["Classification"]["Highly Recommended"] = ["Logistic Regression", "Decision Tree"]
            recommendations["Classification"]["Recommended"] = ["Random Forest", "SVM"]
            recommendations["Classification"]["Optional"] = ["KNN", "Naive Bayes"]
        elif num_rows < 100000:
            recommendations["Classification"]["Highly Recommended"] = ["Random Forest", "XGBoost"]
            recommendations["Classification"]["Recommended"] = ["LightGBM", "CatBoost"]
            recommendations["Classification"]["Optional"] = ["SVM", "MLP"]
        else:
            recommendations["Classification"]["Highly Recommended"] = ["LightGBM", "CatBoost"]
            recommendations["Classification"]["Recommended"] = ["XGBoost", "Random Forest"]
            recommendations["Classification"]["Optional"] = ["SVM", "Logistic Regression"]
            
        if is_imbalanced:
            recommendations["Classification"]["Highly Recommended"].append("Balanced Random Forest")
            recommendations["Classification"]["Recommended"].append("Class Weighted Models")
            
    # Regression recommendations
    else:
        if num_rows < 1000:
            recommendations["Regression"]["Highly Recommended"] = ["Linear Regression", "Decision Tree"]
            recommendations["Regression"]["Recommended"] = ["Random Forest", "Lasso"]
            recommendations["Regression"]["Optional"] = ["Ridge", "ElasticNet"]
        elif num_rows < 100000:
            recommendations["Regression"]["Highly Recommended"] = ["Random Forest", "XGBoost"]
            recommendations["Regression"]["Recommended"] = ["LightGBM", "CatBoost"]
            recommendations["Regression"]["Optional"] = ["SVM", "KNN"]
        else:
            recommendations["Regression"]["Highly Recommended"] = ["LightGBM", "CatBoost"]
            recommendations["Regression"]["Recommended"] = ["XGBoost", "Random Forest"]
            recommendations["Regression"]["Optional"] = ["Linear Regression", "Lasso"]
            
        # Check for linearity
        if num_features < 20:  # Simple feature space
            recommendations["Regression"]["Highly Recommended"].append("Linear Regression")
    
    return recommendations

# Alias float8 to float16 for pandas compatibility
def ensure_float8_support(df):
    for col in df.select_dtypes(include=['float8']).columns:
        df[col] = df[col].astype(np.float16)
    return df

# Feature Encoding Suggestions
def get_encoding_suggestions(df, target_col=None):
    """Provide smart encoding suggestions for categorical features"""
    suggestions = {}
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in cat_cols:
        nunique = df[col].nunique()
        
        if nunique > 50:  # High cardinality
            if target_col:
                suggestions[col] = {
                    "recommended": "Target Encoding",
                    "alternatives": ["Frequency Encoding", "Leave-One-Out Encoding"],
                    "warning": "High cardinality - One-Hot Encoding may create too many features"
                }
            else:
                suggestions[col] = {
                    "recommended": "Frequency Encoding",
                    "alternatives": ["Hash Encoding", "Embedding"],
                    "warning": "High cardinality - Consider dimensionality reduction"
                }
        elif 10 < nunique <= 50:  # Medium cardinality
            suggestions[col] = {
                "recommended": "One-Hot Encoding",
                "alternatives": ["Target Encoding", "Label Encoding"],
                "warning": None
            }
        else:  # Low cardinality
            suggestions[col] = {
                "recommended": "One-Hot Encoding",
                "alternatives": ["Label Encoding", "Binary Encoding"],
                "warning": None
            }
    
    return suggestions

# Cross-validation visualization
def plot_cross_val_results(cv_scores, metric_name):
    """Visualize cross-validation results"""
    fig = go.Figure()
    
    # Add box plot
    fig.add_trace(go.Box(
        y=cv_scores,
        name=metric_name,
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        marker=dict(color='rgb(7,40,89)'),
        line=dict(color='rgb(7,40,89)')
    ))
    
    # Add mean line
    fig.add_hline(y=np.mean(cv_scores), line_dash="dot",
                 annotation_text=f"Mean: {np.mean(cv_scores):.3f}",
                 annotation_position="bottom right")
    
    fig.update_layout(
        title=f"Cross-Validation {metric_name} Distribution",
        yaxis_title=metric_name,
        showlegend=False
    )
    
    st.plotly_chart(fig)

# SHAP visualization
def plot_shap_summary(model, X, feature_names):
    """Generate SHAP summary plot with FLAML model support"""
    try:
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Create explainer based on model type
        try:
            # First try TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            # For classification models, we might get a list of arrays (one per class)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Typically use class 1 for binary classification
        except:
            # Fall back to KernelExplainer if TreeExplainer fails
            def model_predict(X):
                return model.predict(X)

            explainer = shap.KernelExplainer(model_predict, shap.sample(X, 100))
            shap_values = explainer.shap_values(X)

        # Summary plot
        st.subheader("SHAP Summary Plot")
        fig_summary, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        st.pyplot(fig_summary)
        plt.close()

        # Dependence plot for top feature
        if len(feature_names) > 1:
            top_feature = np.abs(shap_values).mean(0).argmax()
            st.subheader(f"SHAP Dependence Plot for {feature_names[top_feature]}")
            fig_dep, ax = plt.subplots(figsize=(10, 6))
            shap.dependence_plot(top_feature, shap_values, X,
                                 feature_names=feature_names, show=False, ax=ax)
            st.pyplot(fig_dep)
            plt.close()

        return True
    except Exception as e:
        st.warning(f"Could not generate SHAP plots: {str(e)}")
        return False

# Model stacking function
def create_stacked_model(models, problem_type):
    """Create a stacked ensemble model"""
    if problem_type == "Classification":
        return StackingClassifier(
            estimators=[(f"model_{i}", model) for i, model in enumerate(models)],
            final_estimator=LogisticRegression(max_iter=1000),
            stack_method='predict_proba'
        )
    else:
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import LinearRegression
        return StackingRegressor(
            estimators=[(f"model_{i}", model) for i, model in enumerate(models)],
            final_estimator=LinearRegression()
        )

# Enhanced Standard ML Training Section with new features
def standard_ml_training(df):
    st.subheader("📈 Standard ML Training")

    # Two column selection
    col1, col2 = st.columns(2)
    with col1:
        feature_col = st.selectbox(
            "Select Feature Column",
            [f"{col} ({df[col].dtype})" for col in df.columns],
            key="feature_col_select"
        )
        feature_col = feature_col.split(" (")[0]
    with col2:
        target_col = st.selectbox(
            "Select Target Column",
            [f"{col} ({df[col].dtype})" for col in df.columns],
            key="target_col_select"
        )
        target_col = target_col.split(" (")[0]

    # Problem type detection
    if df[target_col].dtype == 'object' or df[target_col].nunique() < 20:
        problem_type = "Classification"
    else:
        problem_type = "Regression"

    st.info(f"Detected Problem Type: {problem_type}")

    # Show encoding suggestions for categorical features
    if st.checkbox("Show Feature Encoding Suggestions", False):
        encoding_suggestions = get_encoding_suggestions(df[[feature_col]], target_col)
        if encoding_suggestions:
            st.subheader("Encoding Recommendations")
            for col, suggestion in encoding_suggestions.items():
                with st.expander(f"Column: {col}"):
                    st.success(f"Recommended: {suggestion['recommended']}")
                    st.write(f"Alternatives: {', '.join(suggestion['alternatives'])}")
                    if suggestion['warning']:
                        st.warning(suggestion['warning'])
        else:
            st.info("No categorical features to encode")

    # Model recommendation system
    st.subheader("Model Recommendations")
    model_recs = get_model_recommendations(df, target_col, problem_type)

    rec_col1, rec_col2, rec_col3 = st.columns(3)
    with rec_col1:
        st.markdown("**Highly Recommended**")
        for model in model_recs[problem_type]["Highly Recommended"]:
            st.success(f"✅ {model}")
    with rec_col2:
        st.markdown("**Recommended**")
        for model in model_recs[problem_type]["Recommended"]:
            st.info(f"🔹 {model}")
    with rec_col3:
        st.markdown("**Optional**")
        for model in model_recs[problem_type]["Optional"]:
            st.warning(f"⚙️ {model}")

    # Metric selection
    st.subheader("Evaluation Metrics")
    if problem_type == "Classification":
        metrics = st.multiselect("Select metrics to evaluate", 
                               ["Accuracy", "F1", "ROC AUC", "Precision", "Recall"],
                               default=["Accuracy", "F1"])
    else:
        metrics = st.multiselect("Select metrics to evaluate", 
                               ["R2", "MSE", "MAE", "RMSE"],
                               default=["R2", "MSE"])

    # Add XGBoost to unique_models if not present
    unique_models = []
    seen_models = set()
    for model in (model_recs[problem_type]["Highly Recommended"] + 
                 model_recs[problem_type]["Recommended"] + 
                 model_recs[problem_type]["Optional"]):
        if model not in seen_models:
            unique_models.append(model)
            seen_models.add(model)
    if "XGBoost" not in unique_models:
        unique_models.append("XGBoost")

    # Model selection - now with unique names
    selected_model = st.selectbox("Select Model", unique_models)

    # Advanced options
    with st.expander("Advanced Options"):
        use_cv = st.checkbox("Use Cross-Validation", True)
        cv_folds = st.slider("Number of CV folds", 3, 10, 5)
        
        if problem_type == "Classification":
            handle_imbalance = st.checkbox("Handle Class Imbalance", True, key="handle_imbalance_standard")

        if st.checkbox("Enable Model Stacking", False):
            num_models_to_stack = st.slider("Number of models to stack", 2, 5, 3)
            models_to_stack = st.multiselect("Select models to stack", 
                                           unique_models,
                                           default=unique_models[:num_models_to_stack])

    # Train-test split with visualization
    st.subheader("Train-Test Split")
    test_size = st.slider("Test set size (%)", 10, 40, 20)

    X = df[[feature_col]]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    # Visualize train-test distribution
    if problem_type == "Classification":
        train_dist = y_train.value_counts(normalize=True).reset_index()
        test_dist = y_test.value_counts(normalize=True).reset_index()
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(
            x=train_dist[target_col],
            y=train_dist['proportion'],
            name='Train',
            marker_color='blue'
        ))
        fig_dist.add_trace(go.Bar(
            x=test_dist[target_col],
            y=test_dist['proportion'],
            name='Test',
            marker_color='red'
        ))
        fig_dist.update_layout(
            title="Class Distribution in Train/Test Sets",
            xaxis_title=target_col,
            yaxis_title="Proportion",
            barmode='group'
        )
    else:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=y_train,
            name='Train',
            marker_color='blue',
            opacity=0.75
        ))
        fig_dist.add_trace(go.Histogram(
            x=y_test,
            name='Test',
            marker_color='red',
            opacity=0.75
        ))
        fig_dist.update_layout(
            title="Target Distribution in Train/Test Sets",
            xaxis_title=target_col,
            yaxis_title="Count",
            barmode='overlay'
        )
    
    st.plotly_chart(fig_dist)
    
    # Model training
    if st.button("🚀 Train Model"):
        with st.spinner(f"Training {selected_model}..."):
            try:
                # Initialize model based on selection
                if selected_model == "Linear Regression":
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                elif selected_model == "Decision Tree":
                    if problem_type == "Classification":
                        from sklearn.tree import DecisionTreeClassifier
                        model = DecisionTreeClassifier()
                    else:
                        from sklearn.tree import DecisionTreeRegressor
                        model = DecisionTreeRegressor()
                elif selected_model == "Random Forest":
                    if problem_type == "Classification":
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(class_weight='balanced' if handle_imbalance else None)
                    else:
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor()
                elif selected_model == "XGBoost":
                    if problem_type == "Classification":
                        from xgboost import XGBClassifier
                        model = XGBClassifier(scale_pos_weight=sum(y==0)/sum(y==1) if handle_imbalance else 1)
                    else:
                        from xgboost import XGBRegressor
                        model = XGBRegressor()
                elif selected_model == "LightGBM":
                    if problem_type == "Classification":
                        from lightgbm import LGBMClassifier
                        model = LGBMClassifier(class_weight='balanced' if handle_imbalance else None)
                    else:
                        from lightgbm import LGBMRegressor
                        model = LGBMRegressor()
                elif selected_model == "Logistic Regression":
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(class_weight='balanced' if handle_imbalance else None)
                elif selected_model == "SVM":
                    if problem_type == "Classification":
                        from sklearn.svm import SVC
                        model = SVC(class_weight='balanced' if handle_imbalance else None, probability=True)
                    else:
                        from sklearn.svm import SVR
                        model = SVR()
                elif selected_model == "KNN":
                    if problem_type == "Classification":
                        from sklearn.neighbors import KNeighborsClassifier
                        model = KNeighborsClassifier()
                    else:
                        from sklearn.neighbors import KNeighborsRegressor
                        model = KNeighborsRegressor()
                elif selected_model == "Lasso":
                    from sklearn.linear_model import Lasso
                    model = Lasso()
                elif selected_model == "Ridge":
                    from sklearn.linear_model import Ridge
                    model = Ridge()
                elif selected_model == "ElasticNet":
                    from sklearn.linear_model import ElasticNet
                    model = ElasticNet()
                elif selected_model == "CatBoost":
                    if problem_type == "Classification":
                        from catboost import CatBoostClassifier
                        model = CatBoostClassifier(verbose=0, auto_class_weights='Balanced' if handle_imbalance else None)
                    else:
                        from catboost import CatBoostRegressor
                        model = CatBoostRegressor(verbose=0)
                elif selected_model == "Balanced Random Forest":
                    from imblearn.ensemble import BalancedRandomForestClassifier
                    model = BalancedRandomForestClassifier()
                elif selected_model == "Naive Bayes":
                    from sklearn.naive_bayes import GaussianNB
                    model = GaussianNB()
                elif selected_model == "MLP":
                    if problem_type == "Classification":
                        from sklearn.neural_network import MLPClassifier
                        model = MLPClassifier()
                    else:
                        from sklearn.neural_network import MLPRegressor
                        model = MLPRegressor()
                
                # Train model
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Cross-validation if enabled
                cv_scores = {}
                if use_cv:
                    st.subheader("Cross-Validation Results")
                    
                    if problem_type == "Classification":
                        scoring_metrics = {
                            'accuracy': 'Accuracy',
                            'f1_weighted': 'F1',
                            'roc_auc': 'ROC AUC'
                        }
                    else:
                        scoring_metrics = {
                            'r2': 'R2',
                            'neg_mean_squared_error': 'MSE'
                        }
                    
                    for metric, name in scoring_metrics.items():
                        scores = cross_val_score(model, X, y, cv=cv_folds, scoring=metric)
                        cv_scores[name] = scores
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"Mean CV {name}", f"{np.mean(scores):.4f}")
                        with col2:
                            st.metric(f"Std CV {name}", f"{np.std(scores):.4f}")
                        
                        plot_cross_val_results(scores, name)
                
                # Store model in session state
                st.session_state.standard_model = model
                st.session_state.standard_model_type = selected_model
                st.session_state.X_train_std = X_train
                st.session_state.y_train_std = y_train
                st.session_state.X_test_std = X_test
                st.session_state.y_test_std = y_test
                st.session_state.feature_col = feature_col
                st.session_state.target_col_std = target_col
                st.session_state.problem_type_std = problem_type
                st.session_state.cv_scores = cv_scores
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                st.subheader("Test Set Performance")
                metric_values = {}
                
                if problem_type == "Classification":
                    if "Accuracy" in metrics:
                        accuracy = accuracy_score(y_test, y_pred)
                        metric_values["Accuracy"] = accuracy
                        st.metric("Accuracy", f"{accuracy:.4f}")
                    
                    if "F1" in metrics:
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        metric_values["F1"] = f1
                        st.metric("F1 Score", f"{f1:.4f}")
                    
                    if "ROC AUC" in metrics and len(np.unique(y_test)) == 2:
                        roc_auc = roc_auc_score(y_test, y_pred)
                        metric_values["ROC AUC"] = roc_auc
                        st.metric("ROC AUC", f"{roc_auc:.4f}")
                    
                    if "Precision" in metrics:
                        precision = precision_score(y_test, y_pred, average='weighted')
                        metric_values["Precision"] = precision
                        st.metric("Precision", f"{precision:.4f}")
                    
                    if "Recall" in metrics:
                        recall = recall_score(y_test, y_pred, average='weighted')
                        metric_values["Recall"] = recall
                        st.metric("Recall", f"{recall:.4f}")
                else:
                    if "R2" in metrics:
                        r2 = r2_score(y_test, y_pred)
                        metric_values["R2"] = r2
                        st.metric("R² Score", f"{r2:.4f}")
                    
                    if "MSE" in metrics:
                        mse = mean_squared_error(y_test, y_pred)
                        metric_values["MSE"] = mse
                        st.metric("Mean Squared Error", f"{mse:.4f}")
                    
                    if "MAE" in metrics:
                        mae = mean_absolute_error(y_test, y_pred)
                        metric_values["MAE"] = mae
                        st.metric("Mean Absolute Error", f"{mae:.4f}")
                    
                    if "RMSE" in metrics:
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        metric_values["RMSE"] = rmse
                        st.metric("Root Mean Squared Error", f"{rmse:.4f}")
                
                st.success(f"Model trained successfully in {training_time:.2f} seconds!")
                
                # Plot actual vs predicted
                st.subheader("Actual vs Predicted Results")
                fig = go.Figure()
                
                # Add actual values as scatter plot
                fig.add_trace(go.Scatter(
                    x=X_test[feature_col],
                    y=y_test,
                    mode='markers',
                    name='Actual',
                    marker=dict(color='blue', size=8)
                ))
                
                # Add predicted values as line plot
                sorted_idx = np.argsort(X_test[feature_col].values.ravel())
                fig.add_trace(go.Scatter(
                    x=X_test[feature_col].iloc[sorted_idx],
                    y=y_pred[sorted_idx],
                    mode='lines',
                    name='Predicted',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title=f"{selected_model} Predictions vs Actual",
                    xaxis_title=feature_col,
                    yaxis_title=target_col,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig)
                
                # Download plot button
                plot_bytes = fig.to_image(format="png")
                st.download_button(
                    label="Download Plot as PNG",
                    data=plot_bytes,
                    file_name=f"{selected_model}_predictions.png",
                    mime="image/png"
                )
                
                # Feature importance if available
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    importance = model.feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': [feature_col],
                        'Importance': importance
                    })
                    
                    fig_imp = px.bar(importance_df, x='Feature', y='Importance', 
                                     title="Feature Importance Score")
                    st.plotly_chart(fig_imp)
                
                # SHAP explainability
                if st.checkbox("Show SHAP Explainability", True):
                    try:
                        # Convert X_test to DataFrame if it's not already
                        if not isinstance(X_test, pd.DataFrame):
                            X_test_df = pd.DataFrame(X_test, columns=[feature_col])
                        else:
                            X_test_df = X_test.copy()
                        
                        # Handle categorical features
                        if X_test_df[feature_col].dtype == 'object':
                            # For demo purposes, use label encoding
                            le = LabelEncoder()
                            X_test_df[feature_col] = le.fit_transform(X_test_df[feature_col])
                        
                        plot_shap_summary(model, X_test_df.values, [feature_col])
                    except Exception as e:
                        st.warning(f"SHAP explanation not available: {str(e)}")
                
                # Model download
                st.subheader("Model Export")
                export_format = st.radio("Export format", ["joblib", "ONNX"], horizontal=True)
                
                if export_format == "joblib":
                    model_bytes = BytesIO()
                    joblib.dump(model, model_bytes)
                    st.download_button(
                        label="Download Trained Model",
                        data=model_bytes.getvalue(),
                        file_name=f"{selected_model}_{problem_type}.joblib",
                        mime="application/octet-stream"
                    )
                else:
                    try:
                        from skl2onnx import convert_sklearn
                        from skl2onnx.common.data_types import FloatTensorType
                        
                        initial_type = [('float_input', FloatTensorType([None, 1]))]
                        onx = convert_sklearn(model, initial_types=initial_type)
                        
                        st.download_button(
                            label="Download ONNX Model",
                            data=onx.SerializeToString(),
                            file_name=f"{selected_model}_{problem_type}.onnx",
                            mime="application/octet-stream"
                        )
                    except Exception as e:
                        st.error(f"ONNX conversion failed: {str(e)}")
                
                # Generate inference code snippet
                st.subheader("Inference Code")
                inference_code = f"""
# Python code to load and use your trained model
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('{selected_model}_{problem_type}.joblib')

# Prepare input data
input_data = pd.DataFrame({{
    '{feature_col}': [your_input_value]  # Replace with your input
}})

# Make prediction
prediction = model.predict(input_data)
print(f"Predicted {'class' if problem_type == 'Classification' else 'value'}:", prediction)
"""
                st.code(inference_code, language='python')
                
                # Add to model history
                if 'model_history' not in st.session_state:
                    st.session_state.model_history = []
                
                st.session_state.model_history.append({
                    'timestamp': datetime.datetime.now(),
                    'model_type': selected_model,
                    'problem_type': problem_type,
                    'metrics': metric_values,
                    'training_time': training_time,
                    'features': [feature_col],
                    'target': target_col
                })
                
            except Exception as e:
                st.error(f"Model training failed: {str(e)}")
import re

def sanitize_column_names(df):
    # Replace any special JSON character with underscore
    df = df.rename(columns=lambda x: re.sub(r'[\"\'\\/\{\}\[\]:,]', '_', str(x)))
    return df

def prediction_section(df):
    st.subheader("🚀 Advanced Future Prediction Engine")
    
    if 'ml_data' not in st.session_state:
        st.warning("Please load and process your data in the ML Training section first")
        return
    
    # Filter numeric columns only
    numeric_types = [
        'int8', 'int16', 'int32', 'int64',
        'float16', 'float32', 'float64'
    ]
    numeric_cols = df.select_dtypes(include=numeric_types).columns.tolist()
    
    if not numeric_cols:
        st.error("No numeric columns found for prediction")
        return
    
    # Model selection
    model_options = ["Linear Regression", "XGBoost (Recommended)", "LightGBM (Large Datasets)", 
                    "CatBoost (Accurate)", "Random Forest (Robust)"]
    selected_model = st.selectbox(
        "Select Prediction Model",
        model_options,
        key="pred_model_select"
    )
    
    # Select target and features
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox(
            "Select Target Column for Prediction",
            numeric_cols,
            key="pred_target_select"
        )
    
    with col2:
        available_features = [col for col in numeric_cols if col != target_col]
        if not available_features:
            st.error("No available feature columns found")
            return
        
        feature_cols = st.multiselect(
            "Select Feature Columns (1-3)",
            available_features,
            key="pred_feature_select",
            max_selections=3
        )
    
    if not feature_cols:
        st.error("Please select at least one feature column")
        return
    
    # Advanced options
    advanced_options = st.expander("⚙️ Advanced Model Configuration")
    with advanced_options:
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Validation Set Size (%)", 10, 40, 20)
            n_estimators = st.slider("Number of Estimators", 50, 1000, 300)
            
        with col2:
            learning_rate = st.slider("Learning Rate", 0.001, 1.0, 0.1, step=0.01)
            max_depth = st.slider("Max Depth", 3, 15, 6)
        
        use_cross_val = st.checkbox("Enable Cross-Validation", True)
        early_stopping = st.checkbox("Enable Early Stopping", True)
        hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning", True)
    
    if st.button("🚀 Train Predictive Model with Advanced Analytics") or 'trained_model' in st.session_state:
        if 'trained_model' not in st.session_state or st.button:
            with st.spinner("Training advanced predictive model with deep pattern recognition..."):
                try:
                    # Prepare data - robust handling
                    if not feature_cols:  # Double-check feature selection
                        st.error("No features selected. Please select at least one feature.")
                        return
                        
                    X = df[feature_cols].copy()
                    y = df[target_col].copy()
                    
                    # Data validation checks
                    if X.empty or y.empty:
                        st.error("Selected columns contain no data")
                        return
                        
                    # Handle missing values robustly
                    for col in feature_cols:
                        if X[col].isnull().all():
                            st.error(f"Column '{col}' contains only missing values")
                            return
                        X[col].fillna(X[col].mean(), inplace=True)
                    
                    if y.isnull().all():
                        st.error(f"Target column '{target_col}' contains only missing values")
                        return
                    y.fillna(y.mean(), inplace=True)
                    
                    # Ensure we have sufficient data
                    if len(X) < 10:
                        st.error("Insufficient data for modeling (min 10 samples required)")
                        return

                    # Split data with size validation
                    test_sample_size = max(1, int(len(X) * test_size / 100))
                    if len(X) - test_sample_size < 2:
                        st.error("Insufficient data for training after split")
                        return
                        
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=42
                    )
                    
                    # Store split data in session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                    # Model selection and hyperparameters
                    model = None
                    model_params = {}
                    
                    if selected_model == "Linear Regression":
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression()
                        model_params = {'fit_intercept': [True, False]}
                        
                    elif selected_model.startswith("XGBoost"):
                        from xgboost import XGBRegressor
                        model = XGBRegressor(
                            objective='reg:squarederror',
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            early_stopping_rounds=20 if early_stopping else None,
                            eval_metric='rmse'
                        )
                        model_params = {
                            'learning_rate': [0.01, 0.05, 0.1],
                            'max_depth': [3, 6, 9],
                            'subsample': [0.8, 0.9, 1.0]
                        }
                        
                    elif selected_model.startswith("LightGBM"):
                        from lightgbm import LGBMRegressor
                        model = LGBMRegressor(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            early_stopping_round=20 if early_stopping else None
                        )
                        model_params = {
                            'num_leaves': [31, 63, 127],
                            'min_child_samples': [20, 50, 100],
                            'reg_alpha': [0, 0.1, 1]
                        }
                        
                    elif selected_model.startswith("CatBoost"):
                        from catboost import CatBoostRegressor
                        model = CatBoostRegressor(
                            iterations=n_estimators,
                            learning_rate=learning_rate,
                            depth=max_depth,
                            verbose=0,
                            early_stopping_rounds=20 if early_stopping else None
                        )
                        model_params = {
                            'depth': [4, 6, 8],
                            'l2_leaf_reg': [1, 3, 5],
                            'border_count': [32, 64, 128]
                        }
                        
                    elif selected_model.startswith("Random Forest"):
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42
                        )
                        model_params = {
                            'max_features': ['sqrt', 'log2'],
                            'min_samples_split': [2, 5, 10],
                            'bootstrap': [True, False]
                        }
                    
                    # Hyperparameter tuning
                    best_model = model
                    if hyperparameter_tuning and model_params and len(X_train) > 10:
                        try:
                            from sklearn.model_selection import GridSearchCV
                            grid_search = GridSearchCV(
                                estimator=model,
                                param_grid=model_params,
                                cv=min(3, len(X_train)//2),  # Dynamic CV folds
                                n_jobs=-1,
                                scoring='neg_mean_squared_error'
                            )
                            grid_search.fit(X_train, y_train)
                            best_model = grid_search.best_estimator_
                            st.success(f"Best parameters: {grid_search.best_params_}")
                        except Exception as e:
                            st.warning(f"Hyperparameter tuning skipped: {str(e)}")
                            best_model = model.fit(X_train, y_train)
                    else:
                        best_model = model.fit(X_train, y_train)
                    
                    # Store the trained model and metrics in session state
                    st.session_state.trained_model = best_model
                    st.session_state.feature_cols = feature_cols
                    
                    # Cross-validation
                    cv_scores = None
                    if use_cross_val and len(X_train) > 5:
                        try:
                            from sklearn.model_selection import cross_val_score
                            cv_scores = cross_val_score(
                                best_model, X_train, y_train, 
                                cv=min(5, len(X_train)//2),  # Dynamic CV folds
                                scoring='r2'
                            )
                            st.session_state.cv_scores = cv_scores
                        except Exception as e:
                            st.warning(f"Cross-validation failed: {str(e)}")
                    
                    # Make predictions
                    train_pred = best_model.predict(X_train)
                    test_pred = best_model.predict(X_test)
                    
                    # Calculate metrics
                    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                    train_r2 = r2_score(y_train, train_pred)
                    test_r2 = r2_score(y_test, test_pred)
                    mae = mean_absolute_error(y_test, test_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                    
                    # Store metrics in session state
                    st.session_state.train_r2 = train_r2
                    st.session_state.test_r2 = test_r2
                    st.session_state.mae = mae
                    st.session_state.rmse = rmse
                    st.session_state.train_pred = train_pred
                    st.session_state.test_pred = test_pred
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                except Exception as e:
                    st.error(f"Advanced prediction failed: {str(e)}")
                    st.error("Please ensure: \n"
                             "1. Selected columns contain valid numerical data\n"
                             "2. There are no missing values in key columns\n"
                             "3. You have sufficient data (min 10 samples)\n"
                             "4. Feature and target columns have valid relationships")
                    return
        
        # Display results from session state if available
        if 'trained_model' in st.session_state:
            best_model = st.session_state.trained_model
            feature_cols = st.session_state.feature_cols
            X_train = st.session_state.X_train
            
            # Display results
            st.subheader("📊 Model Performance Analysis")
            col1, col2, col3 = st.columns(3)
            col1.metric("R² (Training)", f"{st.session_state.train_r2:.4f}", 
                       "Excellent" if st.session_state.train_r2 > 0.9 else 
                       "Good" if st.session_state.train_r2 > 0.8 else "Fair")
            col2.metric("R² (Validation)", f"{st.session_state.test_r2:.4f}", 
                       "Excellent" if st.session_state.test_r2 > 0.85 else 
                       "Good" if st.session_state.test_r2 > 0.7 else "Fair")
            col3.metric("MAE", f"{st.session_state.mae:.4f}", 
                       "Low" if st.session_state.mae < 0.1 * st.session_state.y_test.mean() else "Moderate")
            
            st.info(f"RMSE: {st.session_state.rmse:.4f} ({st.session_state.rmse/st.session_state.y_test.mean()*100:.1f}% of target mean)")
            
            if 'cv_scores' in st.session_state:
                st.info(f"Cross-Validation R²: {np.mean(st.session_state.cv_scores):.4f} ± {np.std(st.session_state.cv_scores):.4f}")
            
            # Feature importance
            st.subheader("🔍 Feature Importance Analysis")
            try:
                if hasattr(best_model, 'feature_importances_'):
                    importances = best_model.feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    fig_imp = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Feature Importance Scores',
                        color='Importance',
                        color_continuous_scale='Bluered'
                    )
                    fig_imp.update_layout(height=400)
                    st.plotly_chart(fig_imp, use_container_width=True)
                    
                elif hasattr(best_model, 'coef_'):
                    coefficients = best_model.coef_
                    if isinstance(coefficients, np.ndarray) and coefficients.ndim > 1:
                        coefficients = coefficients[0]
                    coef_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Coefficient': coefficients
                    }).sort_values('Coefficient', key=abs, ascending=False)
                    
                    fig_coef = px.bar(
                        coef_df,
                        x='Coefficient',
                        y='Feature',
                        orientation='h',
                        title='Feature Coefficients',
                        color='Coefficient',
                        color_continuous_scale='balance'
                    )
                    fig_coef.update_layout(height=400)
                    st.plotly_chart(fig_coef, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model")
            except Exception as e:
                st.warning(f"Feature importance visualization failed: {str(e)}")
            
            # Actual vs Predicted plot
            st.subheader("📈 Actual vs Predicted Values")
            fig = go.Figure()
            
            # Training data
            fig.add_trace(go.Scatter(
                x=st.session_state.y_train,
                y=st.session_state.train_pred,
                mode='markers',
                name='Training Data',
                marker=dict(color='#636EFA', size=8, opacity=0.7),
                hovertext=[f"Actual: {a:.2f}<br>Predicted: {p:.2f}" 
                          for a, p in zip(st.session_state.y_train, st.session_state.train_pred)]
            ))
            
            # Test data
            fig.add_trace(go.Scatter(
                x=st.session_state.y_test,
                y=st.session_state.test_pred,
                mode='markers',
                name='Validation Data',
                marker=dict(color='#EF553B', size=8, opacity=0.7),
                hovertext=[f"Actual: {a:.2f}<br>Predicted: {p:.2f}" 
                          for a, p in zip(st.session_state.y_test, st.session_state.test_pred)]
            ))
            
            # Perfect prediction line
            max_val = max(st.session_state.y_test.max(), st.session_state.test_pred.max())
            min_val = min(st.session_state.y_test.min(), st.session_state.test_pred.min())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='#00CC96', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title='Model Predictive Accuracy',
                xaxis_title='Actual Values',
                yaxis_title='Predicted Values',
                showlegend=True,
                template='plotly_white',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Partial Dependence Plots
            if len(feature_cols) > 1:
                st.subheader("🌐 Partial Dependence Analysis")
                selected_pdp_feature = st.selectbox(
                    "Select feature for partial dependence analysis",
                    feature_cols,
                    key="pdp_feature_select"  # Add key to preserve selection
                )
                
                try:
                    from sklearn.inspection import PartialDependenceDisplay
                    pdp_fig, ax = plt.subplots(figsize=(10, 6))
                    PartialDependenceDisplay.from_estimator(
                        best_model, X_train, [selected_pdp_feature], 
                        ax=ax, line_kw={"color": "red"}
                    )
                    ax.set_title(f'Partial Dependence Plot for {selected_pdp_feature}')
                    ax.set_ylabel('Partial Dependence')
                    st.pyplot(pdp_fig)
                except Exception as e:
                    st.warning(f"Partial dependence plot failed: {str(e)}")
            
            # Prediction explorer
            st.subheader("🔮 Prediction Explorer")
            st.write("Enter feature values to get predictions:")
            
            input_data = {}
            cols = st.columns(len(feature_cols))
            for i, col_name in enumerate(feature_cols):
                with cols[i]:
                    col_mean = df[col_name].mean()
                    col_std = df[col_name].std()
                    input_data[col_name] = st.number_input(
                        f"{col_name}",
                        value=float(col_mean),
                        step=float(col_std/10) if col_std > 0 else 0.1,
                        key=f"pred_input_{col_name}"  # Add key to preserve input values
                    )
            
            input_df = pd.DataFrame([input_data])
            prediction = best_model.predict(input_df)[0]
            
            st.metric("Predicted Value", f"{prediction:.4f}", 
                     delta_color="off")
            
            # Model persistence
            st.subheader("💾 Model Persistence")
            import joblib
            import io
            model_buffer = io.BytesIO()
            joblib.dump(best_model, model_buffer)
            model_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Trained Model",
                data=model_buffer,
                file_name="advanced_prediction_model.joblib",
                mime="application/octet-stream"
            )
            
            st.success("Advanced predictive modeling completed!")
            st.balloons()
                
# Enhanced AutoML Training Section with new features
def automl_training(df):
    st.subheader("🤖 AutoML Training")

    # Feature selection options
    st.subheader("Feature Selection")
    feature_options = st.radio("Select Features",
                               ["Use All Columns", "Select Specific Columns"],
                               key="feature_options_radio")
    # Target selection with unique key
    target_col = st.selectbox(
        "Select Target Column",
        [f"{col} ({df[col].dtype})" for col in df.columns],
        key="automl_target_select"
    )
    target_col = target_col.split(" (")[0]

    # Problem type detection
    problem_type = "Classification" if (
            df[target_col].dtype == 'object' or df[target_col].nunique() < 20) else "Regression"
    st.info(f"Detected Problem Type: {problem_type}")

    if feature_options == "Select Specific Columns":
        available_features = [col for col in df.columns if col != target_col]
        selected_features = st.multiselect("Choose Features", available_features,
                                           default=available_features,
                                           key="features_multiselect")
        X = df[selected_features]
    else:
        X = df.drop(columns=[target_col])

    y = df[target_col]

    # AutoML options - Add verbose_level definition here
    st.subheader("AutoML Configuration")
    verbose_level = st.selectbox("Verbosity Level", [0, 1, 2, 3], index=0,
                                 help="0 = silent, 1 = warnings, 2 = info, 3 = debug")
    time_limit = st.slider("Time Limit (seconds)", 30, 600, 120, key="time_limit_slider")

    # Handle class imbalance for classification problems
    handle_imbalanced = False
    if problem_type == "Classification":
        handle_imbalanced = st.checkbox("Handle Class Imbalance", True, key="handle_imbalance_automl")

    # Metric selection - now after problem_type is defined
    st.subheader("Optimization Metric")
    if problem_type == "Classification":
        automl_metric = st.selectbox("Select Metric",
                                     ["accuracy", "f1", "roc_auc", "precision", "recall"],
                                     index=0,
                                     key="automl_metric_class_select")
    else:
        automl_metric = st.selectbox("Select Metric",
                                     ["r2", "mse", "mae", "rmse"],
                                     index=0,
                                     key="automl_metric_reg_select")

    if st.button("🚀 Run AutoML"):
        with st.spinner(f"Running AutoML with time limit: {time_limit} seconds..."):
            try:
                # Sanitize feature names for FLAML compatibility
                X = sanitize_column_names(X)

                # Preprocess data - ensure consistent numerical types first
                for col in X.select_dtypes(include=['number']).columns:
                    if X[col].dtype.kind in 'iuf':
                        # Convert all numeric to float32 for consistency
                        X[col] = X[col].astype(np.float32)

                # Then preprocess categorical features
                cat_cols = X.select_dtypes(include=['object', 'category']).columns
                if len(cat_cols) > 0:
                    for col in cat_cols:
                        X[col] = LabelEncoder().fit_transform(X[col]).astype(np.float32)

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Run AutoML using FLAML
                from flaml import AutoML
                automl = AutoML()

                settings = {
                    "time_budget": time_limit,
                    "metric": automl_metric,
                    "task": problem_type.lower(),
                    "verbose": verbose_level,
                    "log_file_name": None,  # Disable logging to file
                }

                if problem_type == "Classification" and handle_imbalanced:
                    settings["eval_method"] = "cv"
                    settings["split_ratio"] = 0.2

                automl.fit(X_train, y_train, **settings)

                # Store model in session state
                st.session_state.automl_model = automl
                st.session_state.X_train_auto = X_train
                st.session_state.y_train_auto = y_train
                st.session_state.X_test_auto = X_test
                st.session_state.y_test_auto = y_test
                st.session_state.target_col_auto = target_col
                st.session_state.problem_type_auto = problem_type

                # Get best model and metrics
                best_model = automl.model
                model_name = type(best_model).__name__

                # Make predictions
                y_pred = automl.predict(X_test)

                # Calculate metrics
                st.success("AutoML completed successfully!")
                st.subheader(f"Best Model: {model_name}")

                if problem_type == "Classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    roc_auc = roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) == 2 else None
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                        if roc_auc is not None:
                            st.metric("ROC AUC", f"{roc_auc:.4f}")
                    with col2:
                        st.metric("F1 Score", f"{f1:.4f}")
                        st.metric("Precision", f"{precision:.4f}")
                        st.metric("Recall", f"{recall:.4f}")
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mse)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("R² Score", f"{r2:.4f}")
                        st.metric("Mean Absolute Error", f"{mae:.4f}")
                    with col2:
                        st.metric("Mean Squared Error", f"{mse:.4f}")
                        st.metric("Root Mean Squared Error", f"{rmse:.4f}")

                # Feature importance
                st.subheader("Feature Importance")
                try:
                    if hasattr(best_model, 'feature_importances_'):
                        importances = best_model.feature_importances_
                        indices = np.argsort(importances)[::-1]

                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=importances[indices][:10],
                            y=X.columns[indices][:10],
                            orientation='h'
                        ))

                        fig.update_layout(
                            title="Top 10 Most Important Features",
                            yaxis_title="Features",
                            xaxis_title="Importance Score",
                            height=500
                        )

                        st.plotly_chart(fig)

                        # SHAP explainability
                        if st.checkbox("Show SHAP Explainability", True):
                            try:
                                # Ensure data is properly encoded for SHAP
                                X_test_encoded = X_test.copy()
                                for col in X_test_encoded.select_dtypes(include=['object', 'category']):
                                    X_test_encoded[col] = LabelEncoder().fit_transform(X_test_encoded[col])

                                plot_shap_summary(best_model, X_test_encoded.values, X_test_encoded.columns.tolist())
                            except Exception as e:
                                st.warning(f"SHAP explanation not available: {str(e)}")
                except Exception as e:
                    st.warning(f"Could not generate feature importance: {str(e)}")
                
                # Model download
                st.subheader("Model Export")
                model_bytes = BytesIO()
                joblib.dump(automl, model_bytes)
                st.download_button(
                    label="Download AutoML Model",
                    data=model_bytes.getvalue(),
                    file_name=f"automl_model_{problem_type}.pkl",
                    mime="application/octet-stream"
                )
                
                # Add to model history
                if 'model_history' not in st.session_state:
                    st.session_state.model_history = []
                
                metrics = {}
                if problem_type == "Classification":
                    metrics = {
                        "Accuracy": accuracy,
                        "F1": f1,
                        "ROC AUC": roc_auc,
                        "Precision": precision,
                        "Recall": recall
                    }
                else:
                    metrics = {
                        "R2": r2,
                        "MSE": mse,
                        "MAE": mae,
                        "RMSE": rmse
                    }
                
                st.session_state.model_history.append({
                    'timestamp': datetime.datetime.now(),
                    'model_type': f"AutoML ({model_name})",
                    'problem_type': problem_type,
                    'metrics': metrics,
                    'training_time': time_limit,
                    'features': X.columns.tolist(),
                    'target': target_col
                })
                
            except Exception as e:
                st.error(f"AutoML failed: {str(e)}")

# Model history viewer
def show_model_history():
    if 'model_history' in st.session_state and st.session_state.model_history:
        st.subheader("Model Training History")
        
        # Convert to DataFrame
        history_df = pd.DataFrame(st.session_state.model_history)
        
        # Convert timestamp to string for display
        history_df['timestamp'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Show table
        st.dataframe(history_df)
        
        # Plot performance over time
        if len(history_df) > 1:
            st.subheader("Performance Trend")
            
            # Get first metric name
            first_metric = next(iter(history_df['metrics'].iloc[0].keys()), None)
            
            if first_metric:
                # Extract metric values
                history_df[first_metric] = history_df['metrics'].apply(lambda x: x.get(first_metric, None))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history_df['timestamp'],
                    y=history_df[first_metric],
                    mode='lines+markers',
                    name=first_metric
                ))
                
                fig.update_layout(
                    title=f"{first_metric} Over Time",
                    xaxis_title="Timestamp",
                    yaxis_title=first_metric
                )
                
                st.plotly_chart(fig)
    else:
        st.info("No model training history available")

def render_ml_section():
    st.title("🧠 Advanced Machine Learning Studio")
    st.markdown("Transform your data into predictive insights with our powerful ML tools")
    
    # File uploader with drag and drop
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"], 
                                   help="Maximum file size: 200MB")
    
    if uploaded_file is not None:
        # Load data with memory optimization
        with st.spinner("Optimizing dataset..."):
            df = load_large_file(uploaded_file)
            
            if df is not None:
                # Store in session state
                st.session_state.ml_data = df
                st.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Display data preview
                st.subheader("Data Preview")
                try:
                    st.dataframe(df.head())
                except Exception as e:
                    st.warning(f"Could not display dataframe: {str(e)}")
                    st.write("First 5 rows as text:")
                    st.write(df.head().to_dict())
                
                # Create tabs for different sections
                tab_names = ["ML Training", "AutoML", "Future Prediction"]
                if 'model_history' in st.session_state and st.session_state.model_history:
                    tab_names.append("Model History")
                
                tabs = st.tabs(tab_names)
                
                with tabs[0]:
                    standard_ml_training(df)
                with tabs[1]:
                    automl_training(df)
                with tabs[2]:
                    prediction_section(df)
                if len(tabs) > 3:
                    with tabs[3]:
                        show_model_history()
    else:
        # Welcome screen with features
        st.info("Upload a dataset to get started with machine learning")
        
        # Feature showcase
        with st.expander("🌟 Key Features", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Standard ML")
                st.markdown("""
                - Smart model recommendations
                - Feature encoding suggestions
                - Cross-validation reports
                - SHAP/LIME explainability
                - Single prediction playground
                - Model export (joblib/ONNX)
                - Training history tracking
                """)
                
            with col2:
                st.subheader("AutoML")
                st.markdown("""
                - Automatic model selection
                - Advanced feature engineering
                - Training logs visualization
                - Feature importance analysis
                - Prediction explanations
                - Handles complex datasets
                - Performance metrics tracking
                """)
            
            with col3:
                st.subheader("Future Prediction")
                st.markdown("""
                - Time series forecasting
                - Trend analysis
                - Multi-period predictions
                - Prophet integration
                - Visual trend displays
                - Prediction exports
                - Multiple model support
                """)
        
        with st.expander("📚 User Guide", expanded=False):
            st.markdown("""
            **Standard ML Workflow:**
            1. Upload your dataset
            2. Select feature and target columns
            3. View model recommendations
            4. Choose evaluation metrics
            5. Train model with cross-validation
            6. Analyze feature importance
            7. Explain predictions with SHAP/LIME
            8. Download model and inference code
            
            **AutoML Workflow:**
            1. Upload your dataset
            2. Select target column
            3. Configure AutoML settings
            4. View training logs and best model
            5. Analyze feature importance
            6. Explain individual predictions
            7. Test predictions in playground
            8. Download complete model package
            
            **Future Prediction Workflow:**
            1. Select target and feature columns
            2. Choose prediction horizon
            3. Select forecasting model
            4. Generate future trends
            5. Analyze prediction results
            6. Export forecasts as CSV
            7. Compare with historical data
            """)
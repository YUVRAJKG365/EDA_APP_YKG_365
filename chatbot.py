# chatbot.py (Smart and Context-Aware AI Data Scientist)
import numpy as np
import datetime
import streamlit as st
import requests
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder

# ==== ADVANCED CONFIGURATION ====
GEMINI_API_KEY = "AIzaSyAOEk8MUDADA3tFkN7ylA3Pfe8QkDGNNYY"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
PROMPT_LIMIT = 25
COOLDOWN_HOURS = 24

def initialize_prompt_tracking():
    if "gemini_prompt_count" not in st.session_state:
        st.session_state.gemini_prompt_count = 0
    if "first_prompt_time" not in st.session_state:
        st.session_state.first_prompt_time = datetime.datetime.now()

def can_prompt():
    initialize_prompt_tracking()
    return st.session_state.gemini_prompt_count < PROMPT_LIMIT

def increment_prompt():
    st.session_state.gemini_prompt_count += 1
    if st.session_state.gemini_prompt_count == 1:
        st.session_state.first_prompt_time = datetime.datetime.now()

def get_reset_time():
    if "first_prompt_time" in st.session_state:
        return st.session_state.first_prompt_time + datetime.timedelta(hours=COOLDOWN_HOURS)
    return datetime.datetime.now() + datetime.timedelta(hours=COOLDOWN_HOURS)

def format_reset_time(reset_time):
    return reset_time.strftime("%Y-%m-%d at %H:%M %p")

def convert_df_for_analysis(df):
    """Convert DataFrame columns to appropriate numeric types"""
    df_clean = df.copy()
    for col in df_clean.columns:
        try:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
        except:
            pass
        
        if df_clean[col].dtype == 'object':
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='ignore')
            except:
                if df_clean[col].nunique() < 50:
                    le = LabelEncoder()
                    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    return df_clean

def is_data_related(query):
    """Determine if the query is data-related"""
    data_keywords = [
        'data', 'analy', 'statistic', 'model', 'predict', 'visualiz',
        'clean', 'process', 'machine learning', 'ai', 'dataset',
        'pandas', 'numpy', 'excel', 'csv', 'json', 'sql', 'table',
        'column', 'row', 'feature', 'target', 'regress', 'classif',
        'cluster', 'plot', 'chart', 'graph', 'correlat', 'missing',
        'outlier', 'normaliz', 'standardiz', 'etl', 'transform','analyze',
        'insight', 'trend', 'pattern', 'business intelligence', 'dashboard',
        'report', 'query', 'big data', 'data science', 'data engineer',
        'data mining', 'data wrangling', 'data preparation', 'data cleaning',
        'data visualization', 'data analysis', 'data modeling', 'data architecture',
        'data governance', 'data quality', 'data integration', 'data pipeline', 
        'data lake', 'data warehouse', 'data mart', 'data catalog', 'data lineage',
        'data profiling', 'data transformation', 'data extraction', 'data loading',
        'data aggregation', 'data summarization', 'data segmentation', 'data enrichment',
        'data classification', 'data regression', 'data clustering', 'data association',
        'data anomaly detection', 'data feature engineering', 'data dimensionality reduction',
        'data time series', 'data forecasting', 'data simulation', 'data optimization',
        'data validation', 'data verification', 'data compliance', 'data security',
        'data privacy', 'data ethics', 'data stewardship', 'data lifecycle', 'data governance',
        'data standards', 'data policies', 'data procedures', 'data documentation',
        'data extractioin', 'data transformation', 'data loading', 'data processing',
    ]
    query = query.lower()
    return any(keyword in query for keyword in data_keywords)

def chatbot_response(user_input, context=None):
    if not GEMINI_API_KEY or GEMINI_API_KEY.strip() == "AIzaSyAOEk8MUDADA3tFkN7ylA3Pfe8QkDGNNYY":
        return "🔑 Gemini API key not configured. Please set your API key in chatbot.py."

    if not can_prompt():
        reset_time = get_reset_time()
        return (f"⏳ You've reached the session limit of {PROMPT_LIMIT} prompts. "
                f"Chat will reset on {format_reset_time(reset_time)}. "
                "Please return after this time for more advanced analysis.")

    # Build context string
    context_str = ""
    if context:
        for k, v in context.items():
            if v is not None and str(v).strip() != "":
                context_str += f"\n[{k.upper()}]: {str(v)[:1000]}{'...' if len(str(v)) > 1000 else ''}"

    if "tracker" in st.session_state:
        tracker_log = st.session_state.tracker.get_context()
        context_str += f"\n\n[APP OPERATIONS LOG]:\n{tracker_log}"

    # Determine response style based on mode and query type
    expert_mode = st.session_state.get('expert_mode', False)
    data_related = is_data_related(user_input)
    
    if not data_related:
        # Casual conversation response - friendly, fun, younger style
        system_prompt = (
            "You are a fun, friendly AI buddy. Respond like a cheerful young person (18-25 age range). "
            "Use simple words, emojis occasionally, and keep it lighthearted. "
            "You can make appropriate jokes or use casual slang when suitable. "
            "If the conversation turns to data analysis, gently guide it back to your capabilities.\n\n"
            "Response Style Guide:\n"
            "- Use contractions (you're, don't, etc.)\n"
            "- Keep sentences short and punchy\n"
            "- Add personality and warmth\n"
            "- Use emojis sparingly (1-2 per response max)\n"
            "- If confused: 'Hmm, not sure about that one! 😅'\n\n"
            f"USER MESSAGE: {user_input}\n\n"
            "CASUAL RESPONSE:"
        )
    elif expert_mode:
        # Expert technical response - wise, experienced elder style
        system_prompt = (
            "You are a distinguished AI Data Scientist with decades of experience. "
            "Respond like a seasoned professor (60+ age) with deep expertise. "
            "Your tone should be:\n"
            "- Precise and authoritative yet approachable\n"
            "- Rich with wisdom and practical insights\n"
            "- Using sophisticated but clear language\n"
            "- Occasionally sharing 'from experience' anecdotes\n"
            "- Structuring complex ideas methodically\n\n"
            "Response Structure:\n"
            "1. Answer with Explanation in deeper and precisely along with examples\n"
            "2. Technical nuances (when needed)\n"
            "3. Valuable insights and Treands\n"
            "4. Real-world application\n"
            "5. Pro tips from experience\n\n"
            "Example Phrases:\n"
            "- 'In my years of practice...'\n"
            "- 'The essential consideration is...'\n"
            "- 'What we've found most effective...'\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"USER QUESTION: {user_input}\n\n"
            "EXPERT RESPONSE:"
        )
    else:
        # Normal data-related response - professional young adult style
        system_prompt = (
            "You are a knowledgeable Data Assistant (30-40 age range). "
            "Respond like a skilled professional explaining to a colleague:\n"
            "- Clear and patient\n"
            "- Practical and solution-oriented\n"
            "- Using analogies when helpful\n"
            "- Avoiding jargon but not oversimplifying\n\n"
            "Response Structure:\n"
            "1. Main and Direct in simplified way answer (1-2 sentences)\n"
            "2. Key points (bulleted if >3 items)\n"
            "3. Actionable next steps\n\n"
            "Example Phrases:\n"
            "- 'Here's how I'd approach this...'\n"
            "- 'The key things to consider are...'\n"
            "- 'For your situation, I recommend...'\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"USER QUESTION: {user_input}\n\n"
            "PROFESSIONAL RESPONSE:"
        )

    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    data = {
        "contents": [
            {"role": "user", "parts": [{"text": system_prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.3 if expert_mode else 0.7,
            "topK": 20,
            "topP": 0.95,
            "maxOutputTokens": 2048
        }
    }

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data, timeout=25)
        increment_prompt()
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return answer.strip() or "🤔 Please rephrase your question for more precise insights."
        else:
            return f"⚠️ API Error ({response.status_code}): {response.text[:300]}"
    except Exception as e:
        return f"🚨 Connection Error: {str(e)}"

def build_expert_context():
    context = {}
    
    # Dataset context
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        
        # Clean and convert dataframe
        df_clean = convert_df_for_analysis(df)
        
        context['dataset'] = {
            'shape': f"{df_clean.shape[0]} rows × {df_clean.shape[1]} columns",
            'columns': list(df_clean.columns),
            'dtypes': {col: str(dtype) for col, dtype in df_clean.dtypes.items()},
            'missing_values': df_clean.isnull().sum().sum()
        }
        
        # Add statistical summary only for numeric columns
        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            context['dataset']['statistical_summary'] = df_clean[numeric_cols].describe().to_dict()
            
            # Add correlation matrix only if we have 2+ numeric columns
            if len(numeric_cols) > 1:
                context['dataset']['correlation_matrix'] = df_clean[numeric_cols].corr().to_dict()
    
    # Text/PDF context
    if 'text_data' in st.session_state and st.session_state.text_data is not None:
        context['text_analysis'] = {
            'length': len(st.session_state.text_data),
            'topics': extract_topics(st.session_state.text_data),
            'sentiment': analyze_sentiment(st.session_state.text_data)
        }
    
    # Image context
    if 'image_data' in st.session_state and st.session_state.image_data is not None:
        img = st.session_state.image_data
        context['image_analysis'] = {
            'format': img.format,
            'size': img.size,
            'mode': img.mode
        }
    
    return context

def extract_topics(text):
    return ["Data Analysis", "Business Insights", "Trends"]

def analyze_sentiment(text):
    return "Neutral"

def display_usage_status():
    initialize_prompt_tracking()
    reset_time = get_reset_time()
    
    status_container = st.container()
    with status_container:
        cols = st.columns([0.6, 0.4])
        cols[0].progress(st.session_state.gemini_prompt_count / PROMPT_LIMIT)
        cols[1].markdown(f"**{st.session_state.gemini_prompt_count}/{PROMPT_LIMIT} prompts used**")
        
        if st.session_state.gemini_prompt_count >= PROMPT_LIMIT:
            st.warning(f"⏳ Next chat available: {format_reset_time(reset_time)}")
        else:
            st.info(f"🔄 Resets on: {format_reset_time(reset_time)}")

def chatbot_ui():
    """Advanced AI Data Scientist Chat Interface"""
    st.markdown("## 🧠 AI Data Scientist Assistant")
    st.markdown("""
    <div style="border-left: 4px solid #4e73df; padding-left: 1rem; margin-bottom: 1.5rem">
        Interact with your expert AI Data Scientist. Get insights, analysis recommendations, 
        and data-driven strategies tailored to your specific dataset.
    </div>
    """, unsafe_allow_html=True)
    
    # Expert mode toggle
    st.session_state.expert_mode = st.toggle(
        "🔬 Expert Mode", 
        value=st.session_state.get('expert_mode', False),
        help="Enable for detailed technical analysis and advanced explanations"
    )
    
    # Display usage status
    display_usage_status()
    
    # Build expert-level context
    context = build_expert_context()
    
    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display conversation history
    chat_container = st.container(height=400, border=True)
    with chat_container:
        for msg in st.session_state.chat_history:
            avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
    
    # Input and processing
    user_input = st.chat_input("Ask your AI Data Scientist...")
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with chat_container:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(user_input)
            
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🔍 Analyzing..." if not st.session_state.expert_mode else "🔍 Conducting deep analysis..."):
                    response = chatbot_response(user_input, context=context)
                st.markdown(response)
        
        # Add AI response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
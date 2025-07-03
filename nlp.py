import pandas as pd
import numpy as np
import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
from nltk.util import ngrams
from collections import Counter
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import spacy
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import umap
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import time
from datetime import datetime
import io
import spacy

from utils.data_loader import display_file_info, handle_file_upload
from utils.session_state_manager import get_session_manager

# Load English language model for NLP
nlp = spacy.load("en_core_web_sm")

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.warning("spaCy English model not found. Installing...")
    from spacy.cli import download

    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Initialize pretrained models
vader = SentimentIntensityAnalyzer()
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except:
    summarizer = None
    st.warning("Could not load BART summarizer. Using extractive summarization only.")


# Text preprocessing functions
def preprocess_text(text, lowercase=True, remove_punct=True, remove_stopwords=True,
                   lemmatize=True, stem=False, remove_html=True, remove_emojis=True):
    """
    Preprocess text with various options and better alphanumeric handling
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove HTML tags
    if remove_html:
        text = BeautifulSoup(text, "html.parser").get_text()

    # Remove emojis
    if remove_emojis:
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

    # Lowercase
    if lowercase:
        text = text.lower()

    # Remove punctuation (but keep alphanumeric and basic punctuation for context)
    if remove_punct:
        text = re.sub(r'[^\w\s-]', '', text)  # Keep word chars, spaces, and hyphens

    # Tokenize with better handling of alphanumeric tokens
    words = word_tokenize(text)
    words = [word for word in words if any(c.isalpha() for c in word)]  # Keep only tokens with letters

    # Remove stopwords
    if remove_stopwords:
        words = [word for word in words if word not in stop_words]

    # Lemmatize
    if lemmatize:
        words = [lemmatizer.lemmatize(word) for word in words]

    # Stem
    if stem:
        words = [stemmer.stem(word) for word in words]

    return ' '.join(words)

def calculate_text_stats(text_series):
    """
    Calculate various text statistics
    """
    stats = {}
    all_text = ' '.join(text_series.astype(str))

    # Word count
    words = word_tokenize(all_text)
    stats['word_count'] = len(words)

    # Sentence count
    sentences = sent_tokenize(all_text)
    stats['sentence_count'] = len(sentences)

    # Unique words
    unique_words = set(words)
    stats['unique_word_count'] = len(unique_words)
    stats['unique_word_ratio'] = len(unique_words) / len(words) if len(words) > 0 else 0

    # Average word length
    avg_word_length = sum(len(word) for word in words) / len(words) if len(words) > 0 else 0
    stats['avg_word_length'] = avg_word_length

    # Average sentence length
    avg_sentence_length = sum(len(sent.split()) for sent in sentences) / len(sentences) if len(sentences) > 0 else 0
    stats['avg_sentence_length'] = avg_sentence_length

    # Reading time (assuming 200 words per minute)
    stats['reading_time_minutes'] = len(words) / 200

    return stats


def get_top_ngrams(text_series, n=10, ngram_range=(2, 2)):
    """
    Get top n-grams from text with better error handling
    """
    try:
        # Pre-filter text to ensure we have valid data
        text_series = text_series.astype(str).dropna()
        text_series = text_series[text_series.str.contains('[a-zA-Z]')]  # Only keep text with letters
        
        if len(text_series) == 0:
            return []

        vectorizer = CountVectorizer(
            ngram_range=ngram_range, 
            stop_words='english',
            token_pattern=r'(?u)\b[\w-]+\b'  # Include alphanumeric and hyphenated words
        )
        
        X = vectorizer.fit_transform(text_series)
        words = vectorizer.get_feature_names_out()
        counts = X.sum(axis=0).A1
        
        # Filter out pure numeric ngrams
        valid_indices = [i for i, word in enumerate(words) if any(c.isalpha() for c in word)]
        words = words[valid_indices]
        counts = counts[valid_indices]
        
        if len(words) == 0:
            return []
            
        top_indices = counts.argsort()[::-1][:n]
        return [(words[i], counts[i]) for i in top_indices]
        
    except Exception as e:
        st.warning(f"Could not extract n-grams: {str(e)}")
        return []


def perform_advanced_sentiment_analysis(df, text_col, method='distilbert'):
    """
    Advanced sentiment analysis using pre-trained ML models with enhanced visualizations
    
    Parameters:
    - df: DataFrame containing text data
    - text_col: Column name containing text to analyze
    - method: Analysis method ('distilbert', 'roberta', or 'ensemble')
    
    Returns:
    - DataFrame with sentiment analysis results
    """
    df = df.copy()
    
    # Check if dataframe is empty
    if df.empty:
        st.warning("⚠️ The input dataframe is empty!")
        return pd.DataFrame()
    
    # Text preprocessing with spaCy
    def preprocess_text(text):
        text = str(text)
        # Remove URLs, mentions, and special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Lemmatization with spaCy
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
        return ' '.join(tokens)
    
    with st.spinner('🔍 Preprocessing text data...'):
        df['cleaned_text'] = df[text_col].apply(preprocess_text)
    
    # Initialize sentiment analysis pipeline based on selected method
    try:
        with st.spinner('⚙️ Loading sentiment analysis model...'):
            if method == 'distilbert':
                sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
                )
            elif method == 'roberta':
                sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="siebert/sentiment-roberta-large-english"
                )
            elif method == 'ensemble':
                sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
            else:
                raise ValueError("Invalid method. Choose 'distilbert', 'roberta', or 'ensemble'")
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return pd.DataFrame()
    
    # Perform sentiment analysis with batching for large datasets
    def analyze_sentiment_batch(texts, pipeline):
        results = pipeline(texts)
        return results
    
    batch_size = 32  # Adjust based on your memory constraints
    text_batches = [df['cleaned_text'].iloc[i:i+batch_size].tolist() 
                   for i in range(0, len(df), batch_size)]
    
    sentiment_results = []
    with st.spinner('🧠 Analyzing sentiment...'):
        for batch in text_batches:
            try:
                results = analyze_sentiment_batch(batch, sentiment_pipeline)
                sentiment_results.extend(results)
            except Exception as e:
                st.warning(f"⚠️ Error processing batch: {str(e)}")
                # Fallback to neutral for failed batches
                sentiment_results.extend([{'label': 'NEUTRAL', 'score': 0.5}] * len(batch))
    
    # Process results
    df['sentiment_label'] = [result['label'] for result in sentiment_results]
    df['sentiment_score'] = [result['score'] if result['label'].upper() in ['POSITIVE', 'LABEL_2'] 
                            else -result['score'] if result['label'].upper() in ['NEGATIVE', 'LABEL_0'] 
                            else 0 for result in sentiment_results]
    
    # Standardize labels across different models
    df['sentiment'] = df['sentiment_label'].apply(
        lambda x: 'positive' if x.upper() in ['POSITIVE', 'LABEL_2'] else
                 'negative' if x.upper() in ['NEGATIVE', 'LABEL_0'] else 'neutral')
    
    # Calculate confidence scores
    df['confidence'] = [abs(result['score']) for result in sentiment_results]
    
    st.success("✅ Sentiment analysis completed!")
    
    # Display sample results
    with st.expander("📊 View Detailed Sentiment Analysis Results", expanded=False):
        st.dataframe(
            df[[text_col, 'sentiment', 'sentiment_score', 'confidence']]
            .sort_values('confidence', ascending=False)
            .head(10)
            .style.background_gradient(cmap='RdYlGn', subset=['sentiment_score'])
            .format({'sentiment_score': '{:.2f}', 'confidence': '{:.2f}'})
        )
    
    # Sentiment Score Summary
    pos = (df["sentiment"] == "positive").sum()
    neg = (df["sentiment"] == "negative").sum()
    neu = (df["sentiment"] == "neutral").sum()
    total = len(df)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Summary", "📊 Distribution", "☁️ Word Clouds", "📌 Key Insights"])
    
    with tab1:
        st.markdown("### Sentiment Score Summary")
        
        # Create metrics columns
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Positive Sentiment", f"{pos} ({pos/total:.1%})", 
                     delta=f"{pos/total:.1%} confidence")
        with m2:
            st.metric("Neutral Sentiment", f"{neu} ({neu/total:.1%})", 
                     delta=f"{neu/total:.1%} confidence")
        with m3:
            st.metric("Negative Sentiment", f"{neg} ({neg/total:.1%})", 
                     delta=f"{neg/total:.1%} confidence")
        
        # Enhanced pie chart
        fig = px.pie(
            names=['Positive', 'Neutral', 'Negative'],
            values=[pos, neu, neg],
            title='<b>Sentiment Distribution</b>',
            color=['Positive', 'Neutral', 'Negative'],
            color_discrete_map={"Positive": "#2ecc71", "Neutral": "#3498db", "Negative": "#e74c3c"},
            hole=0.3,
            template='plotly_dark'
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            font=dict(size=14),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Sentiment Distribution Bar Chart
        sentiment_counts = df["sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
        fig = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            labels={'x': 'Sentiment', 'y': 'Count'},
            color=sentiment_counts.index,
            color_discrete_map={"positive": "#2ecc71", "neutral": "#3498db", "negative": "#e74c3c"},
            title="<b>Sentiment Distribution</b>",
            text=sentiment_counts.values,
            template='plotly_dark'
        )
        fig.update_layout(
            xaxis_title="Sentiment",
            yaxis_title="Count",
            font=dict(size=14)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Polarity distribution histogram
        fig = px.histogram(df, x='sentiment_score', nbins=30,
                           title='<b>Sentiment Score Distribution</b>',
                           labels={'sentiment_score': 'Sentiment Score'},
                           color_discrete_sequence=['#3498db'],
                           template='plotly_dark')
        fig.update_layout(
            xaxis_title="Sentiment Score",
            yaxis_title="Count",
            font=dict(size=14)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Generate word clouds for each sentiment
        st.markdown("### Word Clouds by Sentiment")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Positive Sentiment")
            positive_text = " ".join(df[df['sentiment'] == 'positive']['cleaned_text'])
            if positive_text:
                wordcloud = WordCloud(width=800, height=400, 
                                     background_color='white', 
                                     colormap='Greens').generate(positive_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.warning("No positive sentiment text found")
        
        with col2:
            st.markdown("#### Negative Sentiment")
            negative_text = " ".join(df[df['sentiment'] == 'negative']['cleaned_text'])
            if negative_text:
                wordcloud = WordCloud(width=800, height=400, 
                                     background_color='white', 
                                     colormap='Reds').generate(negative_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.warning("No negative sentiment text found")
    
    with tab4:
        # Key insights section
        st.markdown("### Key Insights")
        
        # Most positive and negative comments
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Most Positive Feedback")
            most_positive = df.nlargest(3, 'sentiment_score')
            for idx, row in most_positive.iterrows():
                st.info(f"**Score:** {row['sentiment_score']:.2f}\n\n{row[text_col][:200]}...")
        
        with col2:
            st.markdown("#### Most Negative Feedback")
            most_negative = df.nsmallest(3, 'sentiment_score')
            for idx, row in most_negative.iterrows():
                st.error(f"**Score:** {row['sentiment_score']:.2f}\n\n{row[text_col][:200]}...")
        
        # Most common words by sentiment
        st.markdown("#### Most Frequent Words")
        
        def get_top_words(text_series, n=10):
            all_words = ' '.join(text_series).split()
            return Counter(all_words).most_common(n)
        
        positive_words = get_top_words(df[df['sentiment'] == 'positive']['cleaned_text'])
        negative_words = get_top_words(df[df['sentiment'] == 'negative']['cleaned_text'])
        
        fig = px.bar(
            x=[word[0] for word in positive_words],
            y=[word[1] for word in positive_words],
            labels={'x': 'Word', 'y': 'Frequency'},
            title='<b>Top Positive Words</b>',
            color_discrete_sequence=['#2ecc71'],
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.bar(
            x=[word[0] for word in negative_words],
            y=[word[1] for word in negative_words],
            labels={'x': 'Word', 'y': 'Frequency'},
            title='<b>Top Negative Words</b>',
            color_discrete_sequence=['#e74c3c'],
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Overall conclusion with confidence
    if total == 0:
        conclusion = "No data to analyze."
    else:
        majority = df["sentiment"].mode()[0]
        majority_pct = (df["sentiment"] == majority).sum() / total
        
        confidence_level = ""
        if majority_pct > 0.7:
            confidence_level = " (High Confidence)"
        elif majority_pct > 0.5:
            confidence_level = " (Moderate Confidence)"
        else:
            confidence_level = " (Low Confidence - mixed sentiments)"
        
        avg_score = df['sentiment_score'].mean()
        sentiment_strength = ""
        if abs(avg_score) > 0.5:
            sentiment_strength = " strongly"
        elif abs(avg_score) > 0.2:
            sentiment_strength = ""
        else:
            sentiment_strength = " slightly"
        
        conclusion = (
            f"Overall Sentiment: **{majority.capitalize()}{sentiment_strength}**{confidence_level}\n\n"
            f"- Average sentiment score: {avg_score:.2f}\n"
            f"- Sentiment distribution: {pos/total:.1%} positive, {neu/total:.1%} neutral, {neg/total:.1%} negative"
        )
    
    st.success(conclusion)
    
    return df[[text_col, 'sentiment', 'sentiment_score', 'confidence']]

def perform_ner(text_series):
    """
    Perform Named Entity Recognition on text
    """
    ner_results = []
    for text in text_series.astype(str):
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        ner_results.append(entities)

    # Count entity types
    all_entities = [ent for sublist in ner_results for ent in sublist]
    if all_entities:
        entity_df = pd.DataFrame(all_entities, columns=['Entity', 'Type'])
        entity_counts = entity_df['Type'].value_counts().reset_index()
        entity_counts.columns = ['Entity Type', 'Count']

        # Display entity type distribution
        st.markdown("### Named Entity Distribution")
        fig = px.bar(entity_counts, x='Entity Type', y='Count',
                     title='Frequency of Entity Types',
                     color='Entity Type')
        st.plotly_chart(fig, use_container_width=True)

        # Display sample entities
        with st.expander("View Named Entities"):
            st.dataframe(entity_df.head(20))
    else:
        st.warning("No named entities found in the text.")

    return ner_results


def perform_topic_modeling(df, text_col, num_topics=5, method="LDA", enhanced=False):
    """
    Topic modeling with LDA and NMF
    """
    df = df.copy()
    text_data = df[text_col].astype(str).fillna("")

    if method not in ["LDA", "NMF"]:
        st.error("Unsupported topic modeling method. Please choose LDA or NMF.")
        return None

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(text_data)
    feature_names = vectorizer.get_feature_names_out()

    if method == "LDA":
        model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    elif method == "NMF":
        model = NMF(n_components=num_topics, random_state=42, init='nndsvd', max_iter=400)

    model.fit(dtm)
    topics = []

    st.markdown("### Top Words per Topic")
    for idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
        st.write(f"**Topic {idx + 1}:** {', '.join(top_words)}")
        topics.append(top_words)

    # WordCloud for each topic
    st.markdown("### WordClouds for Topics")
    fig, axes = plt.subplots(1, num_topics, figsize=(5 * num_topics, 4))
    if num_topics == 1:
        axes = [axes]
    for idx, topic in enumerate(model.components_):
        word_freq = {feature_names[i]: topic[i] for i in topic.argsort()[-20:]}
        wc = WordCloud(width=400, height=300, background_color='white',
                       colormap='rainbow').generate_from_frequencies(word_freq)
        axes[idx].imshow(wc, interpolation='bilinear')
        axes[idx].axis('off')
        axes[idx].set_title(f"Topic {idx + 1}")
    st.pyplot(fig)

    # Topic distribution per document
    if enhanced:
        topic_results = model.transform(dtm)
        df['Dominant_Topic'] = topic_results.argmax(axis=1)
        st.markdown("### Topic Distribution Across Documents")
        fig = px.histogram(df, x='Dominant_Topic', nbins=num_topics,
                           title='Document Count per Topic',
                           labels={'Dominant_Topic': 'Topic Number'})
        st.plotly_chart(fig, use_container_width=True)

        return topics, topic_results
    else:
        return topics

def perform_text_summarization(text_series, method='extractive', num_sentences=3):
    """
    Perform text summarization using extractive or abstractive methods
    """
    if method == 'extractive':
        # Using LexRank from sumy
        summaries = []
        for text in text_series:
            try:
                parser = PlaintextParser.from_string(text, Tokenizer("english"))
                summarizer_lex = LexRankSummarizer()
                summary = summarizer_lex(parser.document, num_sentences)
                summary_text = " ".join([str(sentence) for sentence in summary])
                summaries.append(summary_text)
            except:
                summaries.append("")
        return summaries

    elif method == 'abstractive' and summarizer is not None:
        # Using BART model
        summaries = []
        for text in text_series:
            if len(text) > 100:  # Minimum length for abstractive summarization
                try:
                    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                except:
                    summaries.append("")
            else:
                summaries.append(text)
        return summaries
    else:
        st.warning("Abstractive summarization not available. Using extractive method.")
        return perform_text_summarization(text_series, method='extractive', num_sentences=num_sentences)


def perform_text_clustering(df, text_col, num_clusters=5, method='tfidf'):
    """
    Perform text clustering and visualization
    """
    text_data = df[text_col].astype(str).fillna("")

    if method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(text_data)

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)

        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X.toarray())

    elif method == 'bert':
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(text_data.tolist())

            # Perform KMeans clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)

            # Reduce dimensionality for visualization
            umap_model = umap.UMAP(n_components=2, random_state=42)
            X_tsne = umap_model.fit_transform(embeddings)
        except ImportError:
            st.error("SentenceTransformers not installed. Using TF-IDF instead.")
            return perform_text_clustering(df, text_col, num_clusters=num_clusters, method='tfidf')

    # Create visualization
    viz_df = pd.DataFrame({
        'x': X_tsne[:, 0],
        'y': X_tsne[:, 1],
        'cluster': clusters,
        'text': text_data.str.wrap(50).str.replace('\n', '<br>')  # For hover text
    })

    fig = px.scatter(
        viz_df, x='x', y='y', color='cluster',
        hover_data=['text'], title='Text Clustering Visualization'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show cluster sizes
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    fig = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        labels={'x': 'Cluster', 'y': 'Count'},
        title='Cluster Sizes'
    )
    st.plotly_chart(fig, use_container_width=True)

    return clusters


def search_text(df, text_col, query, use_regex=False):
    """
    Search text with optional regex support
    """
    if use_regex:
        try:
            mask = df[text_col].astype(str).str.contains(query, regex=True)
            results = df[mask]
        except:
            st.error("Invalid regular expression pattern")
            return pd.DataFrame()
    else:
        mask = df[text_col].astype(str).str.contains(query, case=False)
        results = df[mask]

    st.info(f"Found {len(results)} matching documents")
    return results


def plot_wordcloud(df, text_col, max_words=200):
    """
    Enhanced word cloud visualization with error handling
    """
    try:
        # Check if column exists and has text data
        if text_col not in df.columns:
            st.warning(f"Column '{text_col}' not found in data")
            return
        
        # Filter out non-string and empty values
        text_data = df[text_col].astype(str).dropna()
        if len(text_data) == 0:
            st.warning("No text data available for word cloud")
            return

        # Combine all text
        text = " ".join(text_data)
        
        # Check if text contains any alphabetic characters
        if not any(c.isalpha() for c in text):
            st.warning("Text contains no alphabetic characters for word cloud")
            return

        # Generate word frequencies with better filtering
        words = word_tokenize(text)
        words = [word.lower() for word in words 
                if word.isalnum() and any(c.isalpha() for c in word) 
                and word.lower() not in stop_words]
        
        if not words:
            st.warning("No valid words found after processing")
            return

        word_freq = Counter(words)

        col1, col2 = st.columns(2)

        with col1:
            # Word cloud
            try:
                wc = WordCloud(width=800, height=400, background_color='white',
                            colormap='rainbow', stopwords=STOPWORDS, 
                            max_words=max_words).generate_from_frequencies(word_freq)
                plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud')
                st.pyplot(plt)
            except Exception as e:
                st.error(f"Could not generate word cloud: {str(e)}")

        with col2:
            # Top words bar chart
            try:
                top_words = word_freq.most_common(20)
                if top_words:
                    top_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                    fig = px.bar(
                        top_df,
                        x='Word',
                        y='Frequency',
                        labels={'x': 'Word', 'y': 'Frequency'},
                        title='Top 20 Words'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not create word frequency chart: {str(e)}")

    except Exception as e:
        st.error(f"Error in word cloud generation: {str(e)}")


def show_text_stats(df, text_col):
    """
    Display comprehensive text statistics
    """
    stats = calculate_text_stats(df[text_col])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Words", stats['word_count'])
        st.metric("Unique Words", stats['unique_word_count'])

    with col2:
        st.metric("Unique Word Ratio", f"{stats['unique_word_ratio']:.1%}")
        st.metric("Average Word Length", f"{stats['avg_word_length']:.1f}")

    with col3:
        st.metric("Sentences", stats['sentence_count'])
        st.metric("Reading Time", f"{stats['reading_time_minutes']:.1f} mins")

    # Show top unigrams, bigrams, trigrams
    st.markdown("### Most Common N-grams")

    tab1, tab2, tab3 = st.tabs(["Unigrams", "Bigrams", "Trigrams"])

    with tab1:
        top_unigrams = get_top_ngrams(df[text_col], n=10, ngram_range=(1, 1))
        if top_unigrams:  # Only plot if we have results
            unigram_df = pd.DataFrame(top_unigrams, columns=['Word', 'Frequency'])
            fig = px.bar(
                unigram_df,
                x='Word',
                y='Frequency',
                labels={'x': 'Word', 'y': 'Frequency'},
                title='Top 10 Unigrams'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No unigrams found")

    with tab2:
        top_bigrams = get_top_ngrams(df[text_col], n=10, ngram_range=(2, 2))
        if top_bigrams:  # Only plot if we have results
            bigram_df = pd.DataFrame(top_bigrams, columns=['Phrase', 'Frequency'])
            fig = px.bar(
                bigram_df,
                x='Phrase',
                y='Frequency',
                labels={'x': 'Phrase', 'y': 'Frequency'},
                title='Top 10 Bigrams'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No bigrams found")

    with tab3:
        top_trigrams = get_top_ngrams(df[text_col], n=10, ngram_range=(3, 3))
        if top_trigrams:  # Only plot if we have results
            trigram_df = pd.DataFrame(top_trigrams, columns=['Phrase', 'Frequency'])
            fig = px.bar(
                trigram_df,
                x='Phrase',
                y='Frequency',
                labels={'x': 'Phrase', 'y': 'Frequency'},
                title='Top 10 Trigrams'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No trigrams found")


def render_nlp_section():
    tracker = st.session_state.tracker  # Ensure tracker is available
    session_manager = get_session_manager()  # Get session manager instance
    section = "NLP"  # Define section name

    tracker.log_section(section)
    st.markdown("## 🔤 Advanced Natural Language Processing")
    st.markdown("Perform comprehensive NLP analysis on your text data")

    # --- Unified support for DataFrame, Text, and PDF ---
    df_nlp = None
    text_cols = []
    
    # --- Single file uploader for NLP analysis ---
    uploaded_filenlp = handle_file_upload(
        section=section,
        file_types=['csv', 'txt', 'pdf', 'xlsx', 'xls'],
        title="Upload a file for NLP analysis",
        help_text="Supports CSV, Excel, Text, and PDF files"
    )

    # Get data from session manager
    df_nlp = session_manager.get_dataframe(section)
    text_data = session_manager.get_text_data(section)
    pdf_data = session_manager.get_data(section, 'pdf_data')

    # 1. Structured data (CSV/Excel)
    if df_nlp is not None and hasattr(df_nlp, "select_dtypes"):
        text_cols = df_nlp.select_dtypes(include=['object', 'string']).columns.tolist()

    # 2. Plain text file
    elif text_data is not None:
        # Convert text to DataFrame (one row per line)
        lines = [line for line in text_data.split('\n') if line.strip()]
        df_nlp = pd.DataFrame({'text': lines})
        text_cols = ['text']

    # 3. PDF file
    elif pdf_data is not None and pdf_data.get('text'):
        lines = [line for line in pdf_data['text'].split('\n') if line.strip()]
        df_nlp = pd.DataFrame({'text': lines})
        text_cols = ['text']

    # Display file info if processed
    if session_manager.get_data(section, 'file_processed', False):
        display_file_info(section)

    if df_nlp is not None and len(text_cols) > 0:
        text_col = st.selectbox("Select Text Column:", text_cols, key="nlp_text_col")

        # Preprocessing options
        with st.expander("🛠️ Text Preprocessing Options"):
            tracker.log_tab("Text Preprocessing Options")
            col1, col2 = st.columns(2)

            with col1:
                lowercase = st.checkbox("Lowercase", True)
                remove_punct = st.checkbox("Remove Punctuation", True)
                remove_stopwords = st.checkbox("Remove Stopwords", True)
                remove_html = st.checkbox("Remove HTML Tags", True)

            with col2:
                lemmatize = st.checkbox("Lemmatization", True)
                stem = st.checkbox("Stemming", False)
                remove_emojis = st.checkbox("Remove Emojis", True)

            if st.button("Apply Preprocessing"):
                with st.spinner("Preprocessing text..."):
                    df_nlp['processed_text'] = df_nlp[text_col].apply(
                        lambda x: preprocess_text(
                            x, lowercase, remove_punct, remove_stopwords,
                            lemmatize, stem, remove_html, remove_emojis
                        )
                    )
                    text_col = 'processed_text'
                    st.success("Text preprocessing completed!")
                    st.dataframe(df_nlp[[text_col]].head())
                    tracker.log_operation("Applied text preprocessing")

        # Main NLP features
        st.markdown("### NLP Features")

        # Radio button navigation
        nlp_option = st.radio(
            "Choose NLP Feature:",
            options=["📊 Stats", "😊 Sentiment", "🔍 NER", "🗂️ Topics", "📝 Summarize", "🔎 Search"],
            horizontal=True,
            label_visibility="collapsed"
        )
        tracker.log_tab(nlp_option)

        # Show the selected option
        if nlp_option == "📊 Stats":  # Text Statistics
            tracker.log_operation("Viewed Text Statistics")
            show_text_stats(df_nlp, text_col)
            plot_wordcloud(df_nlp, text_col)

        elif nlp_option == "😊 Sentiment":  # Sentiment Analysis
            tracker.log_operation("Viewed Sentiment Analysis")
            # Method selection with validation
            method_options = {
                "DistilBERT (Fast)": "distilbert",
                "RoBERTa (Accurate)": "roberta", 
                "Ensemble (Most Reliable)": "ensemble"
            }

            selected_method = st.radio(
                "🛠️ Select Sentiment Analysis Model:",
                options=list(method_options.keys()),
                horizontal=True,
                help="Choose the NLP model for analysis. Ensemble combines multiple models for best results."
            )

            if st.button("🚀 Run Sentiment Analysis", key="run_sentiment_analysis"):
                with st.spinner("🔍 Analyzing sentiment with advanced NLP model..."):
                    try:
                        # Get the method key from the display options
                        method = method_options[selected_method]

                        # Clear previous results if any
                        if 'df_sentiment' in st.session_state:
                            del st.session_state.df_sentiment

                        # Create progress bar with stages
                        progress_text = st.empty()
                        progress_bar = st.progress(0)

                        # Step 1: Preprocessing
                        progress_text.text("🔄 Preprocessing text data...")
                        progress_bar.progress(20)

                        # Step 2: Loading model
                        progress_text.text("⚙️ Loading NLP model (this may take a minute)...")
                        progress_bar.progress(40)

                        # Run analysis
                        st.session_state.df_sentiment = perform_advanced_sentiment_analysis(
                            df_nlp,
                            text_col,
                            method=method
                        )

                        # Step 3: Generating visualizations
                        progress_text.text("📊 Creating interactive visualizations...")
                        progress_bar.progress(80)

                        # Complete
                        progress_text.text("✅ Analysis complete!")
                        progress_bar.progress(100)

                        # Success message
                        st.balloons()
                        st.success(f"🎉 {selected_method} analysis completed successfully!")

                        # Download button for results
                        csv = st.session_state.df_sentiment.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Sentiment Results",
                            data=csv,
                            file_name=f'sentiment_results_{method}.csv',
                            mime='text/csv',
                            key='download_sentiment'
                        )

                    except Exception as e:
                        st.error(f"❌ Error in sentiment analysis: {str(e)}")
                        st.error("Please check your input data and try again.")
                        if 'df_sentiment' in st.session_state:
                            del st.session_state.df_sentiment
                    finally:
                        # Ensure progress bar is complete
                        if 'progress_bar' in locals():
                            progress_bar.progress(100)

        elif nlp_option == "🔍 NER":  # Named Entity Recognition
            tracker.log_operation("Viewed Named Entity Recognition")
            if st.button("Extract Named Entities"):
                with st.spinner("Identifying named entities..."):
                    ner_results = perform_ner(df_nlp[text_col])

                    # Show sample text with highlighted entities
                    sample_text = df_nlp[text_col].iloc[0]
                    doc = nlp(sample_text)

                    html = ""
                    for token in doc:
                        if token.ent_type_:
                            html += f"<mark style='background-color: #ffdd57; padding: 2px; margin: 2px; border-radius: 3px;' title='{token.ent_type_}'>{token.text}</mark>"
                        else:
                            html += token.text + " "

                    st.markdown("### Sample Text with Named Entities")
                    st.markdown(html, unsafe_allow_html=True)

        elif nlp_option == "🗂️ Topics":  # Topic Modeling
            tracker.log_operation("Viewed Topic Modeling")
            col1, col2 = st.columns(2)

            with col1:
                method = st.selectbox("Topic Modeling Method",
                                     ["LDA", "NMF"],
                                     help="LDA: Traditional probabilistic topic modeling\nNMF: Matrix factorization approach")
                num_topics = st.slider("Number of Topics", 2, 10, 5)

            with col2:
                enhanced = st.checkbox("Enhanced Visualization", True)
                show_doc_topics = st.checkbox("Show Document-Topic Distribution", False)

            if st.button("Run Topic Modeling"):
                with st.spinner("Identifying topics..."):
                    try:
                        topics, topic_results = perform_topic_modeling(
                            df_nlp, text_col, num_topics, method, enhanced)

                        if show_doc_topics:
                            st.markdown("### Document-Topic Distribution")
                            topic_df = pd.DataFrame(topic_results,
                                                   columns=[f"Topic {i + 1}" for i in range(num_topics)])
                            st.dataframe(topic_df.head(10))
                    except Exception as e:
                        st.error(f"Error in topic modeling: {str(e)}")

        elif nlp_option == "📝 Summarize":  # Text Summarization
            tracker.log_operation("Viewed Text Summarization")
            col1, col2 = st.columns(2)

            with col1:
                method = st.selectbox("Summarization Method",
                                     ["extractive", "abstractive"],
                                     help="Extractive: Selects important sentences\nAbstractive: Generates new summary text")
                num_sentences = st.slider("Summary Length (sentences)", 1, 10, 3)

            if st.button("Generate Summaries"):
                with st.spinner("Summarizing text..."):
                    try:
                        df_nlp['summary'] = perform_text_summarization(
                            df_nlp[text_col], method, num_sentences)

                        st.success("Summarization completed!")
                        st.dataframe(df_nlp[[text_col, 'summary']].head())
                    except Exception as e:
                        st.error(f"Error in summarization: {str(e)}")

        elif nlp_option == "🔎 Search":  # Text Search
            tracker.log_operation("Viewed Text Search")
            query = st.text_input("Enter search query")
            use_regex = st.checkbox("Use Regular Expression", False)

            if query:
                results = search_text(df_nlp, text_col, query, use_regex)
                if not results.empty:
                    st.dataframe(results.head(20))

        else:
            st.info("ℹ️ Please upload a CSV, Excel, Text, or PDF file with text data for NLP analysis.")
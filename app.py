"""
Streamlit Web Application
Interactive UI for Sentiment Analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import sys


# Import custom modules
from data_ingestion import load_imdb_data
from preprocessing import TextPreprocessor
from feature_extraction import FeatureExtractor
#from model_training import SentimentModel, compare_models
from main import SentimentAnalysisPipeline
from sklearn.model_selection import train_test_split


# Page configuration
st.set_page_config(
    page_title="NLP Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
    }
    .positive {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        color: #155724;
    }
    .negative {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'is_trained' not in st.session_state:
        st.session_state.is_trained = False
    if 'training_data' not in st.session_state:
        st.session_state.training_data = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None


def sidebar_config():
    """Sidebar configuration"""
    st.sidebar.title("⚙️ Configuration")
    
    st.sidebar.markdown("### Pipeline Settings")
    
    # Preprocessing settings
    use_lemmatization = st.sidebar.checkbox("Use Lemmatization", value=True)
    
    # Feature extraction settings
    feature_method = st.sidebar.selectbox(
        "Feature Extraction Method",
        ["tfidf", "bow"],
        format_func=lambda x: "TF-IDF" if x == "tfidf" else "Bag of Words"
    )
    
    max_features = st.sidebar.slider(
        "Maximum Features",
        min_value=1000,
        max_value=10000,
        value=5000,
        step=1000
    )
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["logistic_regression", "naive_bayes", "svm", "random_forest"],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Training settings
    st.sidebar.markdown("### Training Settings")
    sample_size = st.sidebar.slider(
        "Training Sample Size",
        min_value=500,
        max_value=10000,
        value=2000,
        step=500
    )
    
    return {
        'use_lemmatization': use_lemmatization,
        'feature_method': feature_method,
        'max_features': max_features,
        'model_type': model_type,
        'sample_size': sample_size
    }


def main_page():
    """Main page content"""
    st.markdown('<h1 class="main-header">🎬 Movie Review Sentiment Analysis</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Learn NLP from Data Ingestion to Deployment")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📚 Overview",
        "📥 Data Exploration", 
        "🔧 Preprocessing",
        "🤖 Model Training",
        "🎯 Predictions",
        "📊 Analysis"
    ])
    
    # Tab 1: Overview
    with tab1:
        show_overview()
    
    # Tab 2: Data Exploration
    with tab2:
        show_data_exploration()
    
    # Tab 3: Preprocessing
    with tab3:
        show_preprocessing()
    
    # Tab 4: Model Training
    with tab4:
        show_model_training()
    
    # Tab 5: Predictions
    with tab5:
        show_predictions()
    
    # Tab 6: Analysis
    with tab6:
        show_analysis()


def show_overview():
    """Overview tab content"""
    st.markdown("## 📖 About This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Learning Objectives
        - Understand complete NLP pipeline
        - Learn text preprocessing techniques
        - Master feature extraction (BoW, TF-IDF)
        - Train and evaluate ML models
        - Deploy sentiment analysis system
        
        ### 📊 Dataset
        - **Source**: IMDB Movie Reviews
        - **Size**: 50,000 reviews
        - **Classes**: Positive / Negative
        - **Balance**: 50-50 split
        """)
    
    with col2:
        st.markdown("""
        ### 🔄 Pipeline Steps
        1. **Data Ingestion** - Load IMDB dataset
        2. **Preprocessing** - Clean and normalize text
        3. **Feature Extraction** - Convert text to numbers
        4. **Model Training** - Train ML classifier
        5. **Evaluation** - Measure performance
        6. **Deployment** - Interactive predictions
        
        ### 🛠️ Technologies
        - Python, NLTK, Scikit-learn
        - Streamlit for UI
        - Multiple ML algorithms
        """)
    
    st.markdown("---")
    st.info("👈 Use the sidebar to configure your pipeline and get started!")


def show_data_exploration():
    """Data exploration tab"""
    st.markdown("## 📥 Data Exploration")
    
    if st.button("🔄 Load Sample Data"):
        with st.spinner("Loading IMDB dataset..."):
            train_df, test_df = load_imdb_data(sample_size=1000)
            st.session_state.training_data = pd.concat([train_df, test_df], ignore_index=True)
        
        st.success("✅ Data loaded successfully!")
    
    if st.session_state.training_data is not None:
        df = st.session_state.training_data
        
        # Dataset statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", len(df))
        with col2:
            st.metric("Positive", len(df[df['sentiment'] == 'positive']))
        with col3:
            st.metric("Negative", len(df[df['sentiment'] == 'negative']))
        with col4:
            avg_length = df['review'].str.len().mean()
            st.metric("Avg Length", f"{avg_length:.0f} chars")
        
        # Sentiment distribution
        st.markdown("### 📊 Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Balance",
            color=sentiment_counts.index,
            color_discrete_map={'positive': '#28a745', 'negative': '#dc3545'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample reviews
        st.markdown("### 📝 Sample Reviews")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Positive Review")
            pos_sample = df[df['sentiment'] == 'positive'].iloc[0]['review']
            st.success(pos_sample[:300] + "...")
        
        with col2:
            st.markdown("#### ❌ Negative Review")
            neg_sample = df[df['sentiment'] == 'negative'].iloc[0]['review']
            st.error(neg_sample[:300] + "...")
        
        # Show dataframe
        st.markdown("### 📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)


def show_preprocessing():
    """Preprocessing demonstration tab"""
    st.markdown("## 🔧 Text Preprocessing")
    
    st.markdown("""
    ### Why Preprocessing?
    Raw text needs to be cleaned and standardized before machine learning models can use it.
    """)
    
    # Sample text input
    sample_text = st.text_area(
        "Enter sample text to preprocess:",
        value="This movie was AMAZING! I absolutely loved it. <br/> Best film ever! 10/10",
        height=100
    )
    
    # Preprocessing options
    col1, col2 = st.columns(2)
    
    with col1:
        use_stemming = st.checkbox("Use Stemming", value=False)
    with col2:
        use_lemmatization = st.checkbox("Use Lemmatization", value=True)
    
    if st.button("🔄 Apply Preprocessing"):
        preprocessor = TextPreprocessor(
            use_stemming=use_stemming,
            use_lemmatization=use_lemmatization
        )
        
        # Show step-by-step preprocessing
        st.markdown("### 📋 Preprocessing Steps")
        
        steps = []
        
        # Original
        steps.append(("Original", sample_text))
        
        # Lowercase
        text = preprocessor.step1_lowercase(sample_text)
        steps.append(("1. Lowercase", text))
        
        # Remove HTML
        text = preprocessor.step2_remove_html(text)
        steps.append(("2. Remove HTML", text))
        
        # Remove punctuation
        text = preprocessor.step4_remove_punctuation(text)
        steps.append(("3. Remove Punctuation", text))
        
        # Remove numbers
        text = preprocessor.step5_remove_numbers(text)
        steps.append(("4. Remove Numbers", text))
        
        # Tokenization
        tokens = preprocessor.step7_tokenization(text)
        steps.append(("5. Tokenization", str(tokens[:20])))
        
        # Remove stopwords
        tokens = preprocessor.step8_remove_stopwords(tokens)
        steps.append(("6. Remove Stopwords", str(tokens[:20])))
        
        # Lemmatization
        if use_lemmatization:
            tokens = preprocessor.step10_lemmatization(tokens)
            steps.append(("7. Lemmatization", str(tokens[:20])))
        
        # Final
        final_text = ' '.join(tokens)
        steps.append(("Final Result", final_text))
        
        # Display steps
        for step_name, step_result in steps:
            with st.expander(step_name):
                st.code(step_result[:500])


def show_model_training():
    """Model training tab"""
    st.markdown("## 🤖 Model Training")
    
    config = sidebar_config()
    
    st.markdown(f"""
    ### Current Configuration:
    - **Preprocessing**: {'Lemmatization' if config['use_lemmatization'] else 'No normalization'}
    - **Features**: {config['feature_method'].upper()} with {config['max_features']} features
    - **Model**: {config['model_type'].replace('_', ' ').title()}
    - **Sample Size**: {config['sample_size']} reviews
    """)
    
    if st.button("🚀 Train Model"):
        with st.spinner("Training model... This may take a minute..."):
            # Create pipeline
            pipeline = SentimentAnalysisPipeline(
                use_lemmatization=config['use_lemmatization'],
                feature_method=config['feature_method'],
                max_features=config['max_features'],
                model_type=config['model_type']
            )
            
            # Train with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Loading data...")
            progress_bar.progress(20)
            
            # Load data
            train_df, test_df = load_imdb_data(sample_size=config['sample_size'])
            df = pd.concat([train_df, test_df], ignore_index=True)
            
            status_text.text("Preprocessing text...")
            progress_bar.progress(40)
            
            # Preprocess
            df = pipeline.preprocessor.preprocess_dataframe(df)
            
            status_text.text("Extracting features...")
            progress_bar.progress(60)
            
            # Feature extraction
            X = pipeline.feature_extractor.fit_transform(df['cleaned_text'].tolist())
            y = df['label'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            status_text.text("Training model...")
            progress_bar.progress(80)
            
            # Train
            pipeline.model.train(X_train, y_train)
            
            status_text.text("Evaluating model...")
            progress_bar.progress(90)
            
            # Evaluate
            metrics = pipeline.model.evaluate(X_test, y_test)
            
            progress_bar.progress(100)
            status_text.text("Training complete!")
            
            # Store in session state
            st.session_state.pipeline = pipeline
            st.session_state.is_trained = True
            st.session_state.metrics = metrics
            
            pipeline.is_trained = True
        
        st.success("✅ Model trained successfully!")
        
        # Display metrics
        st.markdown("### 📊 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
        
        # Metric explanations
        with st.expander("📖 Understanding the Metrics"):
            st.markdown("""
            - **Accuracy**: Percentage of correct predictions
            - **Precision**: Of predicted positive, how many are truly positive
            - **Recall**: Of actual positive, how many did we catch
            - **F1-Score**: Balance between precision and recall
            """)


def show_predictions():
    """Predictions tab"""
    st.markdown("## 🎯 Make Predictions")
    
    if not st.session_state.is_trained:
        st.warning("⚠️ Please train a model first (go to Model Training tab)")
        return
    
    pipeline = st.session_state.pipeline
    
    # Single prediction
    st.markdown("### 📝 Single Review Prediction")
    
    user_input = st.text_area(
        "Enter a movie review:",
        height=150,
        placeholder="Type your movie review here..."
    )
    
    if st.button("🔮 Predict Sentiments"):
        if user_input.strip():
            prediction, confidence = pipeline.predict_sentiment(user_input)
            
            sentiment = "POSITIVE 😊" if prediction == 1 else "NEGATIVE 😞"
            color = "positive" if prediction == 1 else "negative"
            
            st.markdown(f'<div class="{color}"><h2>{sentiment}</h2></div>', 
                       unsafe_allow_html=True)
            
            if confidence:
                st.progress(float(confidence))
                st.markdown(f"**Confidence**: {confidence:.2%}")
            
            # Show preprocessing
            with st.expander("🔍 See preprocessing steps"):
                cleaned = pipeline.preprocessor.preprocess_text(user_input)
                st.write("**Original:**", user_input)
                st.write("**Cleaned:**", cleaned)
        else:
            st.error("Please enter some text!")
    
    # Batch prediction
    st.markdown("---")
    st.markdown("### 📦 Batch Prediction")
    
    sample_reviews = st.text_area(
        "Enter multiple reviews (one per line):",
        height=200,
        placeholder="Review 1\nReview 2\nReview 3..."
    )
    
    if st.button("🔮 Predict All"):
        if sample_reviews.strip():
            reviews = [r.strip() for r in sample_reviews.split('\n') if r.strip()]
            
            predictions, probabilities = pipeline.predict_batch(reviews)
            
            results = []
            for i, (review, pred) in enumerate(zip(reviews, predictions)):
                sentiment = "Positive" if pred == 1 else "Negative"
                conf = probabilities[i][pred] if probabilities is not None else None
                
                results.append({
                    'Review': review[:50] + '...' if len(review) > 50 else review,
                    'Sentiment': sentiment,
                    'Confidence': f"{conf:.2%}" if conf else "N/A"
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)


def show_analysis():
    """Analysis tab"""
    st.markdown("## 📊 Model Analysis")
    
    if not st.session_state.is_trained:
        st.warning("⚠️ Please train a model first (go to Model Training tab)")
        return
    
    # Show metrics
    if st.session_state.metrics:
        st.markdown("### 📈 Performance Metrics")
        
        metrics = st.session_state.metrics
        
        # Bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                y=[metrics['accuracy'], metrics['precision'], 
                   metrics['recall'], metrics['f1_score']],
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            )
        ])
        fig.update_layout(
            title="Model Performance Metrics",
            yaxis_title="Score",
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (for applicable models)
    st.markdown("### 🔍 Feature Analysis")
    st.info("Feature importance visualization can be added here for tree-based models")


def main():
    """Main application"""
    initialize_session_state()
    main_page()


if __name__ == "__main__":
    main()
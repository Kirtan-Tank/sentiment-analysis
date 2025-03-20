import streamlit as st
from transformers import pipeline

@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    try:
        return pipeline(
            "sentiment-analysis",
            model="siebert/sentiment-roberta-large-english",
            truncation=True  # ensures input texts are appropriately truncated
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the pipeline once and cache it
sentiment_pipeline = load_sentiment_pipeline()

st.title("Sentiment Analysis with RoBERTa-Large")

# Get user input
user_input = st.text_area("Enter text for sentiment prediction:")

if st.button("Analyze"):
    if not sentiment_pipeline:
        st.error("The sentiment analysis model is unavailable.")
    elif not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        try:
            result = sentiment_pipeline(user_input)
            if result and isinstance(result, list) and "label" in result[0]:
                st.write("Predicted Sentiment:", result[0]["label"])
                st.write("Confidence Score:", result[0]["score"])
            else:
                st.error("Unexpected response format from the model.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

import streamlit as st
from transformers import pipeline

# Initialize the sentiment analysis pipeline with the chosen model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="siebert/sentiment-roberta-large-english",
    truncation=True  # ensures input texts are appropriately truncated
)

st.title("Sentiment Analysis with RoBERTa-Large")

# Get user input
user_input = st.text_area("Enter text for sentiment prediction:")

if st.button("Analyze"):
    if user_input:
        # Get prediction from the model
        result = sentiment_pipeline(user_input)
        st.write("Predicted Sentiment:", result[0]["label"])
        st.write("Confidence Score:", result[0]["score"])
    else:
        st.warning("Please enter some text to analyze.")

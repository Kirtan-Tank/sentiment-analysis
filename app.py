import streamlit as st
from transformers import pipeline

# Inject custom CSS for a clean, modern, and rich look
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

body {
    font-family: 'Roboto', sans-serif;
    background-color: #f4f7f6;
    color: #2E4053;
}

h1 {
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.5rem;
}

h3 {
    text-align: center;
    color: #2E4053;
    margin-bottom: 2rem;
}

.stButton>button {
    background-color: #2E86C1;
    color: white;
    border-radius: 5px;
    border: none;
    padding: 10px 24px;
    font-size: 1rem;
    font-weight: 600;
}

.stTextInput>div>div>input, 
.stTextArea>div>div>textarea {
    font-size: 1.1rem;
    padding: 10px;
    border: 1px solid #dfe6e9;
    border-radius: 4px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Updated app title and tagline
st.markdown("<h1>SentiAnalyze: Unveil the Emotions</h1>", unsafe_allow_html=True)
st.markdown("<h3>Analyze sentiment with style and precision</h3>", unsafe_allow_html=True)

# Initialize the sentiment analysis pipeline with the chosen model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="siebert/sentiment-roberta-large-english",
    truncation=True  # ensures input texts are appropriately truncated
)

# Get user input
user_input = st.text_area("Enter text for sentiment prediction:")

if st.button("Analyze"):
    if user_input.strip():
        try:
            result = sentiment_pipeline(user_input)
            if result and isinstance(result, list) and "label" in result[0]:
                st.markdown(f"""
                    <div style='text-align: center; font-size: 1.2rem;'>
                        <strong>Predicted Sentiment:</strong> {result[0]['label']}<br>
                        <strong>Confidence Score:</strong> {result[0]['score']:.2f}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Unexpected response format from the model.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text to analyze.")

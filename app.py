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

# Sidebar: select mode (Basic vs. Advanced)
mode = st.sidebar.radio("Select Mode", ["Basic (Sentiment Analysis)", "Advanced (Emotion Detection)"])

# Load the appropriate sentiment analysis pipeline based on the mode
@st.cache_resource(show_spinner=False)
def load_pipeline(selected_mode):
    try:
        if selected_mode == "Basic (Sentiment Analysis)":
            model_name = "siebert/sentiment-roberta-large-english"
        else:
            model_name = "SamLowe/roberta-base-go_emotions"
        pl = pipeline("sentiment-analysis", model=model_name, truncation=True)
        return pl, model_name
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

sentiment_pipeline, used_model = load_pipeline(mode)

# Display current mode in the title
st.markdown(f"<h1>SentiAnalyze: {mode}</h1>", unsafe_allow_html=True)
st.markdown("<h3>Analyze sentiment with style and precision</h3>", unsafe_allow_html=True)

# In Advanced mode, offer a button to display available emotion classes
if mode == "Advanced (Emotion Detection)":
    if st.sidebar.button("Show Available Emotion Classes"):
        try:
            classes = sentiment_pipeline.model.config.id2label
            st.sidebar.markdown("### Available Emotion Classes:")
            for idx, label in classes.items():
                st.sidebar.write(f"{idx}: {label}")
        except Exception as e:
            st.sidebar.error(f"Error fetching emotion classes: {e}")

# Main UI: text input for sentiment prediction
user_input = st.text_area("Enter text for analysis:")

if st.button("Analyze"):
    if not sentiment_pipeline:
        st.error("The sentiment analysis model is unavailable.")
    elif not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        try:
            result = sentiment_pipeline(user_input)
            if result and isinstance(result, list) and "label" in result[0]:
                st.markdown(f"""
                    <div style='text-align: center; font-size: 1.2rem;'>
                        <strong>Predicted Label:</strong> {result[0]['label']}<br>
                        <strong>Confidence Score:</strong> {result[0]['score']:.2f}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Unexpected response format from the model.")
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

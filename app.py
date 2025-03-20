import streamlit as st
from transformers import pipeline

# Base custom CSS for a clean, modern look
base_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

body {
    font-family: 'Roboto', sans-serif;
    background-color: #f4f7f6;
    color: #2E4053;
    transition: background-color 0.5s ease-in-out;
}

h1 {
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.5rem;
    transition: color 0.5s ease-in-out;
}

h3 {
    text-align: center;
    color: #2E4053;
    margin-bottom: 2rem;
    transition: opacity 0.5s ease-in-out;
}

.stButton>button {
    background-color: #2E86C1;
    color: white;
    border-radius: 5px;
    border: none;
    padding: 10px 24px;
    font-size: 1rem;
    font-weight: 600;
    transition: background-color 0.3s ease;
}

.stButton>button:hover {
    background-color: #1b4f72;
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
st.markdown(base_css, unsafe_allow_html=True)

# Advanced mode additional CSS
advanced_css = """
<style>
body.advanced-mode {
    background-color: #e8f0fe;
}
.fade-in {
    animation: fadeIn 1s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
</style>
"""

# Add a Reload button to clear caches and restart the app
if st.sidebar.button("Reload App"):
    st.cache_resource.clear()
    st.experimental_rerun()

# Use session state to track the last selected mode
if 'last_mode' not in st.session_state:
    st.session_state.last_mode = None

# Sidebar: select mode (Basic vs. Advanced)
mode = st.sidebar.radio("Select Mode", ["Basic (Sentiment Analysis)", "Advanced (Emotion Detection)"])

# If mode has changed, clear cache and rerun to free memory
if st.session_state.last_mode != mode:
    st.cache_resource.clear()
    st.session_state.last_mode = mode
    st.experimental_rerun()

# For Advanced mode, add password protection
if mode == "Advanced (Emotion Detection)":
    adv_password = st.sidebar.text_input("Enter password for advanced mode", type="password")
    if adv_password != "advanced123":
        st.sidebar.error("Incorrect password. Advanced mode is locked.")
        # Force fallback to Basic mode by setting mode and clearing cache
        mode = "Basic (Sentiment Analysis)"
        st.sidebar.warning("Switching to Basic mode.")
        st.cache_resource.clear()
        st.session_state.last_mode = mode
        st.experimental_rerun()

# If advanced mode, inject additional CSS and add a class to the body via JS
if mode == "Advanced (Emotion Detection)":
    st.markdown(advanced_css, unsafe_allow_html=True)
    st.markdown(
        """
        <script>
        document.body.classList.add("advanced-mode");
        </script>
        """,
        unsafe_allow_html=True,
    )

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

# Display current mode in the title with an animated header in advanced mode
if mode == "Advanced (Emotion Detection)":
    st.markdown("<h1 class='fade-in'>SentiAnalyze: Advanced Emotion Detection</h1>", unsafe_allow_html=True)
else:
    st.markdown("<h1>SentiAnalyze: Basic Sentiment Analysis</h1>", unsafe_allow_html=True)
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

# Main UI: text input for analysis
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

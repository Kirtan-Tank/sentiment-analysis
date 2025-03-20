import streamlit as st
from transformers import pipeline
import time

# Try importing psutil; if not available, set to None.
try:
    import psutil
except ImportError:
    psutil = None

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

# Sidebar: Memory usage indicator and manual buttons
def display_sidebar_controls():
    if psutil:
        memory_usage = psutil.virtual_memory().percent
        st.sidebar.metric("Memory Usage", f"{memory_usage}%")
    else:
        st.sidebar.warning("psutil not installed. Memory usage unavailable.")
    
    clear_placeholder = st.sidebar.empty()
    if st.sidebar.button("Clear Memory"):
        st.cache_resource.clear()
        clear_placeholder.success("Memory cache cleared!")
        time.sleep(1)
        clear_placeholder.empty()
    
    reload_placeholder = st.sidebar.empty()
    if st.sidebar.button("Reload App"):
        st.cache_resource.clear()
        reload_placeholder.info("Cache cleared. Please reload your browser to apply changes.")
        time.sleep(1)
        reload_placeholder.empty()

display_sidebar_controls()

# Use session state to track the last selected mode
if 'last_mode' not in st.session_state:
    st.session_state.last_mode = "Basic (Sentiment Analysis)"

# Sidebar: select mode (Basic vs. Advanced)
mode = st.sidebar.radio("Select Mode", ["Basic (Sentiment Analysis)", "Advanced (Emotion Detection)"])

# Automatically clear cache and notify if mode has changed
if st.session_state.last_mode != mode:
    st.cache_resource.clear()
    st.session_state.last_mode = mode
    st.sidebar.success("Memory cleared automatically due to mode change.")
    
# For Advanced mode, add password protection and clear cache automatically upon success
if mode == "Advanced (Emotion Detection)":
    adv_password = st.sidebar.text_input("Enter password for advanced mode", type="password")
    if adv_password == "advanced123":
        st.sidebar.success("Advanced mode unlocked!")
        st.cache_resource.clear()
        st.sidebar.success("Memory cleared automatically upon unlocking advanced mode.")
    else:
        st.sidebar.error("Incorrect password. Advanced mode is locked. Switching to Basic mode.")
        mode = "Basic (Sentiment Analysis)"
        st.cache_resource.clear()
        st.session_state.last_mode = mode
        st.sidebar.success("Memory cleared automatically due to mode switch.")

# If Advanced mode, inject additional CSS and add a class to the body via JS
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

# Display current mode in the title with an animated header (if Advanced)
if mode == "Advanced (Emotion Detection)":
    st.markdown("<h1 class='fade-in'>SentiAnalyze: Advanced Emotion Detection</h1>", unsafe_allow_html=True)
else:
    st.markdown("<h1>SentiAnalyze: Basic Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<h3>Analyze sentiment with style and precision</h3>", unsafe_allow_html=True)

# Offer a button to display available classes in both modes
if st.sidebar.button("Show Available Classes"):
    try:
        classes = sentiment_pipeline.model.config.id2label
        st.sidebar.markdown("### Available Classes:")
        for idx, label in classes.items():
            st.sidebar.write(f"{idx}: {label}")
    except Exception as e:
        st.sidebar.error(f"Error fetching classes: {e}")

# Function to generate dynamic messages based on predicted label and confidence
def get_dynamic_message(label, score):
    label = label.upper()
    if label == "POSITIVE":
        if score >= 0.90:
            return "Absolutely glowing! The positive energy is off the charts!"
        elif score >= 0.75:
            return "Clearly positive sentiment. Great vibes ahead!"
        else:
            return "Positive sentiment, though a bit tentative."
    elif label == "NEGATIVE":
        if score >= 0.90:
            return "Extremely negative sentiment. That's quite disheartening."
        elif score >= 0.75:
            return "Clearly negative sentiment. Something might be off."
        else:
            return "Negative sentiment, but not overwhelmingly so."
    elif label == "NEUTRAL":
        if score >= 0.80:
            return "Strongly neutral. The text is very balanced."
        else:
            return "Somewhat neutral. There's a hint of emotion."
    else:
        return "The sentiment is interesting!"

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
                predicted_label = result[0]['label']
                confidence = result[0]['score']
                dynamic_msg = get_dynamic_message(predicted_label, confidence)
                st.markdown(f"""
                    <div style='text-align: center; font-size: 1.2rem;'>
                        <strong>Predicted Label:</strong> {predicted_label}<br>
                        <strong>Confidence Score:</strong> {confidence:.2f}<br>
                        <em>{dynamic_msg}</em>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Unexpected response format from the model.")
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

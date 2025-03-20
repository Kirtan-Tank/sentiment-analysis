import streamlit as st
from transformers import pipeline
import time

# Try importing psutil; if not available, set to None.
try:
    import psutil
except ImportError:
    psutil = None

# New dark theme CSS using Montserrat and a dark gradient background with card-like containers.
dark_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

body {
    font-family: 'Montserrat', sans-serif;
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #e0e0e0;
    transition: background 0.5s ease-in-out;
    margin: 0;
    padding: 0;
}

.container {
    background: rgba(0, 0, 0, 0.6);
    border-radius: 15px;
    padding: 30px;
    margin: 40px auto;
    max-width: 800px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.5);
}

h1 {
    font-size: 3rem;
    text-align: center;
    margin-bottom: 10px;
    color: #ffffff;
}

h3 {
    font-size: 1.5rem;
    text-align: center;
    margin-bottom: 20px;
    font-weight: 400;
    color: #cccccc;
}

button, .stButton>button {
    background: #ff758c;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 1rem;
    color: #ffffff;
    cursor: pointer;
    transition: background 0.3s ease;
}

button:hover, .stButton>button:hover {
    background: #ff5f7e;
}

input, textarea {
    width: 100%;
    padding: 12px;
    margin-bottom: 15px;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    background: #333333;
    color: #e0e0e0;
}
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# Advanced mode additional CSS with a different dark gradient and fade-in
advanced_css = """
<style>
body.advanced-mode {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
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

# Sidebar: Memory usage indicator and manual buttons with transient notifications
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

# For Advanced mode, add password protection and automatically clear cache upon success
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

# Create a container for the main UI (modern card-like look)
with st.container():
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
                    
                    # Generate dynamic message templates based on confidence and label
                    if predicted_label.upper() == "POSITIVE":
                        dynamic_msg = "Absolutely glowing! The positive energy is off the charts!"
                        st.balloons()  # celebratory animation
                    elif predicted_label.upper() == "NEGATIVE":
                        dynamic_msg = "Extremely negative sentiment. That's quite disheartening."
                        st.snow()  # somber, falling snow animation as a proxy for sadness
                    elif predicted_label.upper() == "NEUTRAL":
                        dynamic_msg = "The text is balanced and neutral."
                    else:
                        dynamic_msg = "The sentiment is interesting!"
                    
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

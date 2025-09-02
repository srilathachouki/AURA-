import streamlit as st
import requests
from streamlit_lottie import st_lottie

# It's good practice to set the page config in each page file
st.set_page_config(page_title="About AURA AI", page_icon="📖", layout="wide")

def local_css():
    """Injects custom CSS for a vibrant, modern look."""
    st.markdown("""
    <style>
        /* Main background gradient */
        .stApp {
            background-image: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: white;
        }
        /* Style for headers to make them stand out */
        h1, h2, h3 {
            color: #FFFFFF !important;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        }
    </style>
    """, unsafe_allow_html=True)

# Apply the custom CSS
local_css()

# --- Page Content ---

st.title("About the AURA AI Project 📖")
st.markdown("---")

st.header("✨ Key Features")
st.markdown(
    """
    - **Dual-Logic AI**: Uses a local `scikit-learn` model for simple intents and the Google Gemini API for advanced conversations.
    - **Voice-Enabled Interaction**: Full support for Speech-to-Text (voice input) and Text-to-Speech (voice output).
    - **Interactive UI**: A modern front-end built with Streamlit, featuring a "glassmorphism" theme.
    - **Multi-Page Design**: A complete website experience with a home page, chatbot, and this documentation page.
    - **Cloud-Native**: Designed and optimized for deployment on cloud platforms like Streamlit Community Cloud.
    """
)
st.markdown("---")

st.header("🏗️ Architecture")
st.write(
    """
    AURA AI processes user input through a streamlined, conditional workflow. 
    When a prompt is received (either via text or voice), it's first evaluated by the local intent model. If the model recognizes the intent with high confidence (e.g., a simple "hello"), it provides a predefined response instantly. 
    If the intent is not recognized, the prompt is securely passed to the Google Gemini API for a detailed, generative response.
    """
)
# Using an external image link for the architecture diagram
st.image("https://i.imgur.com/wS41n2i.png", caption="AURA AI High-Level Architecture", use_column_width=True)
st.markdown("---")

st.header("🛠️ Technology Stack")
st.code(
    """
    - Frontend: Streamlit
    - Backend & Core Logic: Python
    - Generative AI: Google Gemini API
    - Local Intent Classification: Scikit-learn
    - Speech-to-Text: SpeechRecognition
    - Text-to-Speech: gTTS (Google Text-to-Speech)
    """,
    language="markdown"
)
st.markdown("---")

st.header("👤 Author")
st.info("**Chouki Srilatha**", icon="🧑‍💻")
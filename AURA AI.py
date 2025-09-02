import streamlit as st
from streamlit_lottie import st_lottie
import requests

# Set the page configuration. This should be the first Streamlit command.
st.set_page_config(page_title="AURA AI Home", page_icon="🔮", layout="wide")

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

def load_lottieurl(url: str):
    """Fetches a Lottie animation from a URL."""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None

# Apply the custom CSS
local_css()

# --- Page Content ---
st.title("Welcome to AURA AI 🔮")
st.markdown("### Your intelligent, voice-enabled assistant built with the power of Google's Gemini.")
st.markdown("---")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("What is AURA AI?")
    st.write(
        """
        AURA AI is more than just a chatbot. It's an interactive experience designed to be your creative partner, knowledge base, and helpful assistant. 
        
        It uses a dual-logic system:
        - A **local intent model** for lightning-fast answers to common questions.
        - The powerful **Google Gemini API** for complex, creative, and in-depth conversations.

        **Select the `AURA AI Chatbot` from the sidebar to start your conversation!**
        """
    )
    st.info("Navigate to the **About the Project** page to learn about the architecture and technology behind AURA.", icon="💡")

with col2:
    lottie_url = "https://assets10.lottiefiles.com/packages/lf20_v1yudlrx.json"
    lottie_animation = load_lottieurl(lottie_url)
    if lottie_animation:
        st_lottie(lottie_animation, speed=1, width=400, height=400, key="home_robot")
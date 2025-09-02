import streamlit as st
import json
import random
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import requests

# Voice Imports
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO

# Animation Import
from streamlit_lottie import st_lottie

# NOTE: st.set_page_config() is removed from this file. 
# It is now set in the main Home.py file.

# Load environment variables
load_dotenv()
    
# --- INTENTS JSON ---
INTENTS_JSON = """
{
  "intents": [
    { "tag": "greeting", "patterns": ["hi", "hello", "hey", "how are you"], "responses": ["Hello! How can I assist you today? 🚀", "Hi there! What can I do for you?"] },
    { "tag": "goodbye", "patterns": ["bye", "see you later", "goodbye"], "responses": ["Goodbye! Have a great day! 👋", "See you later!"] },
    { "tag": "thanks", "patterns": ["thanks", "thank you", "helpful"], "responses": ["You're welcome! 😊", "Happy to help!"] },
    { "tag": "about", "patterns": ["who are you", "what are you"], "responses": ["I am a chatbot powered by Streamlit and Google's Gemini API! 🤖", "I'm a language model's interface. Ask me anything!"] },
    { "tag": "creator", "patterns": ["who made you", "who created you"], "responses": ["I was built by Chouki Srilatha using Python and Google's Gemini.", "My creator brought me to life using a powerful suite of tools."] }
  ]
}
"""

# --- STYLING AND ANIMATION ---
def load_lottieurl(url: str):
    """Fetches a Lottie animation from a URL."""
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not load Lottie animation: {e}")
        return None

def local_css():
    """Injects custom CSS for a vibrant, modern look."""
    st.markdown("""
    <style>
        /* Main background gradient */
        .stApp {
            background-image: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: white;
        }
        /* Chat messages styling (Glassmorphism) */
        [data-testid="stChatMessage"] {
            border-radius: 20px;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
            background-color: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }
        [data-testid="stChatMessageContent"] { color: #E0E0E0; }
        /* Title and caption */
        h1, .st-caption {
            text-align: center;
            color: #FFFFFF !important;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        }
        /* Chat input */
        [data-testid="stChatInput"] {
            background-color: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
        }
        [data-testid="stChatInput"] > div > div > textarea { color: white; }

        /* Microphone button */
        [data-testid="stButton"] button {
            background-color: #7b2cbf;
            color: white;
            border: none;
            border-radius: 10px;
            width: 100%;
            height: 50px;
            font-size: 24px;
            transition: all 0.3s ease;
        }
        [data-testid="stButton"] button:hover {
            background-color: #9d4edd;
            transform: scale(1.05);
        }
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: rgba(15, 12, 41, 0.8);
            backdrop-filter: blur(5px);
        }
    </style>
    """, unsafe_allow_html=True)

# --- VOICE HELPER FUNCTIONS ---
def text_to_speech(text):
    """Converts text to speech using gTTS and plays it in Streamlit."""
    try:
        tts = gTTS(text=text, lang='en')
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        st.audio(audio_fp, format='audio/mp3')
    except Exception as e:
        st.error(f"Failed to generate speech: {e}")

def speech_to_text():
    """Listens for voice input and converts it to text."""
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Listening... Speak now!")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            st.info("Recognizing...")
            text = r.recognize_google(audio)
            return text
    except AttributeError:
        st.error("No microphone found. Please ensure your microphone is enabled and permissions are granted.", icon="🎙️")
    except sr.UnknownValueError:
        st.warning("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        st.warning(f"An error occurred during speech recognition: {e}")
    return None

# --- CORE LOGIC ---
@st.cache_data
def load_intent_model():
    """Loads and trains the local intent classification model."""
    data = json.loads(INTENTS_JSON)
    patterns, tags = [], []
    responses_dict = {intent['tag']: intent['responses'] for intent in data['intents']}
    for intent in data['intents']:
        for pattern in intent['patterns']:
            patterns.append(pattern.lower())
            tags.append(intent['tag'])
    
    model = make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression(max_iter=500, random_state=42))
    model.fit(patterns, tags)
    return model, responses_dict

@st.cache_resource
def load_gemini_model():
    """Loads the Gemini generative model."""
    return genai.GenerativeModel('gemini-1.5-flash-latest')

def get_chatbot_response(user_input, _intent_model, _responses_dict, _gemini_model):
    """Determines the appropriate response from either local intents or the Gemini API."""
    user_input_lower = user_input.lower()
    probabilities = _intent_model.predict_proba([user_input_lower])[0]
    
    if max(probabilities) > 0.65:
        predicted_tag = _intent_model.classes_[probabilities.argmax()]
        return random.choice(_responses_dict[predicted_tag])
    else:
        try:
            response = _gemini_model.generate_content(user_input)
            return response.text
        except Exception as e:
            st.error(f"An error occurred with the Google Gemini API: {e}", icon="🚨")
            return "Sorry, I'm having trouble connecting to my brain right now."

# --- MAIN APP ---

# Apply the custom CSS
local_css()

# Load models
intent_model, intent_responses = load_intent_model()
gemini_model = load_gemini_model()

# Sidebar
with st.sidebar:
    lottie_url = "https://assets10.lottiefiles.com/packages/lf20_v1yudlrx.json"
    lottie_animation = load_lottieurl(lottie_url)
    if lottie_animation:
        st_lottie(lottie_animation, speed=1, width=250, height=250, key="robot")
    
    st.header("About This App")
    st.markdown("This is a voice-enabled chatbot that uses local intents and the Google Gemini API to answer your questions.")
    
    st.markdown("---")
    st.header("Voice Input")
    if st.button("Click to Speak 🎤", help="Use your voice to chat"):
        voice_prompt = speech_to_text()
        if voice_prompt:
            process_and_display_chat(voice_prompt, intent_model, intent_responses, gemini_model)

st.title("🔮 AURA AI Chatbot")
st.caption("Your friendly AI assistant. Try typing or using the mic.")

# API Key Validation
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("Your Google API key is not set! Please add it to your `.env` file.", icon="🔑")
    st.stop()
else:
    genai.configure(api_key=google_api_key)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def process_and_display_chat(prompt, intent_model, responses_dict, gemini_model):
    """Main function to process a prompt and update the chat UI."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_chatbot_response(prompt, intent_model, responses_dict, gemini_model)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    text_to_speech(response)

# --- Input Handling ---
if text_prompt := st.chat_input("What's on your mind?"):
    process_and_display_chat(text_prompt, intent_model, intent_responses, gemini_model)
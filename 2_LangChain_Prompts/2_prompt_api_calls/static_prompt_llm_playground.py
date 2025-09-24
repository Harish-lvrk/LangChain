import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# --- Load Environment Variables ---
# This loads API keys from your .env file
load_dotenv()

# --- Check for API Keys ---
# The app will check for both keys and inform you if one is missing.
HUGGINGFACE_TOKEN_LOADED = "HUGGINGFACEHUB_API_TOKEN" in os.environ
GOOGLE_API_KEY_LOADED = "GOOGLE_API_KEY" in os.environ

if not HUGGINGFACE_TOKEN_LOADED and not GOOGLE_API_KEY_LOADED:
    st.error("No API keys found. Please create a .env file.", icon="🚨")
    st.info("Create a file named `.env` and add your API keys. See instructions below.")
    st.stop()


# --- Application Code ---

st.title("🚀 Unified AI Model Hub")
st.markdown("Use Hugging Face models and Google's Gemini in one place.")

# --- Available Models (Updated List) ---
# We now include Gemini. Its value is its model name, which we'll use to identify it.
AVAILABLE_MODELS = {
    "Google Gemini Flash": "gemini-1.5-flash-latest", # Gemini Model
    "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.3", # Hugging Face
    "Zephyr-7B-Alpha": "HuggingFaceH4/zephyr-7b-alpha", # Hugging Face
}

# --- UI Controls ---

# User input
base_prompt = st.text_area("Enter your prompt:", "Explain the concept of neural networks.")

# Model selector
model_choice_key = st.selectbox("Choose a model:", list(AVAILABLE_MODELS.keys()))
model_repo_id = AVAILABLE_MODELS[model_choice_key]

# Difficulty levels mapping
DIFFICULTY_PROMPTS = {
    "Simple": "Explain in very simple words, as if teaching a beginner: ",
    "Medium": "Explain clearly with some detail, as if teaching a student: ",
    "Hard": "Explain in depth with advanced details, as if teaching an expert: ",
}
difficulty = st.radio("Choose difficulty:", list(DIFFICULTY_PROMPTS.keys()), index=0)

# Creativity levels mapped to temperature
CREATIVITY_LEVELS = {
    "Low (Focused)": 0.1,
    "Medium (Balanced)": 0.7,
    "High (Creative)": 1.2,
}
creativity = st.radio("Choose creativity level:", list(CREATIVITY_LEVELS.keys()), index=1)
temperature = CREATIVITY_LEVELS[creativity]


# --- Generate Button & Logic ---

if st.button("Generate Response"):
    # Check if the required API key for the selected model is available
    if model_choice_key == "Google Gemini Flash" and not GOOGLE_API_KEY_LOADED:
        st.error("Google API Key not found in .env file. Please add it to use Gemini.", icon="🔑")
    elif model_choice_key != "Google Gemini Flash" and not HUGGINGFACE_TOKEN_LOADED:
        st.error("Hugging Face Token not found in .env file. Please add it to use this model.", icon="🔑")
    else:
        with st.spinner(f"Calling {model_choice_key}..."):
            try:
                # Modify prompt with difficulty
                final_prompt = DIFFICULTY_PROMPTS[difficulty] + base_prompt
                chat_model = None

                # --- CONDITIONAL LOGIC: GEMINI vs HUGGING FACE ---
                if model_choice_key == "Google Gemini Flash":
                    # Initialize and use the Gemini model
                    chat_model = ChatGoogleGenerativeAI(
                        model=model_repo_id,
                        temperature=temperature,
                        convert_system_message_to_human=True # Helps with some prompts
                    )
                else:
                    # Initialize and use a Hugging Face model
                    llm = HuggingFaceEndpoint(
                        repo_id=model_repo_id,
                        task="text-generation",
                        temperature=temperature,
                        max_new_tokens=512, # Increased token limit slightly
                    )
                    chat_model = ChatHuggingFace(llm=llm)

                # Invoke the selected model
                if chat_model:
                    response = chat_model.invoke([HumanMessage(content=final_prompt)])
                    st.subheader("💡 Response:")
                    st.write(response.content)

            except Exception as e:
                # Catch and display any errors from the API calls
                st.error(f"An error occurred: {e}", icon="❌")

# --- Instructions for .env file ---
st.markdown("---")
with st.expander("API Key Instructions"):
    st.info("""
        To use this application, you need to provide your API keys in a `.env` file in the same directory as the app.

        **1. Create a file named `.env`**
        **2. Add your keys like this:**

        ```
        # For Hugging Face Models
        HUGGINGFACEHUB_API_TOKEN="hf_..."

        # For Google Gemini Model
        GOOGLE_API_KEY="AI..."
        ```
        You can use either one, or both. The app will only show an error if you try to use a model without providing its corresponding key.
    """)


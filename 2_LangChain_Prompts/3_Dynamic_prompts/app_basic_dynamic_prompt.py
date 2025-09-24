# research_assistant_app.py

# Import necessary libraries
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# --- Page Configuration ---
st.set_page_config(
    page_title="🤖 AI Research Co-Pilot",
    page_icon="🧠",
    layout="centered"
)

# --- Session State Initialization ---
# We need to track the chat history, the user's last raw prompt,
# the choice to refine, and the final prompt for confirmation.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = None
if "refine_choice_pending" not in st.session_state:
    st.session_state.refine_choice_pending = False
if "prompt_for_confirmation" not in st.session_state:
    st.session_state.prompt_for_confirmation = None


# --- API Key Management ---
load_dotenv()
HUGGINGFACE_TOKEN_LOADED = "HUGGINGFACEHUB_API_TOKEN" in os.environ
GOOGLE_API_KEY_LOADED = "GOOGLE_API_KEY" in os.environ

# --- Main Application Title ---
st.title("🤖 AI Research Co-Pilot")
st.markdown("Set your research context in the sidebar. I will help you craft the perfect prompt for the best possible answer.")

# --- Sidebar for Context and Controls ---
with st.sidebar:
    st.header("⚙️ Research Context")
    AVAILABLE_MODELS = {
        "Google Gemini Flash": "gemini-1.5-flash-latest",
        "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
        "Zephyr-7B-Alpha": "HuggingFaceH4/zephyr-7b-alpha",
    }
    model_choice_key = st.selectbox("Choose an AI model:", list(AVAILABLE_MODELS.keys()))
    model_repo_id = AVAILABLE_MODELS[model_choice_key]

    RESEARCH_TOPICS = [
        "Word2Vec", "GloVe", "BERT", "GPT-4", "Transformers", "Attention Mechanism",
        "LSTM", "RNN", "CNN", "ResNet", "Reinforcement Learning", "Q-Learning",
        "GANs", "Diffusion Models", "Quantum Computing", "Support Vector Machine",
        "Federated Learning", "Graph Neural Networks", "Knowledge Graphs"
    ]
    topic_choice = st.selectbox("Set the main research topic:", RESEARCH_TOPICS)

    DIFFICULTY_LEVELS = ["Simple", "Medium", "Hard"]
    difficulty_choice = st.selectbox("Set explanation difficulty:", DIFFICULTY_LEVELS, index=1)

    CREATIVITY_TEMPERATURE = {"Low": 0.2, "Medium": 0.7, "High": 1.2}
    creativity_choice = st.select_slider("Set creativity level:", options=CREATIVITY_TEMPERATURE.keys(), value="Medium")
    temperature = CREATIVITY_TEMPERATURE[creativity_choice]

    with st.expander("API Key Instructions"):
        st.info("Create a `.env` file and add:\n\n`HUGGINGFACEHUB_API_TOKEN='hf_...'`\n\n`GOOGLE_API_KEY='AI...'`")

# --- Core Functions ---

def get_ai_response(prompt_text):
    """A centralized function to call the selected AI model."""
    try:
        chat_model = None
        is_gemini = "Gemini" in model_choice_key
        if is_gemini:
            chat_model = ChatGoogleGenerativeAI(model=model_repo_id, temperature=temperature, convert_system_message_to_human=True)
        else:
            llm = HuggingFaceEndpoint(repo_id=model_repo_id, task="text-generation", temperature=max(0.1, temperature), max_new_tokens=1024)
            chat_model = ChatHuggingFace(llm=llm)
        
        response = chat_model.invoke([HumanMessage(content=prompt_text)])
        return response.content
    except Exception as e:
        return f"An error occurred: {e}"

def generate_base_prompt(topic, difficulty, user_question):
    """Constructs the high-quality base prompt by merging context with the user's question."""
    base = f"Regarding the research topic '{topic}', the user is asking the following question: '{user_question}'.\n"
    base += "Please provide a detailed, well-structured explanation. "
    if difficulty == "Simple":
        base += "Explain in simple, clear words suitable for a beginner. "
    elif difficulty == "Medium":
        base += "Explain with clear technical details, assuming the audience is an undergraduate student. "
    else: # Hard
        base += "Explain in depth with advanced technical details, mathematical formulations, and nuances suitable for a graduate researcher. "
    base += "The final response should be structured with clear headings and must enclose all mathematical formulas in LaTeX dollar signs (e.g., $E=mc^2$)."
    return base

def refine_prompt_with_ai(base_prompt):
    """Uses Gemini to refine the constructed prompt into an optimal one."""
    if not GOOGLE_API_KEY_LOADED:
        st.error("Google API Key is required for the prompt refinement feature. Please add it to your .env file.", icon="🔑")
        return None
    try:
        refiner_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
        refiner_system_prompt = (
            "You are a prompt engineering expert. Your task is to take a base prompt and refine it into a clear, "
            "detailed, and effective prompt for a large language model that acts as a research assistant. "
            "Ensure the final prompt is a direct instruction to the AI. Do not answer the prompt, only refine it."
        )
        response = refiner_model.invoke([HumanMessage(content=f"{refiner_system_prompt}\n\nBase Prompt:\n{base_prompt}")])
        return response.content
    except Exception as e:
        st.error(f"Error during prompt refinement: {e}", icon="❌")
        return base_prompt # Fallback to the base prompt on error

# --- Main Chat UI ---

# Display the existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- State 1: Ask user whether to refine prompt ---
if st.session_state.refine_choice_pending:
    with st.container(border=True):
        st.write("How should I proceed with your question?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🤖 Refine it for me", use_container_width=True, type="primary"):
                st.session_state.refine_choice_pending = False
                base_prompt = generate_base_prompt(topic_choice, difficulty_choice, st.session_state.user_prompt)
                with st.spinner("Co-Pilot is crafting the perfect prompt..."):
                    refined_prompt = refine_prompt_with_ai(base_prompt)
                if refined_prompt:
                    st.session_state.prompt_for_confirmation = refined_prompt
                st.rerun()
        with col2:
            if st.button("➡️ Use my prompt as is", use_container_width=True):
                st.session_state.refine_choice_pending = False
                final_prompt = generate_base_prompt(topic_choice, difficulty_choice, st.session_state.user_prompt)
                with st.spinner("Thinking..."):
                    response_content = get_ai_response(final_prompt)
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                st.rerun()

# --- State 2: Ask user to confirm the refined prompt ---
elif st.session_state.prompt_for_confirmation:
    with st.container(border=True):
        st.info("🧠 **Prompt Co-Pilot Suggestion**")
        st.markdown("Here is the improved prompt I'll use to get the best answer for you:")
        st.caption(st.session_state.prompt_for_confirmation)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Proceed", use_container_width=True, type="primary"):
                confirmed_prompt = st.session_state.prompt_for_confirmation
                st.session_state.prompt_for_confirmation = None # Clear state
                with st.spinner("Thinking..."):
                    response_content = get_ai_response(confirmed_prompt)
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                st.rerun()
        with col2:
            if st.button("❌ Cancel & Rephrase", use_container_width=True):
                st.session_state.prompt_for_confirmation = None # Clear state
                st.rerun()

# --- Chat Input Logic ---
# Disable input while waiting for any user choice.
is_disabled = st.session_state.refine_choice_pending or bool(st.session_state.prompt_for_confirmation)
if prompt := st.chat_input("Ask a follow-up question...", disabled=is_disabled):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.user_prompt = prompt
    st.session_state.refine_choice_pending = True # Trigger the first choice dialog
    st.rerun()


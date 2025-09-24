import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage

load_dotenv()

st.title("🚀 Hugging Face Models with Difficulty & Creativity")

# Available models
AVAILABLE_MODELS = {
    "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "Zephyr-7B-Alpha": "HuggingFaceH4/zephyr-7b-alpha",
    "Falcon-7B-Instruct": "tiiuae/falcon-7b-instruct",
    "TinyLlama-1.1B-Chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# Difficulty levels mapping
DIFFICULTY_PROMPTS = {
    "Simple": "Explain in very simple words, as if teaching a beginner: ",
    "Medium": "Explain clearly with some detail, as if teaching a student: ",
    "Hard": "Explain in depth with advanced details, as if teaching an expert: ",
}

# Creativity levels mapped to temperature
CREATIVITY_LEVELS = {
    "Low (Deterministic)": 0.0,
    "Medium (Balanced)": 0.7,
    "High (Creative)": 1.2,
}

# User input
base_prompt = st.text_area("Enter your prompt:", "Explain Artificial Intelligence.")

# Difficulty selector
difficulty = st.radio("Choose difficulty:", list(DIFFICULTY_PROMPTS.keys()), index=0)

# Model selector
model_choice = st.selectbox("Choose a model:", list(AVAILABLE_MODELS.keys()))

# Creativity selector (instead of temperature numbers)
creativity = st.radio("Choose creativity level:", list(CREATIVITY_LEVELS.keys()), index=1)
temperature = CREATIVITY_LEVELS[creativity]

# Generate button
if st.button("Generate Response"):
    with st.spinner("Thinking..."):
        llm = HuggingFaceEndpoint(
            repo_id=AVAILABLE_MODELS[model_choice],
            task="text-generation",
            temperature=temperature,
            max_new_tokens=256,
        )
        chat_model = ChatHuggingFace(llm=llm)

        # Modify prompt with difficulty
        final_prompt = DIFFICULTY_PROMPTS[difficulty] + base_prompt

        response = chat_model.invoke([HumanMessage(content=final_prompt)])

        st.subheader("Response:")
        st.write(response.content)

# Grab the code from the 1_dynamic_prompt from diff cells 


import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage

load_dotenv()

st.title("📚 Research Paper Assistant - Dynamic PromptTemplate Example")

# Predefined topics
RESEARCH_TOPICS = [
    "Word2Vec", "BERT", "GPT", "Transformers", "LSTM",
    "Reinforcement Learning", "GANs", "Diffusion Models", "Quantum Computing"
]

# Difficulty levels
DIFFICULTY_LEVELS = ["Simple", "Medium", "Hard"]

# Creativity levels (mapped to temperature)
CREATIVITY_LEVELS = {"Low":0.0, "Medium":0.7, "High":1.2}

# User selections
topic_choice = st.selectbox("Choose a research topic:", RESEARCH_TOPICS)
difficulty_choice = st.selectbox("Choose difficulty:", DIFFICULTY_LEVELS)
creativity_choice = st.selectbox("Choose creativity level:", list(CREATIVITY_LEVELS.keys()))
model_choice = st.selectbox("Choose a model:", ["Mistral-7B-Instruct", "TinyLlama-1.1B-Chat"])

temperature = CREATIVITY_LEVELS[creativity_choice]

# Define a dynamic prompt template
template = """
Generate a detailed explanation on the research topic: "{topic}".
Difficulty Level: {difficulty}
Include relevant mathematical equations, derivations, or examples if applicable.
"""

prompt = PromptTemplate(
    input_variables=["topic", "difficulty"],
    template=template
)

dynamic_prompt = prompt.format(topic=topic_choice, difficulty=difficulty_choice)

# Generate response
if st.button("Generate Research Content"):
    with st.spinner("Generating..."):
        # Map model_choice to repo_id
        repo_map = {
            "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
            "TinyLlama-1.1B-Chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        }
        llm = HuggingFaceEndpoint(
            repo_id=repo_map[model_choice],
            task="text-generation",
            temperature=temperature,
            max_new_tokens=512,
        )
        chat_model = ChatHuggingFace(llm=llm)
        response = chat_model.invoke([HumanMessage(content=dynamic_prompt)])
        st.subheader("Generated Research Content:")
        st.write(response.content)
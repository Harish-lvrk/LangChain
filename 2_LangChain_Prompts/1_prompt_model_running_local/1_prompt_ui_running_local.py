import streamlit as st
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# Streamlit title
st.title("🦙 Local TinyLlama Chat (LangChain + Streamlit)")

# Load local TinyLlama model
@st.cache_resource  # so the model loads only once
def load_model():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # lightweight model
    pipe = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",     # uses GPU if available, else CPU
        max_new_tokens=256,
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_model()

# User input prompt
user_prompt = st.text_area("Enter your prompt:", "Explain AI in simple words.")

# Generate button
if st.button("Generate"):
    with st.spinner("Thinking..."):
        response = llm.invoke(user_prompt)
        st.subheader("Response:")
        st.write(response)

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

# Use model_id instead of repo_id
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # <-- changed here
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 100
    }
)

# Wrap in ChatHuggingFace
chat_model = ChatHuggingFace(llm=llm)

# Run a query
result = chat_model.invoke("what is the current year")
print(result.content)

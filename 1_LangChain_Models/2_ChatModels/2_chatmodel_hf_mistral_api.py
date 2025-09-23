from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.3",  # Hosted on API
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("what is the current year")
print(result.content)
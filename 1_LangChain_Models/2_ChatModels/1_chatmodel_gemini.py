
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

print("✅ API keys loaded. Initializing Google Gemini model...")

# Initialize the Google Gemini model.
# It automatically finds your GOOGLE_API_KEY from the .env file.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

print("🚀 Model ready. Asking question...")

question = "What is the capital of India?"
response = llm.invoke(question)

print("\n--- Response ---")
print(response.content)
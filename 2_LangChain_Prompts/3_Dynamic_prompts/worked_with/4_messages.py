from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


from langchain_google_genai import GoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()
 

model = GoogleGenerativeAI(model = 'gemini-1.5-flash-latest')

messages = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content="Tell me about the Andreaj Karpathy in 100 words")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result))

print(messages)
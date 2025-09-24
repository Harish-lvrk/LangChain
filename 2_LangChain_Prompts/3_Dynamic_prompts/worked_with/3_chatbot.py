
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()



model = GoogleGenerativeAI(model = 'gemini-1.5-flash-latest')

chat_histroy = [
    SystemMessage(content= 'You are a helpful assistant')

]

while True:
    user_input = input("you:")
    chat_histroy.append(HumanMessage(content= user_input))

    if user_input == 'exit':
        break
    result = model.invoke(chat_histroy)
    chat_histroy.append(AIMessage(content= result))
    print("AI: ",result)
print(chat_histroy)
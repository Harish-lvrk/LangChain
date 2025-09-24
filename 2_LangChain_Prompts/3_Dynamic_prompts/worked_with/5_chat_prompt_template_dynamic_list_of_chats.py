from langchain_core.prompts import ChatPromptTemplate

# from langchain_core.messages import SystemMessage, HumanMessage , AIMessage

# chat_template = ChatPromptTemplate([
#     SystemMessage(content='You are an helpful {domain} expert'),
#      HumanMessage(content='Eplain in simple terms , what is {topic}')
# ])

# prompt = chat_template.invoke({'domain':'cricket','topic':'DuckOut'})
## here the problem is the due to this the values are not assigned into the place holders

chat_template = ChatPromptTemplate(
    [
        ('system','You are an helpful {domain} expert'),
        ('human', 'Eplain in simple terms , what is {topic}')
    ]
)

prompt = chat_template.invoke({'domain':'cricket','topic':'DuckOut'})

print(prompt)
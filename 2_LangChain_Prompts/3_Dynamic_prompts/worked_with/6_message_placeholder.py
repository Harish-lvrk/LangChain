from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#chat template 

chat_template = ChatPromptTemplate(
    [
        ('system','you are an helpful customer support agent'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human','{query}')
    ]

)

chat_history = []
with open('2_LangChain_Prompts/3_Dynamic_prompts/worked_with/chat_history.txt') as f:
    chat_history.extend(f.readlines())

prompt = chat_template.invoke({'chat_history':chat_history, 'query': 'where is my refund'})

print(prompt)

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'mistralai/Mistral-7B-Instruct-v0.3',
    task= 'text-generation',

)
model = ChatHuggingFace(llm= llm)

#first prompt
template1 = PromptTemplate(
    template= "write a detailed report on {topic} ",
    input_variables=['topic']
)

# second prompt

template2 = PromptTemplate(
    template="write a 5 line summary on the following text . /n {text}",
    input_variables=['text']
)

Prompt1 = template1.invoke({'topic':"black hole"})

result = model.invoke(Prompt1)

result
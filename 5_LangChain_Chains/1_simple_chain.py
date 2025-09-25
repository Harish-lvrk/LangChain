from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

model = GoogleGenerativeAI(model = 'gemini-2.5-flash')

template1 = PromptTemplate(
    template= "Describe about the {topic}",
    input_variables=['topic']
    
)
template2 = PromptTemplate(
    template="Give 5 points from the \n {text}",
    input_variables=['text']
)
parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'Lora  and QLora'})

print(result)


graph = chain.get_graph()
graph.print_ascii()
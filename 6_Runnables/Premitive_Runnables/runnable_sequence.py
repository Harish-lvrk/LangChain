from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence

from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model='gemini-2.5-flash-lite')

parser = StrOutputParser()

template = PromptTemplate(
    template= "write a joke on the {topic}",
    input_variables= ['topic']
)

chain = RunnableSequence(template, model, parser)

result = chain.invoke({'topic':'education system in india'})

print(result)

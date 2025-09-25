from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'google/gemma-2-2b-it',
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
parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'india'})


print(result)
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from dotenv import  load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= 'google/gemma-2-2b-it',
    task= 'text-generation',

)


model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template= "Give the name,age and city of a fictional person \n {format_instruction}",

    input_variables= [],

    partial_variables={'format_instruction': parser.get_format_instructions}
)


chian = template | model | parser

result = chian.invoke({})


print(result)
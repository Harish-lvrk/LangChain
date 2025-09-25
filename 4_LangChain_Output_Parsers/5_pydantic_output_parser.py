from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.1-Terminus',
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

class Person(BaseModel):
    name : str = Field(description='Name of the person')
    age: int = Field(description="Age of the person")
    city: str = Field(description="Name of the city the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Genarate the name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}


)

chain = template | model | parser

result = chain.invoke({"place":"indian"})

print(result)

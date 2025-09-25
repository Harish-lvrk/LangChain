from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from dotenv import  load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= 'google/gemma-2-2b-it',
    task= 'text-generation',

)


model = ChatHuggingFace(llm = llm)
schema = [
    ResponseSchema(name = 'fact_1', description= 'Fact_1 about the topic'),
    ResponseSchema(name = 'fact_2', description= 'Fact_2 about the topic'),
    ResponseSchema(name = 'fact_3', description= 'Fact_3 about the topic'),
]
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give me the three facts about the {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables= {'format_instruction':parser.get_format_instructions}
)


chain = template | model | parser

result = chain.invoke({"topic":'india'})

print(result['fact_1'])
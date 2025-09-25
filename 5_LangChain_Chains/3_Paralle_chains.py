from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()
# this is not a good emaple but i am doing
model = GoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Genarate the detailed Notes on the following {topic}",
    input_variables='topic'

)

prompt2 = PromptTemplate(
    template="Genarat The quiz on the follwoing {topic}",
    input_variables='topic',
)

prompt3 = PromptTemplate(
    template = "Merge the both notes and quiz into the single document.\n notes --> {notes} and quiz --> {quiz}",
    input_variables=['notes', 'quiz']


)

parallel_chain = RunnableParallel(
    {
        'notes' : prompt1 | model | parser,
        'quiz' : prompt2 | model | parser
    }
)

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

result = chain.invoke('Lora and QLora')

print(result)
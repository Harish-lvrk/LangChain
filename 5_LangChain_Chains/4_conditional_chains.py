from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from langchain.output_parsers import PydanticOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser = StrOutputParser()

# Step 1: Define schema for sentiment
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description="Classify the review sentiment as either 'positive' or 'negative'."
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Step 2: Prompt for classification
prompt1 = PromptTemplate(
    template="""You are a strict sentiment classifier.
Review: "{review}"

Classify the above review as either:
- positive
- negative

Return the output in the re
graph = chain.get_graph()
graph.print_ascii()quired format:
{format_instruction}
""",
    input_variables=['review'],
    partial_variables={'format_instruction': parser2.get_format_instructions()},
)

# Step 3: Prompts for responses
prompt2 = PromptTemplate(
    template="The customer gave positive feedback. Write a warm and appreciative reply:\n\n{review}",
    input_variables=['review'],
)

prompt3 = PromptTemplate(
    template="The customer gave negative feedback. Write a polite and empathetic reply with an offer to improve:\n\n{review}",
    input_variables=['review'],
)

# Step 4: Build classifier chain
classifier_chain = prompt1 | model | parser2

# Step 5: Conditional chain based on classification
conditional_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "The sentiment is neutral.")
)

# Step 6: Full chain
chain = classifier_chain | conditional_chain

# Example run
result = chain.invoke({'review': 'This mobile  was fantastic'})
print(result)


graph = chain.get_graph()
graph.print_ascii()
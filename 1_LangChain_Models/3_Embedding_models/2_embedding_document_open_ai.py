from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

document = [
    'New Delhi is the capital of the India',
    "Amaravathi is the capital of the AndhraPradesh",
    "Sun rise in the East"
]

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

result = embedding.embed_documents(document)

print(str(result))

# we get the 2d list withe embeddings


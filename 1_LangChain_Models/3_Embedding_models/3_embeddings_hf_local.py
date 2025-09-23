from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

text = "New Delhi is the capital of the India"

vector = embedding.embed_query(text)

print(str(vector))

#output
"""
0.009329627268016338, -0.05499333515763283, -0.015913289040327072, 0.06670641899108887, -0.08044518530368805, 0.08230892568826675, 0.0121979471296072, 0.01837763376533985, 0.007386824581772089, 0.007315260358154774, 0.049246646463871, 0.05397603660821915]
"""

# document embeddings
document = [
    'New Delhi is the capital of the India',
    "Amaravathi is the capital of the AndhraPradesh",
    "Sun rise in the East"
]

vector2 = embedding.embed_documents(document)

print(str(vector2))
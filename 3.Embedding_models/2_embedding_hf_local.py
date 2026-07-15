from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

document = ["Delhi is capital of India ",
            "Mumbai is the capital of Maharashtra",
            'Paris is the capital of France']

vector = embedding.embed_documents(document)

print(str(vector))
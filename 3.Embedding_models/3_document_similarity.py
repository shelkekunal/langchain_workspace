from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
documents = ["Virat Kohli is famous for aggressive batting and remarkable consistency across all cricket formats.",
             'MS Dhoni remained calm under pressure and finished many matches with brilliant leadership.',
             'Rohit Sharma effortlessly hits elegant sixes and owns multiple double centuries in ODIs.',
             'Jasprit Bumrah troubles batters with accurate yorkers and an unusual bowling action.',
             'Sachin Tendulkar inspired millions through extraordinary batting skills and a legendary international career.'
             ]

query = 'tell me about rohit sharma'

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores  = cosine_similarity([query_embedding], doc_embedding)[0]

index, score = sorted(list(enumerate(scores)),key = lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score:",scores)
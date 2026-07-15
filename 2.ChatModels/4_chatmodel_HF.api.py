from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)   
model = ChatHuggingFace(llm=llm)

result1 = model.invoke("What is the capital of India and tell 5 facts about it")
#result2 = model.invoke("write a ")
print(result1.content)
#print(result2.content)


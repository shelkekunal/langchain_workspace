from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Llama-3.1-8B-Instruct',
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are helpful assistant"),
    HumanMessage(content="tell me about langchain")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))
print(messages)
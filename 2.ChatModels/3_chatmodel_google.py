from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='')

result = model.invoke("Who is the Prime Minister of India")

print(result.content)
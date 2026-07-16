from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-12B-it",
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)

# chat template - summary
template1 = PromptTemplate(
    template= "Write down detail summary about {topic}",
    input_variables= ['topic']
)

template2 = PromptTemplate(
    template="Write a 100 words summary on following text. /n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'supernova'})

print(result)
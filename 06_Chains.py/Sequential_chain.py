from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template1 = PromptTemplate(
    template='Give detailed summary about {topic} ',
    input_variables=['topic']

)

template2 = PromptTemplate(
    template= 'write 5 pointer about text, \n {text}',
    input_variables=['text']
)

chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic':'maths for the universe'})

print(result)
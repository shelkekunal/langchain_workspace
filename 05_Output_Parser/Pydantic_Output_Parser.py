from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task= 'text-generation'
)

model = ChatHuggingFace(llm = llm)

class Person(BaseModel):

    name : str = Field(description='name of the moive')
    runtime : int = Field(description='runtime of the moive')
    protagonist : str = Field(description='name of the protagonist')

parser = PydanticOutputParser(pydantic_object=Person) 

template = PromptTemplate(
    template='Give the name, runtime and protagonist of the {name} movie \n {format_instruction}',
    input_variables=['name'],
    partial_variables={'format_instruction':parser.get_format_instructions}
)

chain = template | model | parser

final_result = chain.invoke({'name':'christopher nolan'})

print(final_result)
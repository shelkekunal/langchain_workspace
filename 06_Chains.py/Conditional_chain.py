from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task= 'text-generation'
)

model = ChatHuggingFace(llm=llm)

class Feedback(BaseModel):

    sentiment : Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

template1 = PromptTemplate(
    template = 'Give sentiment on the following sentence positive or negative \n {feedback}\n {format_instruction}',
    input_variable = ['feedback'],
    partial_variables= {'format_instruction':parser2.get_format_instructions()}
)

parser = StrOutputParser()

classifer_chain = template1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda( lambda x: "could not find the sentiment")
)

chain = classifer_chain | branch_chain

result = chain.invoke({'feedback':'This is a super smartphone'})

print(result)
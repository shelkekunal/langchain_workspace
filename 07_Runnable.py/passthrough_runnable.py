from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableParallel, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task = ' text-generation'
)

model = ChatHuggingFace(llm= llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= 'write 2 jokes on {topic}',
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template='explain first joke on {text}',
    input_variables= ['text']
)

joke_chain = RunnableSequence(prompt1, model, parser)

parellel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explaination':RunnableSequence(prompt2,model,parser)
})

final_chain = RunnableSequence(joke_chain,parellel_chain)

result = final_chain.invoke({'topic':'lional messi'})

print(result)
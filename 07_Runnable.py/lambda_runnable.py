from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

def word_count(text):
    return len(text.split())

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task = 'text-generation'
)

model = ChatHuggingFace(llm= llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template= 'write a joke on {topic}',
    input_variables= ['topic']
)

joke_gen = RunnableSequence(prompt, model , parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen, parallel_chain)

result = final_chain.invoke({'topic':'AI'})

print(result)
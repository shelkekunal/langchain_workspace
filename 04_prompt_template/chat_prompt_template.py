from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system','You are a helpfull {domain} expert'),
    ('human', 'tell me about {topic} in simple terms')
])

prompt = chat_template.invoke({'domain':'machine learning','topic':'Regression'})

print(prompt)

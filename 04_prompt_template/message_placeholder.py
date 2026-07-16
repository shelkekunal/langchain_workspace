from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system','you have good knowlege about moives'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])


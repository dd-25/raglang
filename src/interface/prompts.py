from langchain_core.prompts import ChatPromptTemplate

escalation_prompt = ChatPromptTemplate.from_template("""
You are a responsible AI agent deciding whether a user's query should be escalated to a human.

Query: {query}

Decide if this query is sensitive or needs human intervention (e.g. related to mental health, violence, legal advice, medical emergencies). 
Respond ONLY with `True` if it should be escalated, or `False` otherwise.
""")

rag_custom_prompt = ChatPromptTemplate.from_template("""You are a yoga expert. Given the user query: {query}, give proper response or advice to the user on whatsapp.""")
from src.interface.state import ChatState
from src.config.config import Config
from src.interface.prompts import rag_custom_prompt
from src.db.pinecone import PineconeDB

class RAG:
    def __init__(self):
        self.config = Config()
        self.llm = self.config.getLLM()
        self.pc = PineconeDB()

    def respond(self, state: ChatState) -> ChatState:
        rag_chain = rag_custom_prompt | self.llm
        context_parts = []
        temp = self.semanticSearch(state["query"])
        for i, doc in enumerate(temp):
            context_parts.append(f"[Source {i+1}: {doc.get('source', 'unknown')}]:\n{doc['text']}")
        context = "\n\n".join(context_parts)
        state["context"] = context
        print(state["context"])
        llm_response = rag_chain.invoke({"context": state["context"], "query": state["query"]})
        return {**state, "result": llm_response.content.strip()}
    
    def semanticSearch(self, query: str):
        results = self.pc.semanticSearch(query, top_k=5)
        print(f"Search results: {results}")
        return results
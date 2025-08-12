from src.interface.state import ChatState
from src.modules.rag import RAG

class AgentRouter:
    def __init__(self):
        self.rag_instance = RAG()
    
    def route(self, state: ChatState) -> ChatState:
        """
        Route the query and perform semantic search to populate context.
        """
        query = state["query"]
        
        # Perform semantic search to get relevant documents
        search_results = self.rag_instance.semanticSearch(query)
        
        # Populate the context with search results
        state["context"] = search_results
        
        return {**state, "context": search_results}
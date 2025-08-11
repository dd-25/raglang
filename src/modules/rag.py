from src.interface.state import ChatState
from src.config.config import Config
from src.interface.prompts import rag_custom_prompt

class RAG:
    def __init__(self):
        self.config = Config()
        self.llm = self.config.getLLM()

    def respond(self, state: ChatState) -> ChatState:
        rag_chain = rag_custom_prompt | self.llm
        llm_response = rag_chain.invoke({"query": state["query"]})
        return {**state, "result": llm_response.content.strip()}
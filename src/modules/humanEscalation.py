from src.interface.state import ChatState
from langchain_core.tools import tool

class Human:
    @staticmethod
    def human_node(state: ChatState) -> ChatState:
        return {**state, "result": "This issue has been escalated to a human."}
    
    @staticmethod
    @tool
    def humanescalation() -> dict:
        """Escalates to human if query is not suitable for AI."""
        return {"message": "Escalating to human support."}
from src.interface.state import ChatState
from langchain_core.tools import tool

class Helper:
    @staticmethod
    def should_escalate(state: ChatState):
        return "humanescalation" if state["escalate"] else "agentrouter"
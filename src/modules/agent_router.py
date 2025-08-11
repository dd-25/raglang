from src.interface.state import ChatState

class AgentRouter:
    @staticmethod
    def route(state: ChatState) -> ChatState:
        return {**state, "result": f"Agent router handling: {state['query']}"}
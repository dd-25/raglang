from src.interface.state import ChatState
from src.interface.prompts import escalation_prompt
from src.config.config import Config

class Checks:
    def __init__(self):
        self.config = Config()
        self.llm = self.config.getLLM()

    def checkSentiment(self, state: ChatState) -> ChatState:
        escalation_chain = escalation_prompt | self.llm | (lambda output: output.content.strip().lower() == "true")
        escalate = escalation_chain.invoke({"query": state["query"]})
        return {**state, "escalate": escalate}
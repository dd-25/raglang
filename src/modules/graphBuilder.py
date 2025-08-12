from langgraph.graph import StateGraph, END, START
from src.interface.state import ChatState
from src.modules.agentRouter import AgentRouter
from src.modules.checks import Checks
from src.modules.humanEscalation import Human
from src.modules.rag import RAG
from src.utils.helper import Helper

class BuildGraph:
    def __init__(self, state=ChatState):
        pass
    
    def buildRAG(self, state=ChatState):
        agentRouter = AgentRouter()
        checks = Checks()
        human = Human()
        rag = RAG()
        helper = Helper()
        builder = StateGraph(state)
        
        # Nodes
        builder.add_node("checks", checks.checkSentiment)
        builder.add_node("agentrouter", agentRouter.route)
        builder.add_node("rag", rag.respond)
        builder.add_node("humanescalation", human.human_node)
        
        # Edges
        builder.add_edge(START, "checks")
        builder.add_conditional_edges("checks", helper.should_escalate)
        builder.add_edge("agentrouter", "rag")
        builder.add_edge("rag", END)
        builder.add_edge("humanescalation", END)
        
        return builder.compile()
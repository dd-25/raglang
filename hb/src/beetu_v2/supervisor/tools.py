"""
Tools module (previously agent_wrappers) for Supervisor System

This file exposes the actual @tool-decorated callables used by the
supervisor. It combines the lightweight wrapper logic and the LangChain
tool decorators so `workflow.py` can simply import the tools.
"""
from typing import Dict, Any, List, Optional
import ast
import operator
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool


@tool
def yoga_wellness_agent_tool(query: str) -> str:
    """
    Expert knowledge base agent for health, wellness, yoga, and medical information.

    Use this tool for any queries related to:
    • Health conditions, symptoms, and medical information (menopause, hormones, etc.)
    • Yoga poses, sequences, hand mudras, breathing techniques
    • Meditation, ayurveda, wellness practices, and spiritual health
    • Holistic health, mind-body connection, and therapeutic approaches
    • Women's health, lifecycle changes, and hormonal health
    • Nutrition, lifestyle, and preventive health measures

    This tool accesses a comprehensive knowledge base with detailed health and wellness information.
    
    Args:
        query: User's question about health, wellness, yoga, or medical topics
        
    Returns:
        Detailed response based on expert knowledge base content
    """
    try:
        # Import optimized RAG agent
        from beetu_v2.agents.ragagent.rag_agent import rag_agent_instance

        result = rag_agent_instance.process_query(
            query=query,
            user_details=None,  # Could be passed from supervisor context
            conversation_history=[],
            namespace="default"
        )

        if result.success and result.response:
            return result.response
        else:
            # Strict knowledge-base-only response
            return f"I don't have information about '{query}' in my knowledge base. I can only provide answers based on the health, wellness, and yoga documents that have been uploaded. Please try asking about topics that might be covered in the existing knowledge base or ensure relevant documents have been added."

    except Exception as e:
        # Strict error response - no general knowledge
        return f"I'm unable to access my knowledge base to answer your question about '{query}'. I can only provide information from uploaded documents. Please ensure the knowledge base is accessible and contains relevant content for your query."


@tool
def general_knowledge_agent_tool(query: str) -> str:
    """
    Comprehensive general knowledge, friendly and supportive conversation and educational content provider.
    
    Use this tool for any queries requiring general knowledge, explanations,
    educational content, normal conversations or information outside specialized domains.
    
    Args:
        query: User's question requiring general knowledge or educational content
        
    Returns:
        Informative response with accurate general knowledge and explanations
    """
    try:
        # Import settings locally to avoid circular imports at module load
        from beetu_v2.supervisor.constants import SUPERVISOR_SETTINGS

        model = init_chat_model(
            model=SUPERVISOR_SETTINGS.MODEL,
            api_key=SUPERVISOR_SETTINGS.API_KEY,
            temperature=0.3,
        )

        response = model.invoke([
            {"role": "system", "content": "You are a helpful general knowledge assistant."},
            {"role": "user", "content": query},
        ])

        return getattr(response, "content", str(response))

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"General knowledge agent error: {str(e)}")
        return f"I encountered an error while processing your question: {str(e)}"

# For Testing Purpose Only
@tool
def link_agent_tool(
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    user_details: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Calendar / Link Agent - fetches and summarizes scheduled sessions,
    meetings, classes, or any events in the organisation's calendars.

    Use this tool when the user asks for:
    • Upcoming events/sessions (e.g. “next yoga class”, “upcoming 2 meetings”)
    • Past events by name (e.g. “send me the link for last week's demo day”)
    • Event links such as Zoom, Google-Meet, YouTube recordings, or documents
    • General scheduling questions within ±30 days

    The agent understands natural-language constraints like:
    - “upcoming event”, “upcoming 5 events”, “all upcoming events”
    - “past session 'Demo Day'”, “previous marketing meeting”

    Args:
        query: Natural-language scheduling request from the user

    Returns:
        A helpful English answer containing only the relevant events / links,
        or a graceful fallback if nothing is found.
    """
    try:
        # Import inside function to avoid circular-import issues
        from beetu_v2.agents.linkagent.calendar_agent import link_agent_react

        # Use supervisor-supplied context if provided, otherwise fall back to
        # sensible defaults so existing calls without the extra arguments
        # continue to work.
        if conversation_history is None:
            conversation_history = []
        if user_details is None:
            user_details = {}

        response = link_agent_react(
            query=query,
            conversation_history=conversation_history,
            user_details=user_details,
        )

        return response or (
            "I checked the calendar but couldn't find enough information "
            "to answer your request precisely. Please verify the event name "
            "or timeframe and try again."
        )

    except Exception as e:
        # Global fallback in case of unexpected errors (e.g., no calendar access)
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Link agent tool error: {str(e)}")
        return (
            f"I'm sorry, I encountered an error while retrieving calendar information: {str(e)}. "
            "Please try again later or specify your request differently."
        )

@tool
def math_agent_tool(expression: str) -> str:
    """
    Use this tool for any queries involving numbers, calculations, equations,
    mathematical analysis, or quantitative problem solving.
    
    Args:
        expression: User's mathematical question or calculation request
        
    Returns:
        Detailed mathematical solution with step-by-step explanations and reasoning
    """
    try:
        # Only allow basic mathematical characters to reduce risk
        allowed_chars = set('0123456789+-*/().,= ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic math operations allowed (+, -, *, /, parentheses)"

        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }

        def eval_expr(node):
            if isinstance(node, ast.Constant):
                # Python 3.8+ uses Constant for numbers
                return node.value
            if isinstance(node, ast.Num):
                return node.n
            if isinstance(node, ast.BinOp):
                left = eval_expr(node.left)
                right = eval_expr(node.right)
                op_func = ops.get(type(node.op))
                if op_func is None:
                    raise TypeError(f"Unsupported operator: {type(node.op)}")
                return op_func(left, right)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return eval_expr(node.operand)
            raise TypeError(f"Unsupported expression node: {type(node)}")

        parsed = ast.parse(expression, mode='eval')
        result = eval_expr(parsed.body)
        return f"The result is: {result}"

    except (ValueError, TypeError, KeyError, SyntaxError) as e:
        return f"Error calculating expression: {str(e)}"


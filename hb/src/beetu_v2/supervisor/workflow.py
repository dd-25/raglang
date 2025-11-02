"""
LangGraph Supervisor Workflow for Multi-Agent System

This module defines a simplified supervisor system that orchestrates
multiple specialized agents using LangGraph's @tool decorator and supervisor API.
"""

import logging
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
# TODO: Add persistence layer for user details and conversation history later

from beetu_v2.supervisor.prompts import (
    SUPERVISOR_ROUTING_PROMPT,
    SUPERVISOR_FINAL_RESPONSE_PROMPT,
    SUPERVISOR_CONTINUATION_PROMPT,
    YOGA_WELLNESS_AGENT_PROMPT,
    MATH_AGENT_PROMPT,
    GENERAL_AGENT_PROMPT,
    AGENT_TOOL_DESCRIPTIONS
)
from beetu_v2.supervisor.tools import (
    yoga_wellness_agent_tool,
    general_knowledge_agent_tool,
    math_agent_tool,
    link_agent_tool,
)

try:
    from .constants import SUPERVISOR_SETTINGS
except ImportError:
    class SUPERVISOR_SETTINGS:
        MODEL = "gpt-4o-mini"
        API_KEY = ""
        TEMPERATURE = 0.3

# tools are provided by src.beetu_v2.supervisor.tools


def create_supervisor_system(model, max_iterations=3):
    """
    Create a native LangGraph supervisor using @tool decorators and built-in capabilities.
    
    This supervisor uses LangGraph's native tool selection based on docstrings,
    providing intelligent routing and multi-agent coordination.
    """
    
    # Collect all agent tools
    agent_tools = [
        yoga_wellness_agent_tool,
        math_agent_tool,
        general_knowledge_agent_tool,
        link_agent_tool
    ]
    
    # Create supervisor prompt for tool-based routing
    supervisor_prompt = f"""
{SUPERVISOR_ROUTING_PROMPT}

You are a supervisor with access to specialized agent tools. Each tool has detailed docstrings 
describing their capabilities. Use the tools intelligently based on the user's query.

Guidelines for tool usage:
1. Read tool docstrings carefully to understand each agent's expertise
2. Choose the most appropriate tool for the primary aspect of the user's query
3. You can use multiple tools if the query requires different types of expertise
4. Maximum {max_iterations} tool calls per user query
5. Provide comprehensive final responses that integrate all tool outputs

Available tools will be automatically described by their docstrings.
"""
    
    # Create the supervisor agent using create_react_agent with tools
    supervisor_agent = create_react_agent(
        model=model,
        tools=agent_tools,
        prompt=supervisor_prompt
    )
    
    return supervisor_agent


logger = logging.getLogger(__name__)

load_dotenv()


class SupervisorSystem:
    """Native LangGraph supervisor system using @tool decorators for intelligent agent routing."""
    
    def __init__(self, model_name: str = None, max_iterations: int = 3):
        """Initialize supervisor system with native LangGraph approach."""
        self.model_name = model_name or SUPERVISOR_SETTINGS.MODEL
        self.model = init_chat_model(
            model=self.model_name,
            api_key=SUPERVISOR_SETTINGS.API_KEY,
            temperature=0.3
        )
        self.max_iterations = max_iterations
        self.supervisor_agent = create_supervisor_system(self.model, self.max_iterations)
        self.query_count = 0
        
    async def process_query(
        self,
        query: str,
        user_id: str | None = None,
    ) -> dict:
        """Process user query through the supervisor system asynchronously."""
        import time
        start_time = time.time()
        
        # ------------------------------------------------------------------ #
        # Placeholder variables for future implementation                    #
        # ------------------------------------------------------------------ #
        user_id = user_id or "default_user"
        
        # TODO: Implement persistence layer later
        conversation_history = []  # Placeholder - will be loaded from persistence
        user_details = {}          # Placeholder - will be loaded from persistence

        try:
            self.query_count += 1

            # Create initial message
            initial_message = {"role": "user", "content": query}

            # Build full message list: past history then new user message
            messages = conversation_history + [initial_message]

            # Always use async for better performance
            result = await self.supervisor_agent.ainvoke({"messages": messages})

            # Calculate processing time
            processing_time = time.time() - start_time

            # Extract final response
            if "messages" in result and result["messages"]:
                response_content = result["messages"][-1].content

                # ---------------------------------------------------------- #
                # TODO: Save conversation history to persistence later      #
                # ---------------------------------------------------------- #
                updated_history = messages + [{"role": "assistant", "content": response_content}]
                # TODO: persistence.save_conversation_history(user_id, updated_history)

                return {
                    "response": response_content,
                    "processing_time": processing_time,
                    "success": True,
                    "query_count": self.query_count
                }

            logger.warning("No response generated")
            processing_time = time.time() - start_time
            return {
                "response": "No response generated",
                "processing_time": processing_time,
                "success": False,
                "query_count": self.query_count
            }

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing query: {str(e)}"
            logger.error("Query processing failed in %.3f seconds: %s", processing_time, str(e))
            return {
                "response": error_msg,
                "processing_time": processing_time,
                "success": False,
                "query_count": self.query_count,
                "error": str(e)
            }
# Beetu-v2.0

**Beetu-v2.0** is an intelligent agentic chatbot system built with multi-agent architecture for specialized task handling.  
Features a **supervisor-coordinated agent ecosystem** with RAG capabilities, subscription management, calendar integration, and extensible agent framework.

---

## Overview
- **Multi-Agent System**: Specialized agents for different domains (wellness, math, general knowledge, subscriptions, calendar)
- **Intelligent Routing**: Supervisor system that routes queries to appropriate agents
- **RAG Integration**: Knowledge base querying with Pinecone vector storage
- **Subscription Management**: Complete subscription operations (pause, resume, cancel, status)
- **Calendar Integration**: Google Calendar API integration for scheduling
- **Extensible Architecture**: Easy to add new agents and capabilities

---

## Getting Started
```bash
# Clone the repository
git clone https://github.com/habuildserver/beetu-v2.0.git
cd beetu-v2.0

# Install dependencies
poetry install

# Run the development server
poetry run dev

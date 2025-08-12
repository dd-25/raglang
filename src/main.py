"""
Main entry point for the LangGraph RAG application.
Production-grade structure with proper path configuration.
"""

from src.modules.graphBuilder import BuildGraph

def main():
    try:
        graph = BuildGraph().buildRAG()
        input_state = {"query": "tell me benefits of yoga"}
        final_state = graph.invoke(input_state)
        print(f"Result: {final_state.get('result')}")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
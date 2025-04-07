# 2. Retrieval agent
from utils.state import AgentState
from utils.retriever import get_retriever
from copy import deepcopy

def retriever_agent(state: AgentState) -> AgentState:
    """Retrieve relevant documents from the vector database"""
    # Create a deep copy of the state to avoid modifying references directly
    state_copy = deepcopy(state)
    
    query = state_copy.get("analyzed_query") or state_copy.get("query", "")
    
    if not query:
        state_copy["retrieved_docs"] = []
        state_copy["intermediate_steps"].append("Error: No query found")
        return state_copy
    
    # Get retriever
    retriever = get_retriever()
    
    # Retrieve documents - using the new invocation method
    try:
        # Attempt using the new recommended invoke method
        retrieved_docs = retriever.invoke(query)
        
        # Debugging information
        print(f"Number of retrieved documents: {len(retrieved_docs)}")
        if retrieved_docs:
            print(f"First document summary: {retrieved_docs[0].page_content[:100]}...")
    except (AttributeError, TypeError):
        # If it fails, fall back to the old method
        retrieved_docs = retriever.get_relevant_documents(query)
        
        # Debugging information
        print(f"Number of documents retrieved using the old method: {len(retrieved_docs)}")
        if retrieved_docs:
            print(f"First document summary: {retrieved_docs[0].page_content[:100]}...")
    
    # Update state
    state_copy["retrieved_docs"] = retrieved_docs
    state_copy["intermediate_steps"].append(f"Retrieved {len(retrieved_docs)} documents")
    
    return state_copy
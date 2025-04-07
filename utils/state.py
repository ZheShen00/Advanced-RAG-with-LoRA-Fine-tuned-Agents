# utils/state.py
from typing import Dict, List, Any, Optional, TypedDict
from langchain.schema import Document

class AgentState(TypedDict, total=False):
    query: str                        # User query
    analyzed_query: Optional[str]     # Analyzed query
    retrieved_docs: Optional[List[Document]]  # Retrieved documents
    cleaned_docs: Optional[List[Document]]    # Cleaned documents
    relevant_docs: Optional[List[Document]]   # Filtered relevant documents
    answer: Optional[str]             # Final answer
    intermediate_steps: List[str]     # Intermediate steps log
    confidence_score: Optional[float] # Confidence score
    reformulation_count: int          # Query reformulation counter

def initialize_state(query: str) -> AgentState:
    """Initialize state with a query"""
    return {
        "query": query,
        "analyzed_query": None,
        "retrieved_docs": None,
        "cleaned_docs": None,
        "relevant_docs": None,
        "answer": None,
        "intermediate_steps": [],
        "confidence_score": None,
        "reformulation_count": 0
    }
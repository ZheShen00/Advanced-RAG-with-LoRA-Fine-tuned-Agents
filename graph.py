# graph.py
import tempfile
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Annotated
from langchain.schema import Document

from utils.state import AgentState
from utils.decision_functions import should_clean_docs, assess_confidence
from agents import (
    query_analyzer, 
    retriever_agent, 
    document_cleaner, 
    relevance_evaluator,
    answer_generator, 
    retriever_reformulator
)

def build_rag_graph():
    """Build multi-agent RAG system graph"""
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("query_analyzer", query_analyzer)
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("document_cleaner", document_cleaner)
    workflow.add_node("relevance_evaluator", relevance_evaluator)
    workflow.add_node("answer_generator", answer_generator)
    workflow.add_node("retriever_reformulator", retriever_reformulator)
    
    # Add edges
    workflow.add_edge("query_analyzer", "retriever")
    
    # Conditional edge: Determine if document cleaning is needed
    workflow.add_conditional_edges(
        "retriever",
        should_clean_docs,
        {
            "clean": "document_cleaner",
            "skip_cleaning": "relevance_evaluator"
        }
    )
    
    workflow.add_edge("document_cleaner", "relevance_evaluator")
    
    # Conditional edge: Determine path based on confidence assessment
    workflow.add_conditional_edges(
        "relevance_evaluator",
        assess_confidence,
        {
            "generate_answer": "answer_generator",
            "try_reformulate": "retriever_reformulator"
        }
    )
    
    # 为避免循环和状态冲突，重新定义从retriever_reformulator到retriever的路径
    # 使用明确的条件分支而不是基于状态的lambda函数
    workflow.add_conditional_edges(
        "retriever_reformulator",
        lambda state: "skip_retrieval" if state.get("reformulation_count", 0) >= 2 else "do_retrieval",
        {
            "skip_retrieval": "answer_generator",
            "do_retrieval": "retriever"
        }
    )
    
    workflow.add_edge("answer_generator", END)
    
    # Set entry point
    workflow.set_entry_point("query_analyzer")
    
    # Compile graph with proper config to avoid recursion issues
    config = {
        "recursion_limit": 10,  # 减小递归限制，避免过深的递归
    }
    
    rag_chain = workflow.compile()
    
    return workflow, rag_chain

def visualize_rag_graph(workflow):
    """Visualize RAG system graph using Mermaid compatible output"""
    try:
        # Generate Mermaid-based visualization
        mermaid_code = """
graph TD
    start([Start]) --> query_analyzer
    query_analyzer[Query Analysis] --> retriever[Document Retrieval]
    retriever -->|Needs Cleaning| document_cleaner[Document Cleaning]
    retriever -->|Skip Cleaning| relevance_evaluator[Relevance Evaluation]
    document_cleaner --> relevance_evaluator
    relevance_evaluator -->|High Confidence| answer_generator[Answer Generation]
    relevance_evaluator -->|Low Confidence| retriever_reformulator[Query Reformulation]
    retriever_reformulator -->|Reformulation Attempts < 2| retriever
    retriever_reformulator -->|Reformulation Attempts >= 2| answer_generator
    answer_generator --> finish([End])
        """
        
        # Print Mermaid code for frontend rendering
        print("RAG System Workflow View (Mermaid Code):")
        print(mermaid_code)
        
        # Create a temporary file to save the Mermaid graph
        with tempfile.NamedTemporaryFile(delete=False, suffix='.md') as f:
            f.write(mermaid_code.encode())
            mermaid_file = f.name
            
        print(f"Mermaid graph saved at: {mermaid_file}")
        
    except Exception as e:
        print(f"Error occurred during visualization: {e}")
        print("\nRAG system includes the following components and workflow:")
        print("- Query Analyzer: Analyzes user queries")
        print("- Document Retriever: Retrieves documents from a vector database")
        print("- Document Cleaner: Cleans retrieved documents")
        print("- Relevance Evaluator: Evaluates document relevance")
        print("- Answer Generator: Generates the final answer")
        print("- Query Reformulator: Reformulates queries to improve results")
# utils/decision_functions.py
from utils.state import AgentState

def should_clean_docs(state: AgentState) -> str:
    """Decide whether document cleaning is necessary"""
    if state["retrieved_docs"] and len(state["retrieved_docs"]) > 0:
        total_length = sum(len(doc.page_content) for doc in state["retrieved_docs"])
        if total_length > 10000:  # Assume cleaning is needed if the total length exceeds 10,000 characters
            return "clean"
        else:
            return "skip_cleaning"
    else:
        return "skip_cleaning"

def assess_confidence(state: AgentState) -> str:
    """Decide whether additional processing is needed based on confidence score and reformulation attempts"""
    confidence_score = state.get("confidence_score", 0)  # Add default values using get
    reformulation_count = state.get("reformulation_count", 0)
    
    print(f"Current confidence score: {confidence_score}, Reformulation attempts: {reformulation_count}")
    
    # Clearly defined boundary conditions
    if confidence_score is None:
        confidence_score = 0
    
    # More explicit conditional judgments
    if confidence_score >= 5:
        print("Decision: Generate answer - confidence is sufficient")
        return "generate_answer"
    elif reformulation_count >= 2:
        print("Decision: Generate answer - reformulation limit reached")
        return "generate_answer"
    elif reformulation_count > 0 and (not state.get("relevant_docs") or len(state.get("relevant_docs", [])) == 0):
        print("Decision: Generate answer - no relevant docs after reformulation")
        return "generate_answer"
    else:
        print("Decision: Reformulate query - confidence too low and reformulation attempts available")
        return "try_reformulate"
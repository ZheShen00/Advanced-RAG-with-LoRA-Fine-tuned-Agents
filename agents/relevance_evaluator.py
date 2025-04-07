# 4. Relevance Evaluation Agent
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from utils.state import AgentState

def relevance_evaluator(state: AgentState) -> AgentState:
    """Evaluate document relevance and assign a score to each document"""
    query = state["query"]
    docs = state["cleaned_docs"] or state["retrieved_docs"]
    
    if not docs:
        state["relevant_docs"] = []
        state["intermediate_steps"].append("No relevant documents found")
        return state
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Define prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are a document relevance evaluation expert. Your task is to assess the relevance of the following documents to the given query.
        
        Query: {query}
        
        Document list:
        {docs_content}
        
        Please assign a score (1-10) to each document, where 1 means completely irrelevant and 10 means highly relevant.
        
        Return the evaluation result in JSON format, including document index, relevance score, and whether to retain the document (retain if score ≥ 6).
        
        JSON format:
        {{
            "evaluation": [
                {{
                    "document_index": 0,
                    "relevance_score": 8,
                    "retain": true
                }},
                ...
            ],
            "retained_document_indices": [0, ...]
        }}"""
    )
    
    # Prepare document string
    docs_content = "\n\n---Document Separator---\n\n".join([
        f"Document {i}:\n{doc.page_content}" 
        for i, doc in enumerate(docs)
    ])
    
    # Generate evaluation result
    evaluation_result_text = llm.invoke(prompt.format(
        query=query, 
        docs_content=docs_content
    )).content
    
    print(f"Raw evaluation result text: {evaluation_result_text}")
    
    # Parse evaluation result (JSON parsing should be used in actual implementation)
    try:
        evaluation_result = json.loads(evaluation_result_text.replace("'", '"'))
        relevant_indices = evaluation_result.get("retained_document_indices", [])
        
        # Filter relevant documents
        relevant_docs = [docs[i] for i in relevant_indices if i < len(docs)]
        
        # Compute average relevance score as confidence score
        scores = [item.get("relevance_score", 0) for item in evaluation_result.get("evaluation", [])]
        confidence_score = sum(scores) / len(scores) if scores else 0
        
        print(f"Relevant document indices: {relevant_indices}")
        print(f"Confidence score: {confidence_score}")
    except Exception as e:
        print(f"Error parsing evaluation result: {e}")
        # If parsing fails, retain all documents
        relevant_docs = docs
        confidence_score = 5.0  # Default to medium confidence
    
    # Update state
    state["relevant_docs"] = relevant_docs
    state["confidence_score"] = confidence_score
    state["intermediate_steps"].append(f"Evaluation completed, retained {len(relevant_docs)} relevant documents")
    
    return state
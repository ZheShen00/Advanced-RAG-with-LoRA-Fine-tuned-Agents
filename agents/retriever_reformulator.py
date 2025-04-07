from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from utils.state import AgentState
from copy import deepcopy

def retriever_reformulator(state: AgentState) -> AgentState:
    """Reconstruct the retrieval query to obtain better results"""
    # Create a deep copy of the state to avoid modifying references directly
    state_copy = deepcopy(state)
    
    query = state_copy["query"]
    docs = state_copy["relevant_docs"] or []
    
    reformulation_count = state_copy.get("reformulation_count", 0) + 1
    state_copy["reformulation_count"] = reformulation_count
    
    state_copy["intermediate_steps"].append(f"Starting reformulation attempt {reformulation_count}")
    
    # Check if the maximum number of reconfigurations has been reached
    if reformulation_count >= 2:
        state_copy["intermediate_steps"].append(f"Maximum reformulation attempts reached")
        return state_copy
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    prompt = ChatPromptTemplate.from_template(
        """You are an advanced query reformulation expert. The following query has been initially retrieved, but the results are not ideal. Please help reformulate the query to obtain more relevant results.
        
        Original Query: {query}
        
        Retrieved document summaries:
        {docs_summary}
        
        This is reformulation attempt {reformulation_count}. Based on the original query and existing information, generate a query variation significantly different from previous attempts, ensuring the use of new keywords and perspectives.
        
        Return a final query string that incorporates these variations while remaining relevant to the original question but improving retrieval effectiveness."""
    )
    
    docs_summary = "No sufficiently relevant documents found." if not docs else "\n".join([
        f"{i+1}. " + (doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content)
        for i, doc in enumerate(docs[:3])
    ])
    
    reformulated_query = llm.invoke(prompt.format(
        query=query, 
        docs_summary=docs_summary,
        reformulation_count=reformulation_count
    )).content
    
    # Ensure that only the analyzed query is updated and not the original query
    state_copy["analyzed_query"] = reformulated_query
    state_copy["intermediate_steps"].append(f"Reformulated query: {reformulated_query}")
    
    # Clearing previous search results to avoid status confusion
    state_copy["retrieved_docs"] = None
    state_copy["cleaned_docs"] = None
    state_copy["relevant_docs"] = None
    
    return state_copy
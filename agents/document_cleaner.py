# 3. Document cleaning agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from utils.state import AgentState

def document_cleaner(state: AgentState) -> AgentState:
    """Clean retrieved documents by removing noise and extracting the most relevant content"""
    query = state["query"]
    docs = state["retrieved_docs"]
    
    if not docs:
        state["cleaned_docs"] = []
        state["intermediate_steps"].append("No documents found to clean")
        return state
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    cleaned_docs = []
    for i, doc in enumerate(docs):
        # Define prompt template
        prompt = ChatPromptTemplate.from_template(
            """You are a professional document cleaning expert. Your task is to clean and extract relevant information from the retrieved documents.
            
            Query: {query}
            
            Document Content:
            {doc_content}
            
            Please perform the following tasks:
            1. Remove content unrelated to the query
            2. Eliminate redundant information
            3. Extract the most relevant facts and data
            4. Maintain sentence integrity
            
            Return the cleaned document content, ensuring that all important information related to the query is retained."""
        )
        
        # Clean document
        cleaned_content = llm.invoke(prompt.format(
            query=query, 
            doc_content=doc.page_content
        )).content
        
        # Create new document object
        cleaned_doc = Document(
            page_content=cleaned_content,
            metadata=doc.metadata
        )
        
        cleaned_docs.append(cleaned_doc)
    
    # Update state
    state["cleaned_docs"] = cleaned_docs
    state["intermediate_steps"].append(f"Cleaned {len(cleaned_docs)} documents")
    
    return state
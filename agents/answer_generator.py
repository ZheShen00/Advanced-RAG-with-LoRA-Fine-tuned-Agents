from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from utils.state import AgentState

# 5. Answer generation agent
def answer_generator(state: AgentState) -> AgentState:
    """Generate the final answer based on relevant documents"""
    query = state["query"]
    docs = state["relevant_docs"] or state["cleaned_docs"] or state["retrieved_docs"]
    confidence_score = state["confidence_score"] or 0
    reformulation_count = state.get("reformulation_count", 0)
    
    if (not docs or len(docs) == 0) and reformulation_count > 0:
        state["answer"] = (
            f"Sorry, I attempted multiple queries ({reformulation_count} attempts), but could not find relevant information for your question."
            " This may be because: 1) The database does not contain this information; 2) Your question needs more specific details;"
            " 3) It relates to information after September 2021. Please try rephrasing your question or providing more details."
        )
        return state
    
    if not docs:
        state["answer"] = "Sorry, I could not find information related to your question. Please try asking in a different way or provide more details."
        return state
    
    print("Documents used for answer generation:")
    for i, doc in enumerate(docs):
        print(f"Document {i+1} summary: {doc.page_content[:150]}...")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    docs_content = "\n\n".join([
        f"Source {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(docs)
    ])
    
    confidence_prompt = ""
    if confidence_score < 5:
        confidence_prompt = "Please note that the information I found may have low relevance to your question."
    elif confidence_score >= 8:
        confidence_prompt = "I found highly relevant information for your question."
    
    prompt = ChatPromptTemplate.from_template(
       """You are a professional environmental news analysis assistant. Based on the provided document content, answer the user's question.
       
       Question: {query}
       
       {confidence_prompt}
       
       References:
       {docs_content}
       Please provide a comprehensive and accurate response that meets the following requirements:
       1. Directly answer the user's question.
       2. Strictly base your response on the provided references; do not add your own knowledge.
       3. If the references do not contain sufficient information, honestly state that you cannot find the answer from the given sources.
       4. The response should be well-structured and easy to understand.
       5. Cite specific data and facts, clearly indicating the source of information.
       6. Avoid oversimplifying complex issues.
       7. Do not fabricate or speculate on information that is not explicitly mentioned in the references.
       
       Response:"""
    )
    
    answer = llm.invoke(prompt.format(
        query=query, 
        confidence_prompt=confidence_prompt,
        docs_content=docs_content
    )).content
    
    state["answer"] = answer
    
    return state
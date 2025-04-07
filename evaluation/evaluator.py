# evaluation/evaluator.py
import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from utils.state import initialize_state
from graph import build_rag_graph
from utils.retriever import get_retriever
import time

# Define test questions directly in evaluator.py
TEST_QUESTIONS = [
    "What are the environmental policy challenges for the UK government after October 2021?",
    "How have recent European Union regulations affected biodiversity conservation since 2022?",
    "What are the latest global responses to deforestation in the Amazon rainforest post-2021?",
    "How has the transition to renewable energy progressed in China since 2022?",
    "What are the worst hurricanes and extreme weather events in North America since 2022?",
    "Who represented his/her country to receive the 2021 winner of the earthshot protect and restore nature award?",
    "What are the latest developments in carbon capture and storage (CCS) technologies after 2021?",
    "How has the EU's Green Deal evolved after 2021, and what new initiatives have been introduced?",
]

def evaluate_all_systems(output_file="evaluation_results.json"):
    """Evaluating the performance of four systems"""
    print("Starting system evaluation...")
    
    results = {
        "base_llm": {},
        "simple_rag": {},
        "advanced_rag_base": {},
        "advanced_rag_finetuned": {}
    }
    
    # 1. Basic LLM (no RAG)
    print("Evaluating base LLM (no RAG)...")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    for q in TEST_QUESTIONS:
        print(f"Processing question: {q}")
        start_time = time.time()
        results["base_llm"][q] = {
            "answer": llm.invoke(q).content,
            "time_taken": time.time() - start_time
        }
    
    # 2. Simple RAG (Search + LLM only)
    print("Evaluating simple RAG...")
    retriever = get_retriever()
    for q in TEST_QUESTIONS:
        print(f"Processing question: {q}")
        start_time = time.time()
        docs = retriever.get_relevant_documents(q)
        docs_content = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = ChatPromptTemplate.from_template(
            """Based on the following context information, please answer the user's question.
            
            Question: {query}
            
            Context:
            {docs_content}
            
            Please provide a comprehensive and accurate answer based only on the information in the context.
            """
        )
        
        response = llm.invoke(prompt.format(query=q, docs_content=docs_content))
        results["simple_rag"][q] = {
            "answer": response.content,
            "time_taken": time.time() - start_time
        }
    
    # 3. Advanced RAG (no fine tuning)
    print("Evaluating advanced RAG without fine-tuning...")
    # Here set the environment variable to disable the LoRA model
    os.environ["DISABLE_LORA"] = "true"
    _, rag_chain = build_rag_graph()
    for q in TEST_QUESTIONS:
        print(f"Processing question: {q}")
        start_time = time.time()
        state = initialize_state(q)
        result = rag_chain.invoke(state)
        results["advanced_rag_base"][q] = {
            "answer": result["answer"],
            "time_taken": time.time() - start_time,
            "steps": result["intermediate_steps"]
        }
    
    # 4. Advanced RAG (using fine tuning)
    print("Evaluating advanced RAG with fine-tuning...")
    # Re-enabling the LoRA model
    if "DISABLE_LORA" in os.environ:
        del os.environ["DISABLE_LORA"]
    
    _, rag_chain = build_rag_graph()
    for q in TEST_QUESTIONS:
        print(f"Processing question: {q}")
        start_time = time.time()
        state = initialize_state(q)
        result = rag_chain.invoke(state)
        results["advanced_rag_finetuned"][q] = {
            "answer": result["answer"],
            "time_taken": time.time() - start_time,
            "steps": result["intermediate_steps"]
        }
    
    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Evaluation completed. Results saved to {output_file}")
    return results
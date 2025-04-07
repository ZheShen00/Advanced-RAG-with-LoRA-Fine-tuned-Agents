# agents/query_analyzer.py
from typing import Dict, Any
from utils.state import AgentState
from models.lora_model import LoRAModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os

# Singleton pattern to ensure the model is loaded only once
lora_model = None

def get_lora_model():
    global lora_model
    if lora_model is None:
        lora_model = LoRAModel()
    return lora_model

def query_analyzer(state: AgentState) -> AgentState:
    """Analyze user query to enhance search effectiveness"""
    query = state["query"]
    
    # Check if LoRA model is disabled
    if os.environ.get("DISABLE_LORA") == "true":
        # Use standard LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        prompt = ChatPromptTemplate.from_template(
            """You are a professional query analysis expert. Your task is to analyze and refine user queries to improve search effectiveness.
            
            Original Query: {query}
            
            Please analyze this query considering the following points:
            1. What is the main topic of the query?
            2. Does the query contain a specific time range? (Especially after September 2021)
            3. Does the query involve topics such as environment, climate change, or ecological conservation?
            4. Should additional relevant keywords be added for better search results?
            
            Please provide an enhanced query that helps the retrieval system find the most relevant environmental news articles. The returned query should be a comprehensive search string."""
        )
        
        analyzed_query = llm.invoke(prompt.format(query=query)).content
        state["intermediate_steps"].append("Standard LLM used for query analysis (LoRA disabled)")
    else:
        # Use LoRA fine-tuned model
        model = get_lora_model()
        
        prompt = f"""You are a professional query analysis expert. Your task is to analyze and refine user queries to improve search effectiveness.
        
        Original Query: {query}
        
        Please analyze this query considering the following points:
        1. What is the main topic of the query?
        2. Does the query contain a specific time range? (Especially after September 2021)
        3. Does the query involve topics such as environment, climate change, or ecological conservation?
        4. Should additional relevant keywords be added for better search results?
        
        Please provide an enhanced query that helps the retrieval system find the most relevant environmental news articles. The returned query should be a comprehensive search string."""
        
        # Use LoRA model to generate analysis results
        analyzed_query = model.generate(prompt, max_new_tokens=150)
        state["intermediate_steps"].append("LoRA fine-tuned model used for query analysis")
    
    # Update state
    state["analyzed_query"] = analyzed_query
    state["intermediate_steps"].append(f"Query analysis: Original query refined to: {analyzed_query}")
    
    return state
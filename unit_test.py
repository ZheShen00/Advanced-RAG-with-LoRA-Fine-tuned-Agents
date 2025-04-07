# unit_test.py
import os
from dotenv import load_dotenv
from utils.state import initialize_state
from graph import build_rag_graph
from models.lora_model import LoRAModel
from utils.retriever import get_retriever
from agents.query_analyzer import query_analyzer
from agents.retriever_agent import retriever_agent
from agents.document_cleaner import document_cleaner
from agents.relevance_evaluator import relevance_evaluator
from agents.answer_generator import answer_generator

def run_unit_test():
    """运行简单的单元测试以验证系统功能"""
    print("=== Starting Unit Test ===")
    
    # 加载环境变量
    load_dotenv()
    
    # 检查必要的API密钥
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not found in environment variables")
    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError("Pinecone API key not found in environment variables")
    
    # 测试问题
    test_query = "What were the environmental policy challenges for the UK after 2021?"
    
    print("\n1. Testing LoRA Model Loading...")
    try:
        lora_model = LoRAModel()
        test_prompt = f"Analyze this query: {test_query}"
        result = lora_model.generate(test_prompt, max_new_tokens=50)
        print(f"LoRA model generated: {result[:100]}...")
        print("✅ LoRA model test passed")
    except Exception as e:
        print(f"❌ LoRA model test failed: {e}")
    
    print("\n2. Testing Pinecone Retriever...")
    try:
        retriever = get_retriever()
        docs = retriever.get_relevant_documents(test_query)
        print(f"Retrieved {len(docs)} documents")
        if docs:
            print(f"First document sample: {docs[0].page_content[:100]}...")
        print("✅ Retriever test passed")
    except Exception as e:
        print(f"❌ Retriever test failed: {e}")
    
    print("\n3. Testing Query Analyzer Agent...")
    try:
        state = initialize_state(test_query)
        updated_state = query_analyzer(state)
        print(f"Original query: {state['query']}")
        print(f"Analyzed query: {updated_state['analyzed_query']}")
        print("✅ Query analyzer agent test passed")
    except Exception as e:
        print(f"❌ Query analyzer agent test failed: {e}")
    
    print("\n4. Testing Agent Pipeline...")
    try:
        # 初始化状态并测试各个代理
        state = initialize_state(test_query)
        
        # 运行查询分析代理
        print("Running query analyzer...")
        state = query_analyzer(state)
        
        # 运行检索代理
        print("Running retriever agent...")
        state = retriever_agent(state)
        print(f"Retrieved {len(state['retrieved_docs'])} documents")
        
        # 运行文档清洗代理
        print("Running document cleaner...")
        state = document_cleaner(state)
        print(f"Cleaned {len(state['cleaned_docs'])} documents")
        
        # 运行相关性评估代理
        print("Running relevance evaluator...")
        state = relevance_evaluator(state)
        print(f"Selected {len(state['relevant_docs'])} relevant documents")
        print(f"Confidence score: {state['confidence_score']}")
        
        # 运行答案生成代理
        print("Running answer generator...")
        state = answer_generator(state)
        print(f"Generated answer: {state['answer'][:200]}...")
        
        print("✅ Agent pipeline test passed")
    except Exception as e:
        print(f"❌ Agent pipeline test failed: {e}")
    
    print("\n=== Unit Test Completed ===")

if __name__ == "__main__":
    run_unit_test()
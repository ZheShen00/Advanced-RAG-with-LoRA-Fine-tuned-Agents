# app.py
import os
import argparse
from dotenv import load_dotenv

from utils.state import initialize_state
from graph import build_rag_graph, visualize_rag_graph
from interface import create_gradio_interface
from evaluation.evaluator import evaluate_all_systems

def run_rag_system(query: str):
    """Run the multi-agent RAG system"""
    # Build graph
    _, rag_chain = build_rag_graph()
    
    # Initialize state
    state = initialize_state(query)
    
    # Add configuration to prevent infinite recursion
    config = {
        "recursion_limit": 20,  # Increase recursion limit to ensure sufficient execution
        "interrupt_before": [],  # Optional: Interrupt before certain nodes
        "interrupt_after": []    # Optional: Interrupt after certain nodes
    }
    
    try:
        # Run workflow
        print("Starting workflow execution...")
        result = rag_chain.invoke(state, config=config)
        print("Workflow execution completed")
        return result
    except Exception as e:
        # Handle possible errors
        print(f"Error occurred while running RAG system: {e}")
        
        # Emergency handling: If recursion error occurs but documents are retrieved, attempt answer generation
        if "recursion_limit" in str(e) and (state.get("retrieved_docs") or state.get("cleaned_docs")):
            from agents.answer_generator import answer_generator
            print("Detected recursion error but retrieved documents exist. Attempting emergency answer generation...")
            state = answer_generator(state)  # Directly invoke answer generator
        else:
            # Return result with error message
            state["answer"] = f"An error occurred while processing your query. Please try a more specific or different query.\nError details: {str(e)[:100]}..."
            state["intermediate_steps"].append(f"Error: {str(e)}")
        
        return state

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Advanced RAG System with LoRA Fine-tuned Agent")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on all system configurations")
    parser.add_argument("--visualize", action="store_true", help="Generate and display RAG system graph")
    parser.add_argument("--test", action="store_true", help="Run a test query to verify system functionality")
    args = parser.parse_args()
    
    # 加载环境变量
    load_dotenv()
    
    # Build the graph for visualization only
    print("Building multi-agent RAG system...")
    workflow, _ = build_rag_graph()
    
    # 如果指定了--visualize参数，生成并显示RAG系统图
    if args.visualize:
        print("Generating RAG system visualization graph...")
        visualize_rag_graph(workflow)
    
    # 如果指定了--evaluate参数，运行评估
    if args.evaluate:
        print("Evaluating all system configurations...")
        evaluate_all_systems()
        return
    
    # 如果指定了--test参数，运行测试查询
    if args.test:
        print("\nTesting system...")
        test_query = "Who represented his/her country to receive the 2021 winner of the Earthshot Protect and Restore Nature Award?"
        result = run_rag_system(test_query)
        print(f"Query: {test_query}")
        print(f"Answer: {result['answer']}")
        print("Processing Steps:")
        for step in result["intermediate_steps"]:
            print(f"- {step}")
        return
    
    # 否则，启动Gradio界面
    print("\nLaunching Gradio interface...")
    demo = create_gradio_interface(run_rag_system)
    demo.launch(share=True)

if __name__ == "__main__":
    main()
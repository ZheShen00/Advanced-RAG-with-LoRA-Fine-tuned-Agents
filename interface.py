# interface.py
import gradio as gr
import tempfile
import matplotlib.pyplot as plt
import networkx as nx
import os

def create_gradio_interface(run_system_fn):
    """Create a Gradio interface with visualization functionality"""
    
    # Function to generate an image
    def generate_rag_graph_image():
        """Generate a visualization graph of the RAG system and return the file path"""
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes
        nodes = [
            "query_analyzer", 
            "retriever", 
            "document_cleaner", 
            "relevance_evaluator", 
            "answer_generator", 
            "retriever_reformulator"
        ]
        
        # Map node labels
        node_labels = {
            "query_analyzer": "Query Analyzer",
            "retriever": "Document Retriever",
            "document_cleaner": "Document Cleaner",
            "relevance_evaluator": "Relevance Evaluator",
            "answer_generator": "Answer Generator",
            "retriever_reformulator": "Retriever Reformulator"
        }
        
        # Add edges
        edges = [
            ("query_analyzer", "retriever"),
            ("retriever", "document_cleaner"),
            ("retriever", "relevance_evaluator"),
            ("document_cleaner", "relevance_evaluator"),
            ("relevance_evaluator", "answer_generator"),
            ("relevance_evaluator", "retriever_reformulator"),
            ("retriever_reformulator", "retriever"),
            ("retriever_reformulator", "answer_generator")
        ]
        
        # Add nodes and edges to the graph
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        # Draw the graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)  # Use spring layout
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=2, arrowsize=20)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_weight='bold')
        
        # Set graph boundaries
        plt.axis('off')
        plt.tight_layout()
        
        # Create a temporary file to save the image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
            plt_filename = f.name
        
        # Save the graph to the temporary file
        plt.savefig(plt_filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        return plt_filename
    
    # Create system description text
    def get_system_description():
        return """
        ## RAG System Workflow Description
        
        1. **Query Analyzer**:
           - Analyzes user queries to enhance search effectiveness
           - Identifies topics and keywords in the query
           - Adds relevant keywords to improve search results
        
        2. **Document Retriever**:
           - Retrieves relevant documents from a vector database
           - Uses similarity search to find related content
           - Passes retrieved documents to the next processing step
        
        3. **Document Cleaner**:
           - Removes irrelevant content from retrieved documents
           - Eliminates redundant information
           - Extracts the most relevant facts and data
        
        4. **Relevance Evaluator**:
           - Evaluates document relevance to the query
           - Assigns a relevance score from 1 to 10 for each document
           - Filters out low-relevance documents (score < 6)
        
        5. **Retriever Reformulator**:
           - Reformulates the query when relevance scores are low
           - Uses different keywords and perspectives
           - Attempts up to 2 reformulations
        
        6. **Answer Generator**:
           - Generates the final answer based on relevant documents
           - Provides a direct answer to the user query
           - Relies only on referenced information
        """

    # Define process function for different system modes
    def process_query(query, system_mode, run_system_fn):
        """Process query based on selected system mode"""
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from utils.retriever import get_retriever
        from utils.state import initialize_state
        
        # 1. Base LLM (No RAG)
        if system_mode == "Base LLM (No RAG)":
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
            answer = llm.invoke(query).content
            steps = "Used base LLM model without retrieval or agents."
            
        # 2. Simple RAG
        elif system_mode == "Simple RAG":
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
            retriever = get_retriever()
            docs = retriever.get_relevant_documents(query)
            
            docs_content = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = ChatPromptTemplate.from_template(
                """Based on the following context information, please answer the user's question.
                
                Question: {query}
                
                Context:
                {docs_content}
                
                Please provide a comprehensive and accurate answer based only on the information in the context.
                """
            )
            
            answer = llm.invoke(prompt.format(query=query, docs_content=docs_content)).content
            steps = f"Retrieved {len(docs)} documents\nUsed simple RAG approach without advanced agents."
            
        # 3. Advanced RAG without LoRA
        elif system_mode == "Advanced RAG (No Fine-tuning)":
            # Disable LoRA
            os.environ["DISABLE_LORA"] = "true"
            result = run_system_fn(query)
            answer = result["answer"]
            steps = "\n".join(result["intermediate_steps"])
            # Clean up environment
            if "DISABLE_LORA" in os.environ:
                del os.environ["DISABLE_LORA"]
                
        # 4. Advanced RAG with LoRA
        else:  # Default to Advanced RAG with LoRA
            # Make sure LoRA is enabled
            if "DISABLE_LORA" in os.environ:
                del os.environ["DISABLE_LORA"]
            result = run_system_fn(query)
            answer = result["answer"]
            steps = "\n".join(result["intermediate_steps"])
        
        return answer, steps
    
    with gr.Blocks(title="Environmental News Multi-Agent RAG System") as demo:
        gr.Markdown("# Environmental News Multi-Agent Retrieval-Augmented Generation System")
        gr.Markdown("This system collaborates multiple agents to retrieve and answer questions from an environmental news corpus.")
        
        with gr.Row():
            with gr.Column():
                # Collapsible visualization section
                with gr.Accordion("System Workflow", open=False):
                    view_workflow_btn = gr.Button("View System Workflow Graph")
                    workflow_image = gr.Image(label="System Workflow Graph", visible=False)
                    workflow_desc = gr.Markdown(label="System Description", visible=False)
        
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Enter Your Question", 
                    placeholder="Example: What are the environmental policy challenges for the UK government after October 2021?", 
                    lines=2
                )
                # Add dropdown for system mode selection
                system_mode = gr.Dropdown(
                    label="Select System Mode",
                    choices=[
                        "Advanced RAG with LoRA (Default)",
                        "Advanced RAG (No Fine-tuning)",
                        "Simple RAG",
                        "Base LLM (No RAG)"
                    ],
                    value="Advanced RAG with LoRA (Default)"
                )
                submit_btn = gr.Button("Submit Query")
            
            with gr.Column():
                answer_output = gr.Textbox(label="Answer", lines=10)
                steps_output = gr.Textbox(label="Processing Steps", lines=8)
        
        # Add workflow visualization functionality
        def show_workflow():
            # Generate image
            img_path = generate_rag_graph_image()
            # Get system description
            description = get_system_description()
            # Show components
            return img_path, gr.update(visible=True), description, gr.update(visible=True)
        
        view_workflow_btn.click(
            fn=show_workflow,
            inputs=[],
            outputs=[workflow_image, workflow_image, workflow_desc, workflow_desc]
        )
        
        # Process query
        submit_btn.click(
            fn=lambda q, mode: process_query(q, mode, run_system_fn),
            inputs=[query_input, system_mode],
            outputs=[answer_output, steps_output]
        )
    
    return demo
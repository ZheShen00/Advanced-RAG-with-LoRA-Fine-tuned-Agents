# Multi-Agent RAG System for Unstructured Data Retrieval

## Project Overview

This project implements an advanced Retrieval-Augmented Generation (RAG) system enhanced with multiple specialized agents, where at least one agent uses a LoRA fine-tuned model. The system is designed to effectively process and answer queries about environmental news with high accuracy and relevance.

### Key Features

- **Multi-Agent Architecture**: Utilizes 6 specialized agents for different aspects of the RAG workflow
- **LoRA Fine-Tuned Model**: Incorporates a LoRA fine-tuned model for improved answer generation
- **Vector Database Retrieval**: Uses Pinecone for efficient document storage and retrieval
- **Dynamic Query Handling**: Includes query analysis, reformulation, and confidence-based processing
- **Web Interface**: Provides a user-friendly Gradio interface for interactive querying
- **Comparative Evaluation**: Allows comparison between base LLM, simple RAG, and advanced RAG systems

## Core Components

1. **Multi-Agent Workflow**:
   * Query Analyzer: Enhances user queries for better retrieval (Including LoRA)
   * Retriever: Fetches documents from Pinecone
   * Document Cleaner: Eliminates noise and extracts relevant content
   * Relevance Evaluator: Uses LLM to assess document relevance
   * Answer Generator: Creates the final answer

2. **LoRA Integration**:
   * Query Analyzer agent uses the LoRA model.
   * The main feature of this LORA model is that it focuses more on the logic of the content, so it can help database retrieval by providing more accurate keywords.
   * After that, `ChatGPT-3.5-Turbo` evaluates document relevance on a scale of 1-10
   * Documents with scores ≥6 are retained for answer generation

3. **Multiple RAG Implementations**:
   * Base LLM (No RAG): Uses only LLM (ChatGPT-3.5-turbo) to answer
   * Basic RAG: Simple retrieval and answer generation
   * Advanced RAG: Multi-agent workflow without fine-tuning
   * Advanced RAG with LoRA: Complete system with fine-tuned model (Base model: SmolLM2-1.7B-Instruct)

4. **User Interfaces**:
   * Web Interface: Browser-based interaction with Gradio application
   * CLI: Rich command-line interface with interactive mode
   * Evaluation Tools: Scripts for comparing different RAG approaches

## Project Structure

```
project/
├── agents/
│   ├── __init__.py              # Makes agents a package
│   ├── query_analyzer.py        # Query analysis agent (Using LoRA model)
│   ├── retriever_agent.py       # Retrieval agent
│   ├── document_cleaner.py      # Document cleaning agent
│   ├── relevance_evaluator.py   # Relevance evaluation agent
│   ├── answer_generator.py      # Answer generation agent
│   └── retriever_reformulator.py # Query reformulation agent
├── evaluation/
│   ├── __init__.py              # Makes agents a package
│   ├── evaluator.py             # Evaluation utilities for comparing system configurations
│   └── test_question.py         # Include the test questions
├── utils/
│   ├── __init__.py
│   ├── state.py                 # AgentState definition
│   ├── retriever.py             # Pinecone functionality
│   └── decision_functions.py    # Decision functions
├── models/
│   ├── __init__.py
│   └── lora_model.py            # Defining LoRA fine-tuned model loading and text generation classes for relevance assessment
├── lora_mc_model/               # Folder containing the lora model file and its detailed parameter files
│   └── Regular model setting files are omitted here...
├── app.py                       # Main application
├── graph.py                     # Graph building and visualization
├── interface.py                 # Gradio interface
├── requirements.txt             # Project dependencies
├── Dockerfile                   # Docker configuration
├── .env                         # Environment variables
└── README.md                    # This file which you are reading right now
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key and index
- HuggingFace access token (for loading the fine-tuned model)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   - Replace Pinecone and OpenAI API keys in `.env` file:
     ```
     OPENAI_API_KEY=your_openai_api_key
     PINECONE_API_KEY=your_pinecone_api_key
     ```

## Running the Application

There are multiple ways to run the application:

### 1. Basic Gradio Interface (Recommanded if user searching single question)

```bash
python app.py
```

This launches the Gradio web interface where you can enter queries and compare different system configurations.

### 2. Command Line Options (Recommanded if user searching predefined 8 Questions)

- **Evaluation Mode**:
  ```bash
  python app.py --evaluate
  ```
  Runs evaluation on all system configurations using predefined test queries.

### 3. Unit Test (Recommanded if user searching predefined 8 Questions)

To run the unit test:

```bash
python app.py --test
```

The unit test runs a predefined query through the system and displays the result, verifying that all components work correctly.

The unit test typically completes in less than 5 oe 6 minutes, making it an efficient way to verify system functionality. The test query:
```
"What are the environmental policy challenges for the UK government after October 2021?",
"How have recent European Union regulations affected biodiversity conservation since 2022?",
"What are the latest global responses to deforestation in the Amazon rainforest post-2021?",
"How has the transition to renewable energy progressed in China since 2022?",
"What are the worst hurricanes and extreme weather events in North America since 2022?",
"Who represented his/her country to receive the 2021 winner of the earthshot protect and restore nature award?",
"What are the latest developments in carbon capture and storage (CCS) technologies after 2021?",
"How has the EU's Green Deal evolved after 2021, and what new initiatives have been introduced?",
```

This test allows verification of all system components including query analysis, document retrieval, relevance evaluation, and answer generation with minimal time investment.

## Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t advanced-rag-lora .
   ```

2. Run the container:
   ```bash
   docker run -p 7860:7860 --env-file .env advanced-rag-lora
   ```

3. Access the web interface at `http://localhost:7860`

## System Evaluation

The system allows comparison between four different configurations:

1. **Base LLM (No RAG)**: Uses only the base LLM (ChatGPT-3.5-Turbo) without retrieval or agents
2. **Simple RAG**: Basic retrieval followed by LLM answer generation
3. **Advanced RAG (No Fine-tuning)**: Full agent workflow without the LoRA fine-tuned model
4. **Advanced RAG with LoRA (Default)**: Complete system with the LoRA fine-tuned agent (Base model: SmolLM2-1.7B-Instruct)

To run a comprehensive evaluation:

```bash
python app.py --evaluate
```

This will process a set of predefined queries through all four configurations and generate a detailed comparison report in JSON format. The evaluation results will be saved in the `evaluation_results.json` file, which contains the answers from each system configuration along with response times for each query.

### Evaluation Results

The evaluation results show that the response times for all system configurations are relatively short, usually within 2-15 seconds for simple queries and up to 25 seconds for more complex queries. This demonstrates that our system is efficient despite its sophisticated multi-agent architecture.

The different runs of the system (as seen in both the JSON results and the Web_Output_images folder) show slight variations in the exact responses but maintain consistent overall quality and accuracy, which indicates that the system is very stable in its operation.

## Technical Notes

- The system uses HuggingFace's `all-MiniLM-L6-v2` embeddings model (384 dimensions)
- OpenAI's `gpt-3.5-turbo` provides the base LLM capabilities
- The LoRA fine-tuned model is based on `SmolLM2-1.7B-Instruct` and it is integrated with the Query Analyzer agent
- The agent workflow is implemented using LangGraph's StateGraph
- Pinecone vector database is used for document storage and retrieval

### Performance & Stability

As demonstrated in the `evaluation_results.json` file, the system shows consistent performance across different queries:

- **Base LLM**: Fastest responses (0.65-10.73 seconds) but lacks contextual grounding
- **Simple RAG**: Quick responses (1.37-4.10 seconds) with basic document integration
- **Advanced RAG without LoRA**: Moderate response times (8.68-14.86 seconds) with improved relevance
- **Advanced RAG with LoRA**: Slightly longer response times (11.37-25.02 seconds) but with the highest quality answers

The system demonstrates exceptional stability, producing consistent results across multiple runs. Different testing sessions (reflected in both the evaluation JSON and Web_Output_images) show minor variations in exact wording but maintain the same level of accuracy and information density, confirming the robustness of our multi-agent architecture.

#### Analysis of System Differences

1. Base LLM → Simple RAG  
- Provides significantly more relevant and domain-specific information.  
- However, the information is less structured and more scattered.  

2. Simple RAG → Advanced RAG (No Fine-tuning)  
- Better organization and filtering, reducing irrelevant content.  
- Answers are more focused and contextually relevant.  

3. Advanced RAG (No Fine-tuning) → Advanced RAG with LoRA  
- More structured presentation as well as content that focuses almost exclusively on the question itself, with few generalizations.
- Delivers more precise answers.  
- Improves citation clarity and source attribution.  
- Provides a more accurate grasp of intricate question details.  

## Troubleshooting

- If you encounter recursion errors, try increasing the recursion limit in the graph.py file
- For memory issues, reduce the number of retrieved documents or document cleaning threshold
- API rate limits can be managed by adding appropriate sleeps between API calls

## Contact

For any questions or issues, please contact:
- **Name**: Zhe Shen
- **Email**: zheshen2025@u.northwestern.edu

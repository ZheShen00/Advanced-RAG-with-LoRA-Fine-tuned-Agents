# utils/retriever.py
import os
import pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings

# Update Pinecone import, using the new package path
try:
    from langchain_pinecone import Pinecone
except ImportError:
    # If the new package is not installed, fall back to the old import
    from langchain.vectorstores import Pinecone

def get_retriever():
    """Connect to Pinecone vector database and return a retriever"""
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    
    # Initialize Pinecone
    pc = pinecone.Pinecone(
        api_key=PINECONE_API_KEY
    )
    
    # Connect to index
    index_name = "text-embedding-index"
    
    # Initialize embedding model - Using all-MiniLM-L6-v2 (384 dimensions)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store
    try:
        vectorstore = Pinecone(
            index=pc.Index(index_name),
            embedding=embeddings,
            text_key="Cleaned Text"
        )
    except Exception as e:
        print(f"Error creating vector store: {e}")
        print("Attempting fallback method...")
        try:
            # Try fallback method
            vectorstore = Pinecone.from_existing_index(
                index_name=index_name,
                embedding=embeddings,
                text_key="Cleaned Text"
            )
        except Exception as e2:
            print(f"Fallback method also failed: {e2}")
            raise Exception("Unable to connect to Pinecone index, please check index name and API key")
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    return retriever
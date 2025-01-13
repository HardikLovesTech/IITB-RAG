from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
from typing import List, Tuple
import os


def load_documents(directory: str) -> List[Document]:
    """
    Load .txt and .pdf documents from a directory.

    Args:
        directory (str): Path to the directory containing files.

    Returns:
        List[Document]: A list of LangChain Document objects.
    """
    documents = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        try:
            if filename.endswith(".txt"):
                loader = TextLoader(file_path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)

            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)

        except Exception as e:
            print(f"Error loading {filename}: {e}")

    return documents


def initialize_faiss(documents: List[Document], db_path: str) -> FAISS:
    """
    Initializes a FAISS vector database and stores documents.

    Args:
        documents (List[Document]): List of LangChain Document objects.
        db_path (str): Path to store the FAISS database.

    Returns:
        FAISS: FAISS vector store object.
    """
    # Initialize Ollama embeddings
    ollama_embeddings = OllamaEmbeddings(model="llama2")

    # Create FAISS vector store and add documents
    vectorstore = FAISS.from_documents(documents, ollama_embeddings)
    
    # Save the vector store
    vectorstore.save_local(db_path)

    return vectorstore


def query_database(query: str, vectorstore: FAISS, k: int) -> List[Document]:
    """
    Queries the FAISS database for the top-k similar documents.

    Args:
        query (str): Query text.
        vectorstore (FAISS): FAISS vector store object.
        k (int): Number of top results to return.

    Returns:
        List[str]: List of top-k document contents from the database.
    """
    # Perform similarity search in the database
    retrieved_documents = vectorstore.similarity_search(query, k=k)

    # Extract the content from the retrieved documents
    results = [doc.page_content for doc in retrieved_documents]

    return results


# Example Usage
if __name__ == "__main__":
    # Define the path to the directory where your documents are stored
    directory_path = "/home/hardik/Documents/Coding/IIT_Bombay/AI/Coding"

    # Load documents
    documents = load_documents(directory_path)

    # Initialize FAISS with the documents
    db_path = "./vector_db"
    vectorstore = initialize_faiss(documents, db_path)

    # Load existing index (if needed)
    vectorstore = FAISS.load_local(db_path, OllamaEmbeddings(model="llama2"))

    # Define the query and retrieve top-k results
    query_text = "What is attention?"
    top_k = 2
    results = query_database(query_text, vectorstore, top_k)

    # Print out the results
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result}")
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
import os
from typing import List


def load_documents(directory: str) -> List[Document]:
    """
    Skeleton Function: Load .txt and .pdf documents from a directory.

    Args:
        directory (str): Path to the directory containing files.

    Returns:
        List[Document]: A list of LangChain Document objects.
    """
    # Initialize an empty list to store the documents
    documents = []

    # Loop through files in the specified directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        try:
            # Placeholder for loading .txt files
            if filename.endswith(".txt"):
                loader=TextLoader(file_path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                pass  # Replace with code for TextLoader

            # Placeholder for loading .pdf files
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                pass  # Replace with code for PyPDFLoader

        except Exception as e:
            # Print error for files that could not be loaded
            print(f"Error loading {filename}: {e}")

    # Return the list of documents
    return documents



def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits documents into smaller chunks.

    Args:
        documents (List[Document]): List of LangChain Document objects.

    Returns:
        List[Document]: A list of chunked Document objects.
    """
    # Create an instance of the text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = []

    # Iterate over each document and split it into chunks
    for doc in documents:
        # Split the document and add chunks to the list
        chunks.extend(text_splitter.split_documents([doc]))

    return chunks



def generate_embeddings(chunks: List[Document]) -> List[List[float]]:
    """
    Generates vector embeddings for the given chunks.

    Args:
        chunks (List[Document]): List of chunked Document objects.

    Returns:
        List[List[float]]: A list of vector embeddings.
    """
    # Initialize the OpenAI embeddings model
    embeddings = OpenAIEmbeddings(
        api_key="YOUR_API_KEY"
    )
    # Generate embeddings for each chunk
    return [embeddings.embed_query(chunk.page_content) for chunk in chunks]




# Example Usage
if __name__ == "__main__":
    # Sample directory path where documents are stored
    directory_path = "/home/hardik/Documents/Coding/IIT_Bombay/AI/Coding"

    # Load documents (This function is implemented in Assignment 1) ----> Done
    documents = load_documents(directory_path)

    # Chunk the documents into smaller chunks
    chunks = chunk_documents(documents)
    

    # Generate embeddings for the chunks
    embeddings = generate_embeddings(chunks)

    # Display first 5 embeddings for demonstration
    for i, embedding in enumerate(embeddings[:5]):  # Display first 5 embeddings for brevity
        print(f"Embedding {i + 1}: {embedding[:10]}...")  # Print first 10 dimensions for brevity

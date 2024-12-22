from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
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

# Example usage
if __name__ == "__main__":
    # Define the directory path
    directory_path = "/home/hardik/Documents/Coding/IIT_Bombay/AI/Coding"

    # Call the function to load documents
    docs = load_documents(directory_path)

    print(len(docs))
    

    # Iterate through the loaded documents and print metadata and content preview
    for doc in docs:
        #print(f"File: {doc.metadata.get('filename', 'Unknown')}, Content Preview: {doc.page_content[:100]}")
        print('Document(Metadata of the file is : ' + str(doc.metadata) , end = ' ')
        print('Content is :' + doc.page_content[:200] + ')')

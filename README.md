# AI-Powered Document Question Answering System

This project enables users to upload a document, analyze its content using AI-powered vector embeddings, and ask questions to receive answers based on the document. The system processes the document by dividing it into chunks, embedding those chunks, storing them in a vector database, and providing relevant answers through a similarity search. It leverages advanced AI models and techniques to enhance document understanding and query answering.

## Project Overview

The AI-Powered Document Question Answering System is designed to help users easily extract relevant information from large documents. By processing the document content into smaller chunks and generating vector embeddings, the system can perform a similarity search to find the most relevant content in response to user queries. The process is efficient, scalable, and can be applied to various types of documents (e.g., PDFs, Word files, etc.).

The project consists of three main components:

1. **Document Loading**: A script to load the document into the system.
2. **Document Chunking**: A script to divide the document into smaller chunks for efficient processing.
3. **Vector Embedding & Querying**: A script that converts document chunks into vector embeddings using AI models and allows the user to ask questions to retrieve answers based on document content.

## Project Structure

### 1. **Document Loading (`document_loader.py`)**
   - This script is responsible for loading a document into the system.
   - It supports different file types such as PDFs, Word documents, and plain text files.
   - The document is pre-processed (if necessary) to extract its raw content.
   - ![Document Upload](Images/Document_Loading.py.png)

### 2. **Document Chunking (`document_chunker.py`)**
   - After the document is loaded, this script splits the content into smaller, logical chunks for better analysis and indexing.
   - This chunking process is essential for large documents, as it allows the system to handle smaller pieces of text, improving both processing and retrieval times.
   - Each chunk represents a part of the document (e.g., a paragraph, section, or page).
   -    ![Document Chunking](Images/Chunking.py.png)

### 3. **Vector Embedding & Querying (`vector_embedding.py`)**
   - This script performs the vectorization of the document chunks using AI-powered embeddings.
   - The vectorization process converts the text data into numerical vectors, which represent the semantic meaning of the document.
   - A vector database is created to store these embeddings, and a similarity search mechanism is applied to answer user queries based on the vectorized content.
   - The AI embeddings used in this script are powered by Hugging Face models and LangChain, which enable efficient and intelligent query answering.
   -    ![Vector Embedding](Images/Query_Solving.py.png)

## How It Works

### Step 1: **Document Upload**
   - Users upload a document (e.g., a PDF, Word file) to the system using the `document_loader.py` script.
   - The document is parsed, and its content is extracted for further processing.

### Step 2: **Document Chunking**
   - The `document_chunker.py` script divides the document into smaller chunks, ensuring that each chunk is meaningful and manageable for AI processing.
   - This step helps to break large documents into sections that are easier to work with and increases the accuracy of the query results.

### Step 3: **Vector Embedding**
   - Once the document is chunked, the `vector_embedding.py` script embeds each chunk into a vector format.
   - The embedding process involves converting the textual content into dense vector representations using pre-trained models like those available from Hugging Face.
   - These embeddings capture the semantic meaning of each chunk and allow for efficient comparison during query answering.

### Step 4: **Querying and Similarity Search**
   - After the document is embedded, users can input questions related to the document.
   - The system performs a similarity search on the embedded chunks to identify the most relevant information that answers the user's query.
   - The AI model uses cosine similarity or other distance measures to find the chunk that is semantically closest to the query.
   - The system returns the relevant content along with the answer to the user's question.

## Technologies Used

- **AI Embeddings**: Utilized pre-trained models (e.g., BERT, GPT) to process and understand the content of the document, transforming text into vector embeddings.
- **LangChain**: Used for orchestrating the flow of document processing and query answering. LangChain helps in integrating language models with document loaders, chunkers, and vector stores.
- **Hugging Face**: Provides access to a wide range of pre-trained NLP models for embedding and understanding document content.
- **Vector Database**: A custom vector database is created to store the embeddings of document chunks. The vector database supports efficient querying and retrieval based on semantic similarity.
- **Similarity Search**: Employed to match user queries with the most relevant document chunks based on vector similarity.

## Installation and Setup

### 1. **Clone the Repository**
   To get started with the project, first clone the repository to your local machine:
   ```bash
   git clone git@github.com:HardikLovesTech/IITB-RAG.git
   cd your-repository

# Connect with Me

Feel free to check out my LinkedIn profile and let's connect!

🔗 **LinkedIn:** [Hardik Runwal](https://www.linkedin.com/in/hardik-runwal/)

Let's stay in touch and collaborate on exciting opportunities!


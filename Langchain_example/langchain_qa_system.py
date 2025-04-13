"""
Personal Document Q&A System

This script sets up a document Q&A system using:
- LangChain for orchestration
- Document loading and text splitting
- ChromaDB for vector storage
- Ollama for the LLM
"""

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Step 1: Set up environment and paths
CHROMA_DB_DIR = "./chroma_db"
DOCUMENT_DIR = "./documents"

# Step 2: Define document loading function
def load_documents(directory):
    """Load documents from a directory, supporting .txt and .pdf files."""
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if filename.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    
    print(f"Loaded {len(documents)} documents")
    return documents

# Step 3: Define text splitting
def split_documents(documents):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    print(f"Split into {len(splits)} chunks")
    return splits

# Step 4: Setup embeddings and vector store
def setup_vector_store(splits):
    """Create or load a vector store with document chunks."""
    # Using a smaller, faster embedding model suitable for local use
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create or load the vector store
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
        print(f"Loaded existing vector store from {CHROMA_DB_DIR}")
    else:
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        vector_store.persist()
        print(f"Created new vector store in {CHROMA_DB_DIR}")
    
    return vector_store

# Step 5: Setup LLM with Ollama
def setup_ollama_llm(model_name="llama3:latest"):
    """Setup the Ollama LLM."""
    llm = OllamaLLM(model=model_name)
    return llm

# Step 6: Create QA chain
def setup_qa_chain(vector_store, llm):
    """Create a retrieval chain for question answering."""
    # Setup retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:
    
    Context:
    {context}
    
    Question: {input}
    """)
    
    # Create response synthesis chain
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        output_parser=StrOutputParser()
    )
    
    # Create the final retrieval chain
    qa_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )
    
    return qa_chain

# Step 7: Main function to run the Q&A system
def main():
    """Main function to set up and run the Q&A system."""
    # Create directories if they don't exist
    os.makedirs(DOCUMENT_DIR, exist_ok=True)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    
    # Check if there are documents to process
    if not os.listdir(DOCUMENT_DIR):
        print(f"Please add some .txt or .pdf files to the {DOCUMENT_DIR} directory")
        return
    
    # Load and process documents
    documents = load_documents(DOCUMENT_DIR)
    splits = split_documents(documents)
    vector_store = setup_vector_store(splits)
    
    # Setup LLM and QA chain
    llm = setup_ollama_llm()
    qa_chain = setup_qa_chain(vector_store, llm)
    
    # Interactive Q&A loop
    print("\n🔍 Document Q&A System Ready!")
    print("Type 'exit' to quit")
    
    while True:
        question = input("\nQuestion: ")
        if question.lower() == 'exit':
            break
        
        # Get answer
        result = qa_chain.invoke({"input": question})
        answer = result["answer"]
        
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()

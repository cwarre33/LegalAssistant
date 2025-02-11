import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
import tempfile
from typing import List



# Set up the Streamlit UI
st.title("AI-Powered Legal Assistant with Groq")

# Upload contract files
uploaded_files = st.file_uploader(
    "Upload Legal Documents (PDF, DOCX, TXT)", 
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# Input for user query regarding legal clauses
user_query = st.text_input("Enter your query about the legal contract clauses:")

# Select chunking method
chunking_method = st.sidebar.selectbox("Select Chunking Method", [
    "Character-based (default)", "Sentence-based (NLTK)", "Section-based (Headings)"])

# Select retrieval method
retrieval_method = st.sidebar.selectbox("Select Retrieval Method", [
    "Max Marginal Relevance (MMR)", "Standard Similarity", "Hybrid Search"])


def process_documents(uploaded_files: List) -> str:
    """Process multiple document types and return concatenated text"""
    all_text = ""
    
    for uploaded_file in uploaded_files:
        file_content = uploaded_file.read()
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        try:
            if file_extension == ".pdf":
                loader = UnstructuredPDFLoader(tmp_file_path)
            elif file_extension == ".docx":
                loader = UnstructuredWordDocumentLoader(tmp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(tmp_file_path)
            else:
                st.warning(f"Unsupported file type: {file_extension}")
                continue

            docs = loader.load()
            for doc in docs:
                all_text += doc.page_content + "\n\n"

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            os.unlink(tmp_file_path)

    return all_text.strip()

if uploaded_files and user_query:
    with st.spinner("Processing legal documents..."):
        all_text = process_documents(uploaded_files)
        
        if not all_text:
            st.error("No text extracted from documents. Please check file formats.")
            st.stop()

    # Select chunking method
    if chunking_method == "Character-based (default)":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\nSECTION ", "\n\nArticle ", "\n\nCLAUSE ", "\n\n"]
        )
    elif chunking_method == "Sentence-based (NLTK)":
        from langchain.text_splitter import NLTKTextSplitter
        text_splitter = NLTKTextSplitter()
    elif chunking_method == "Section-based (Headings)":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=250,
            separators=["\n\nSECTION ", "\n\nArticle ", "\n\nCLAUSE ", "\n\nHeading "]
        )
    
    with st.spinner("Analyzing document structure..."):
        chunks = text_splitter.split_text(all_text)
        chunk_docs = [Document(page_content=chunk) for chunk in chunks]

    # Generate embeddings
    with st.spinner("Creating document embeddings..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        faiss_store = FAISS.from_documents(chunk_docs, embeddings)

    # Retrieve relevant clauses based on the selected method
    with st.spinner("Searching for relevant clauses..."):
        if retrieval_method == "Max Marginal Relevance (MMR)":
            retrieved_clauses = faiss_store.max_marginal_relevance_search(
                user_query, k=3, fetch_k=10, lambda_mult=0.5
            )
        elif retrieval_method == "Standard Similarity":
            retrieved_clauses = faiss_store.similarity_search(user_query, k=3)
        elif retrieval_method == "Hybrid Search":
            retrieved_clauses = faiss_store.max_marginal_relevance_search(
                user_query, k=5, fetch_k=15, lambda_mult=0.7
            )

    # Groq API integration
    legal_prompt_template = """
    As a legal expert, analyze this clause in depth:
    {clause}

    Provide a structured analysis covering:
    1. Key obligations and parties involved
    2. Potential legal risks and implications
    3. Standard industry practices comparison
    4. Recommended negotiation points
    5. Compliance considerations

    Use clear, professional language suitable for corporate lawyers.
    """

    prompt = PromptTemplate(input_variables=["clause"], template=legal_prompt_template)

    try:
        llm = ChatGroq(
            temperature=0.3,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=st.secrets["GROQ_API_KEY"]
        )

        for i, clause in enumerate(retrieved_clauses):
            with st.spinner(f"Analyzing Clause {i+1} with Groq AI..."):
                formatted_prompt = prompt.format(clause=clause.page_content)
                response = llm.invoke(formatted_prompt)
                
                st.subheader(f"Relevant Clause {i+1}:")
                st.write(clause.page_content)
                st.subheader(f"Groq AI Analysis {i+1}:")
                st.write(response.content)

    except Exception as e:
        st.error(f"Groq API Error: {str(e)}")
        st.info("Ensure you have a valid API key in Streamlit secrets")

else:
    st.info("Please upload legal documents and enter a query to begin analysis.")

# Sidebar updates
st.sidebar.markdown("### Configuration Options")
st.sidebar.markdown("""
- **Chunking Methods**: Character-based, Sentence-based (NLTK), Section-based
- **Chunk Sizes**: 1000, 1500, 2000 tokens
- **Retrieval Modes**: MMR, Standard Similarity, Hybrid Search
""")

st.sidebar.markdown("### Requirements")
st.sidebar.markdown("""
1. Groq API key (set in Streamlit secrets)
2. `langchain-groq` package installed
3. `python-docx` for Word document support
""")

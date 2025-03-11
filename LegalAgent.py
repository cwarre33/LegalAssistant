import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import tempfile
import fitz
import numpy as np
from typing import List, Dict, Any


# Set up the Streamlit UI
st.set_page_config(
    page_title="Advanced Legal Assistant", 
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
<style>
    .main-header {color:#1E88E5; font-size:28px; font-weight:700;}
    .sub-header {color:#0D47A1; font-size:20px; font-weight:600; margin-top:15px;}
    .info-box {background-color:#E3F2FD; padding:15px; border-radius:5px; margin:10px 0;}
    .warning-box {background-color:#FFF3E0; padding:15px; border-radius:5px; margin:10px 0;}
    .success-box {background-color:#E8F5E9; padding:15px; border-radius:5px; margin:10px 0;}
    .human-review {background-color:#FFEBEE; padding:20px; border-radius:5px; margin:10px 0; border-left:4px solid #F44336;}
    .confidence-high {color:#2E7D32; font-weight:600;}
    .confidence-medium {color:#FF8F00; font-weight:600;}
    .confidence-low {color:#C62828; font-weight:600;}
</style>
""", unsafe_allow_html=True)

st.markdown("<p class='main-header'>⚖️ AI-Powered Legal Assistant with Advanced RAG</p>", unsafe_allow_html=True)
st.markdown("<div class='info-box'>This assistant analyzes legal documents and answers questions with improved accuracy using advanced RAG techniques including query rewriting, speculative retrieval, and human oversight.</div>", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("<p class='sub-header'>Configuration</p>", unsafe_allow_html=True)
    
    # Select RAG techniques to use
    st.markdown("### Advanced RAG Techniques")
    use_query_rewriting = st.checkbox("Query Rewriting", value=True, help="Reformulate ambiguous queries for better retrieval")
    use_hyde = st.checkbox("HyDE (Hypothetical Document Embeddings)", value=True, help="Generate hypothetical answers to improve retrieval")
    use_speculative_rag = st.checkbox("Speculative RAG", value=True, help="Generate preliminary answers before retrieval")
    use_human_validation = st.checkbox("Human-in-the-loop", value=True, help="Request human validation for low confidence answers")
    
    # Select chunking method
    chunking_method = st.selectbox("Chunking Method", [
        "Character-based (default)", 
        "Sentence-based (NLTK)", 
        "Section-based (Headings)"
    ])
    
    # Select retrieval method
    retrieval_method = st.selectbox("Retrieval Method", [
        "Max Marginal Relevance (MMR)",
        "Standard Similarity", 
        "Hybrid Search"
    ])
    
    # Confidence threshold for human review
    confidence_threshold = st.slider("Confidence Threshold for Human Review", 0.0, 1.0, 0.6, 0.05)
    
    # API Key configuration
    st.markdown("### API Configuration")
    api_key = st.text_input("Groq API Key", type="password")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    else:
        try:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        except:
            st.warning("Please provide a Groq API key to use the assistant")

# Main content
# Upload contract files
uploaded_files = st.file_uploader(
    "Upload Legal Documents (PDF, DOCX, TXT)", 
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# Input for user query regarding legal clauses
original_query = st.text_input("Enter your question about the legal document:", placeholder="e.g., What are my obligations under the non-compete clause?")


# Function to process documents and extract text
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
                doc = fitz.open(tmp_file_path)
                for page in doc:
                    all_text += page.get_text() + "\n\n"
            elif file_extension == ".docx":
                loader = UnstructuredWordDocumentLoader(tmp_file_path)
                docs = loader.load()
                for doc in docs:
                    all_text += doc.page_content + "\n\n"
            elif file_extension == ".txt":
                loader = TextLoader(tmp_file_path)
                docs = loader.load()
                for doc in docs:
                    all_text += doc.page_content + "\n\n"
            else:
                st.warning(f"Unsupported file type: {file_extension}")
                continue

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            os.unlink(tmp_file_path)

    return all_text.strip()


# Function to rewrite the query for better retrieval
def rewrite_query(query: str) -> Dict[str, Any]:
    """Rewrite ambiguous queries to be more specific and retrievable"""
    try:
        llm = ChatGroq(
            temperature=0.2,
            model_name="llama-3.3-70b-versatile",
        )
        
        rewrite_prompt = PromptTemplate.from_template(
            """You are an expert legal assistant specializing in interpreting vague legal questions.
            Your task is to rewrite the following question to make it more specific, precise, and retrievable
            from a legal document database. Focus on legal terminology, specificity, and clarity.
            
            Original Question: {query}
            
            Please rewrite this question in a way that would improve retrieval from legal documents.
            The rewritten question should:
            1. Use precise legal terminology
            2. Be more specific about what information is being sought
            3. Include any implied legal concepts that might be relevant
            4. Be formulated as a clear, concise question
            
            Rewritten Question:"""
        )
        
        chain = rewrite_prompt | llm | StrOutputParser()
        rewritten_query = chain.invoke({"query": query})
        
        # Calculate a confidence score for the rewrite
        # Higher score for longer, more specific queries with legal terminology
        legal_terms = ["clause", "provision", "contract", "agreement", "party", "obligation", 
                      "liability", "negligence", "breach", "termination", "compliance", 
                      "jurisdiction", "governing law", "indemnity", "warranty", "remedies"]
        
        original_word_count = len(query.split())
        rewritten_word_count = len(rewritten_query.split())
        
        # Count legal terms in rewritten query
        legal_term_count = sum(1 for term in legal_terms if term.lower() in rewritten_query.lower())
        
        # Calculate confidence based on improvement in length and legal terminology
        length_improvement = min(rewritten_word_count / max(original_word_count, 1), 3) / 3
        legal_term_ratio = min(legal_term_count / max(rewritten_word_count, 1) * 10, 1)
        
        confidence = 0.5 + (length_improvement * 0.25) + (legal_term_ratio * 0.25)
        confidence = min(max(confidence, 0.1), 0.95)  # Bound between 0.1 and 0.95
        
        return {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "confidence": confidence
        }
        
    except Exception as e:
        st.error(f"Error rewriting query: {str(e)}")
        return {
            "original_query": query,
            "rewritten_query": query,
            "confidence": 0.5
        }


# Function to generate hypothetical document for HyDE
def generate_hypothetical_document(query: str) -> Dict[str, Any]:
    """Generate a hypothetical legal clause that would answer the query"""
    try:
        llm = ChatGroq(
            temperature=0.3,
            model_name="llama-3.3-70b-versatile",
        )
        
        hyde_prompt = PromptTemplate.from_template(
            """You are a legal document drafting expert. Based on the following legal question,
            generate a hypothetical legal clause or document excerpt that would directly address this question.
            
            Legal Question: {query}
            
            Create a realistic, well-formatted legal clause or document excerpt (about 1-2 paragraphs) 
            that would contain the information needed to answer this question. 
            Make it similar to text that might appear in an actual legal document, with appropriate 
            legal terminology and structure.
            
            Hypothetical Legal Text:"""
        )
        
        chain = hyde_prompt | llm | StrOutputParser()
        hypothetical_doc = chain.invoke({"query": query})
        
        # Assess quality of hypothetical document
        legal_patterns = ["section", "clause", "provision", "party", "agreement", "herein", 
                          "thereof", "pursuant", "shall", "obligations", "hereto", "hereunder"]
        
        # Count legal patterns in hypothetical document
        pattern_count = sum(1 for pattern in legal_patterns if pattern.lower() in hypothetical_doc.lower())
        
        # Calculate confidence based on length and legal patterns
        doc_length = len(hypothetical_doc.split())
        length_factor = min(doc_length / 50, 1)  # Normalize by expected min length
        pattern_factor = min(pattern_count / 3, 1)  # Expect at least 3 legal patterns
        
        confidence = (length_factor * 0.7) + (pattern_factor * 0.3)
        confidence = min(max(confidence, 0.3), 0.9)  # Bound between 0.3 and 0.9
        
        return {
            "hypothetical_document": hypothetical_doc,
            "confidence": confidence
        }
        
    except Exception as e:
        st.error(f"Error generating hypothetical document: {str(e)}")
        return {
            "hypothetical_document": "",
            "confidence": 0.0
        }


# Function to generate speculative answer before retrieval
def generate_speculative_answer(query: str) -> Dict[str, Any]:
    """Generate a preliminary answer based on general legal principles"""
    try:
        llm = ChatGroq(
            temperature=0.2,
            model_name="llama-3.3-70b-versatile",
        )
        
        speculative_prompt = PromptTemplate.from_template(
            """You are a legal expert providing preliminary guidance. Based on the following legal question,
            provide a speculative answer based on general legal principles and common practices.
            
            Legal Question: {query}
            
            Provide a preliminary answer with the following structure:
            1. General legal principle that applies
            2. Common practice in this area
            3. Potential considerations or variables
            4. Disclaimer that specific document review is needed
            
            Your answer should be clear but cautious, highlighting that this is preliminary guidance 
            without reviewing the specific legal documents.
            
            Speculative Answer:"""
        )
        
        confidence_prompt = PromptTemplate.from_template(
            """On a scale from 0.0 to 1.0, rate your confidence in the following speculative legal answer.
            Consider factors such as:
            - How well-established the legal principles are in this area
            - How much variation exists in practice
            - How much the answer would depend on specific document language
            - How much jurisdiction matters in this case
            
            Legal Question: {query}
            Speculative Answer: {speculative_answer}
            
            Provide only a number between 0.0 and 1.0 representing your confidence:"""
        )
        
        # Generate speculative answer
        spec_chain = speculative_prompt | llm | StrOutputParser()
        speculative_answer = spec_chain.invoke({"query": query})
        
        # Generate confidence score
        conf_chain = confidence_prompt | llm | StrOutputParser()
        confidence_str = conf_chain.invoke({"query": query, "speculative_answer": speculative_answer})
        
        # Parse confidence score
        try:
            confidence = float(confidence_str.strip())
            confidence = min(max(confidence, 0.0), 1.0)  # Ensure it's between 0 and 1
        except:
            confidence = 0.5  # Default confidence if parsing fails
        
        return {
            "speculative_answer": speculative_answer,
            "confidence": confidence
        }
        
    except Exception as e:
        st.error(f"Error generating speculative answer: {str(e)}")
        return {
            "speculative_answer": "",
            "confidence": 0.0
        }


# Function to retrieve relevant clauses
def retrieve_clauses(query: str, faiss_store, hypothetical_doc="", method="mmr"):
    """Retrieve relevant clauses from the vector store"""
    try:
        if method.lower() == "mmr" or method.lower() == "max marginal relevance (mmr)":
            return faiss_store.max_marginal_relevance_search(
                query, k=3, fetch_k=10, lambda_mult=0.5
            )
        elif method.lower() == "standard similarity":
            return faiss_store.similarity_search(query, k=3)
        elif method.lower() == "hybrid search":
            # If we have a hypothetical document, include it in the search
            if hypothetical_doc:
                # First get standard results
                standard_results = faiss_store.similarity_search(query, k=2)
                
                # Then get results based on the hypothetical document
                hyde_results = faiss_store.similarity_search(hypothetical_doc, k=2)
                
                # Combine and deduplicate results
                combined_results = []
                seen_content = set()
                
                for doc in standard_results + hyde_results:
                    if doc.page_content not in seen_content:
                        combined_results.append(doc)
                        seen_content.add(doc.page_content)
                
                return combined_results[:3]  # Return top 3 unique results
            else:
                return faiss_store.max_marginal_relevance_search(
                    query, k=5, fetch_k=15, lambda_mult=0.7
                )
        else:
            # Default to MMR
            return faiss_store.max_marginal_relevance_search(
                query, k=3, fetch_k=10, lambda_mult=0.5
            )
    except Exception as e:
        st.error(f"Error retrieving clauses: {str(e)}")
        return []


# Function to generate the final answer
def generate_final_answer(query: str, retrieved_clauses: List[Document], speculative_answer="") -> Dict[str, Any]:
    """Generate a final answer using the retrieved clauses and optionally the speculative answer"""
    try:
        llm = ChatGroq(
            temperature=0.3,
            model_name="llama-3.3-70b-versatile",
        )
        
        # Create a context string from retrieved clauses
        context = "\n\n".join([f"Document Excerpt {i+1}:\n{doc.page_content}" 
                               for i, doc in enumerate(retrieved_clauses)])
        
        # If we have a speculative answer, include it
        spec_context = f"\nPreliminary Analysis: {speculative_answer}\n\n" if speculative_answer else ""
        
        answer_prompt = PromptTemplate.from_template(
            """You are an expert legal assistant analyzing documents for a client.
            
            USER QUESTION: {query}
            
            RETRIEVED DOCUMENT EXCERPTS:
            {context}
            
            {spec_context}
            Based on the retrieved document excerpts, provide a detailed legal analysis addressing the user's question.
            Your response should:
            1. Clearly identify the relevant legal provisions
            2. Explain their implications in plain language
            3. Note any potential ambiguities or areas requiring further clarification
            4. Provide practical next steps or recommendations
            5. Cite specific sections or language from the documents
            
            Structure your answer with clear headings and bullet points for readability.
            
            LEGAL ANALYSIS:"""
        )
        
        confidence_prompt = PromptTemplate.from_template(
            """On a scale from 0.0 to 1.0, rate your confidence in the following legal analysis
            based on how well the retrieved document excerpts address the user's question.
            
            USER QUESTION: {query}
            
            ANALYSIS: {final_answer}
            
            CONFIDENCE SCORE (provide only a number between 0.0 and 1.0):"""
        )
        
        # Generate final answer
        answer_chain = answer_prompt | llm | StrOutputParser()
        final_answer = answer_chain.invoke({
            "query": query,
            "context": context,
            "spec_context": spec_context
        })
        
        # Generate confidence score
        conf_chain = confidence_prompt | llm | StrOutputParser()
        confidence_str = conf_chain.invoke({"query": query, "final_answer": final_answer})
        
        # Parse confidence score
        try:
            confidence = float(confidence_str.strip())
            confidence = min(max(confidence, 0.0), 1.0)  # Ensure it's between 0 and 1
        except:
            confidence = 0.5  # Default confidence if parsing fails
        
        return {
            "final_answer": final_answer,
            "confidence": confidence
        }
        
    except Exception as e:
        st.error(f"Error generating final answer: {str(e)}")
        return {
            "final_answer": "I'm unable to generate a response at this time. Please try again later.",
            "confidence": 0.0
        }


# Function to request human validation
def display_human_validation_interface(query, rewritten_query, retrieved_clauses, speculative_answer, final_answer, confidence):
    """Display interface for human expert to validate and potentially modify the answer"""
    st.markdown("<div class='human-review'>", unsafe_allow_html=True)
    st.markdown("<h3>⚠️ Human Expert Review Required</h3>", unsafe_allow_html=True)
    st.write(f"Confidence in answer: {confidence:.2f} (below threshold of {confidence_threshold})")
    
    st.markdown("### Original Query")
    st.info(query)
    
    if rewritten_query != query:
        st.markdown("### Rewritten Query")
        st.info(rewritten_query)
    
    st.markdown("### Retrieved Context")
    for i, clause in enumerate(retrieved_clauses):
        st.markdown(f"**Document Excerpt {i+1}:**")
        st.text(clause.page_content[:300] + "..." if len(clause.page_content) > 300 else clause.page_content)
    
    if speculative_answer:
        st.markdown("### Preliminary Analysis")
        st.info(speculative_answer)
    
    st.markdown("### Generated Answer")
    st.warning(final_answer)
    
    st.markdown("### Expert Validation")
    expert_action = st.radio(
        "Select action:",
        ["Approve as is", "Edit answer", "Reject and provide new answer"]
    )
    
    if expert_action == "Edit answer":
        expert_answer = st.text_area("Edit the answer:", value=final_answer, height=300)
        if st.button("Submit Edited Answer"):
            st.session_state.final_answer = expert_answer
            st.session_state.expert_validated = True
            st.success("Expert answer submitted successfully!")
            
    elif expert_action == "Reject and provide new answer":
        expert_answer = st.text_area("Provide new answer:", height=300)
        if st.button("Submit New Answer"):
            st.session_state.final_answer = expert_answer
            st.session_state.expert_validated = True
            st.success("Expert answer submitted successfully!")
            
    elif expert_action == "Approve as is":
        if st.button("Approve Answer"):
            st.session_state.expert_validated = True
            st.success("Answer approved by expert!")
    
    st.markdown("</div>", unsafe_allow_html=True)


# Main workflow
if original_query or uploaded_files:
    if "final_answer" not in st.session_state:
        st.session_state.final_answer = None
    if "expert_validated" not in st.session_state:
        st.session_state.expert_validated = False
    
    # Process the workflow
    with st.spinner("Analyzing legal documents..."):
        all_text = ""
        
        if uploaded_files:
            # Step 1: Process documents
            all_text = process_documents(uploaded_files)
            
            if not all_text:
                st.error("No text extracted from documents. Please check file formats.")
                st.stop()
        
        # Step 2: Apply query rewriting if enabled
        if use_query_rewriting:
            with st.spinner("Optimizing your query..."):
                query_info = rewrite_query(original_query)
                query = query_info["rewritten_query"]
        else:
            query = original_query
            query_info = {"original_query": original_query, "rewritten_query": original_query, "confidence": 0.7}
        
        # Step 3: Generate speculative answer if enabled
        speculative_info = {"speculative_answer": "", "confidence": 0.0}
        if use_speculative_rag:
            with st.spinner("Generating preliminary analysis..."):
                speculative_info = generate_speculative_answer(query)
        
        # Step 4: Generate hypothetical document for HyDE if enabled
        hyde_info = {"hypothetical_document": "", "confidence": 0.0}
        if use_hyde:
            with st.spinner("Creating retrieval enhancer..."):
                hyde_info = generate_hypothetical_document(query)
        
        if all_text:
            # Step 5: Chunk documents
            with st.spinner("Analyzing document structure..."):
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
                
                chunks = text_splitter.split_text(all_text)
                chunk_docs = [Document(page_content=chunk) for chunk in chunks]
            
            # Step 6: Generate embeddings
            with st.spinner("Creating document embeddings..."):
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                faiss_store = FAISS.from_documents(chunk_docs, embeddings)
            
            # Step 7: Retrieve relevant clauses
            with st.spinner("Searching for relevant legal clauses..."):
                retrieval_mapping = {
                    "Max Marginal Relevance (MMR)": "mmr",
                    "Standard Similarity": "standard",
                    "Hybrid Search": "hybrid"
                }
                
                retrieved_clauses = retrieve_clauses(
                    query=query,
                    faiss_store=faiss_store,
                    hypothetical_doc=hyde_info["hypothetical_document"] if use_hyde else "",
                    method=retrieval_mapping.get(retrieval_method, "mmr")
                )
        else:
            retrieved_clauses = []
        
        # Step 8: Generate final answer
        with st.spinner("Generating comprehensive legal analysis..."):
            answer_info = generate_final_answer(
                query=query,
                retrieved_clauses=retrieved_clauses,
                speculative_answer=speculative_info["speculative_answer"] if use_speculative_rag else ""
            )
            
            final_answer = answer_info["final_answer"]
            final_confidence = answer_info["confidence"]
            
            # Store in session state
            st.session_state.final_answer = final_answer
        
        # Step 9: Check if human validation is needed
        needs_human_validation = use_human_validation and final_confidence < confidence_threshold
        
        if needs_human_validation and not st.session_state.expert_validated:
            display_human_validation_interface(
                original_query,
                query,
                retrieved_clauses,
                speculative_info["speculative_answer"] if use_speculative_rag else "",
                final_answer,
                final_confidence
            )
        else:
            # Output section
            st.markdown("<p class='sub-header'>📝 Detailed Legal Analysis</p>", unsafe_allow_html=True)
            
            # Show the query rewriting if it was different
            if use_query_rewriting and query_info["original_query"] != query_info["rewritten_query"]:
                with st.expander("🔄 Query Enhancement", expanded=False):
                    st.markdown("**Original Query:**")
                    st.info(query_info["original_query"])
                    st.markdown("**Enhanced Query:**")
                    st.info(query_info["rewritten_query"])
                    st.progress(query_info["confidence"])
            
            # Show the speculative answer if available
            if use_speculative_rag and speculative_info["speculative_answer"]:
                with st.expander("🧠 Preliminary Analysis", expanded=False):
                    st.markdown(speculative_info["speculative_answer"])
                    st.progress(speculative_info["confidence"])
            
            # Show the confidence level
            confidence_level = ""
            if final_confidence >= 0.8:
                confidence_level = "<span class='confidence-high'>High Confidence</span>"
            elif final_confidence >= 0.5:
                confidence_level = "<span class='confidence-medium'>Medium Confidence</span>"
            else:
                confidence_level = "<span class='confidence-low'>Low Confidence</span>"
            
            st.markdown(f"**Confidence Level:** {confidence_level}", unsafe_allow_html=True)
            
            # Display the final answer
            st.markdown(st.session_state.final_answer)
            
            # Show the retrieved clauses
            if retrieved_clauses:
                with st.expander("📄 Referenced Document Sections", expanded=False):
                    for i, clause in enumerate(retrieved_clauses):
                        st.markdown(f"**Document Excerpt {i+1}:**")
                        st.text(clause.page_content)
            
            # Expert validation status
            if st.session_state.expert_validated:
                st.markdown("<div class='success-box'>✅ This response has been reviewed and approved by a legal expert</div>", unsafe_allow_html=True)
            
            # Citation and disclaimer
            st.markdown("<div class='warning-box'>⚠️ Disclaimer: This analysis is for informational purposes only and does not constitute legal advice. Please consult with a qualified legal professional for advice tailored to your specific situation.</div>", unsafe_allow_html=True)
    
else:
    st.markdown("<div class='info-box'>Please upload legal documents and/or enter a query to begin analysis.</div>", unsafe_allow_html=True)
    
    # Show example use cases
    st.markdown("<p class='sub-header'>Example Use Cases</p>", unsafe_allow_html=True)
    st.markdown("""
    - **Contract Review**: Upload employment, vendor, or customer contracts and ask about your rights and obligations
    - **Legal Risk Assessment**: Analyze documents for potential legal risks or compliance issues
    - **Clause Comparison**: Compare similar clauses across multiple documents
    - **Term Definition**: Clarify the meaning of specific legal terms in context
    - **Obligation Identification**: Identify all obligations you must fulfill under an agreement
    """)

# Add footer information
st.markdown("---")
st.caption("Powered by Advanced RAG techniques: Query Rewriting, HyDE, Speculative RAG, and Human-in-the-loop validation")
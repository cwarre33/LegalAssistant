import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader, PyPDFLoader
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
from typing import List, Dict, Any
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from fpdf import FPDF
import base64


# Set up the Streamlit UI
st.set_page_config(
    page_title="AI Research Assistant", 
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
<style>
    .main-header {color:#4527A0; font-size:28px; font-weight:700;}
    .sub-header {color:#283593; font-size:20px; font-weight:600; margin-top:15px;}
    .info-box {background-color:#E8EAF6; padding:15px; border-radius:5px; margin:10px 0;}
    .confidence-high {color:#2E7D32; font-weight:600;}
    .confidence-medium {color:#F57F17; font-weight:600;}
    .confidence-low {color:#C62828; font-weight:600;}
</style>
""", unsafe_allow_html=True)

st.markdown("<p class='main-header'>🔬 AI Research Assistant</p>", unsafe_allow_html=True)
st.markdown("<div class='info-box'>Ask research questions with or without your own papers</div>", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("<p class='sub-header'>Settings</p>", unsafe_allow_html=True)
    
    # Source selection
    source_type = st.radio(
        "Select Research Source",
        ["Web Search", "Uploaded Papers", "Both"]
    )
    
    # Web search settings (only shown when web search is selected)
    if source_type in ["Web Search", "Both"]:
        st.markdown("### Web Search Settings")
        max_web_results = st.slider("Max Web Results", 3, 20, 10)
        st.markdown("---")
    
    # Research technique toggles
    st.markdown("### Research Techniques")
    use_query_refinement = st.checkbox("Query Refinement", value=True)
    
    # Confidence threshold
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.05)
    
    # API Key configuration
    api_key = st.secrets.get("GROQ_API_KEY", None)
    if api_key:
        st.markdown("Groq API Key is set in secrets.")
    else:
        st.markdown("Please set your Groq API Key in the secrets management.")
        st.markdown("You can find your API key in your Groq account settings.")
        st.markdown("You can also set it in the environment variables.")

# Main content
# Upload scientific papers (only shown when documents are selected)
if source_type in ["Uploaded Papers", "Both"]:
    uploaded_papers = st.file_uploader(
        "Upload Scientific Papers (PDF)", 
        type=["pdf"],
        accept_multiple_files=True
    )
else:
    uploaded_papers = []

# Input for research query
original_query = st.text_input("Enter your research question:", placeholder="e.g., How does melatonin affect circadian rhythm disruption?")

# Function to process documents and extract text
def process_documents(uploaded_files: List) -> tuple:
    """Process PDF documents and return concatenated text and metadata"""
    all_text = ""
    metadata_list = []
    
    for uploaded_file in uploaded_files:
        file_content = uploaded_file.read()
        file_name = uploaded_file.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        try:
            doc = fitz.open(tmp_file_path)
            paper_text = ""
            
            # Extract title and abstract
            first_page = doc[0]
            title_area = first_page.get_text("text", clip=(50, 50, first_page.rect.width-50, 200))
            title = title_area.strip().split('\n')[0] if title_area else file_name
            
            # Get abstract from first two pages
            abstract = ""
            for page_num in range(min(2, len(doc))):
                page_text = doc[page_num].get_text()
                if "abstract" in page_text.lower():
                    abstract_start = page_text.lower().find("abstract")
                    abstract_text = page_text[abstract_start:abstract_start+1500]
                    # Try to find end of abstract
                    end_markers = ["introduction", "keywords", "1.", "i."]
                    end_pos = len(abstract_text)
                    for marker in end_markers:
                        marker_pos = abstract_text.lower().find(marker)
                        if marker_pos > 0 and marker_pos < end_pos:
                            end_pos = marker_pos
                    abstract = abstract_text[:end_pos].strip()
                    break
            
            # Extract full text
            for page in doc:
                paper_text += page.get_text() + "\n\n"
            
            metadata = {
                "title": title,
                "filename": file_name,
                "abstract": abstract[:500],
                "page_count": len(doc),
                "source": "uploaded_document"
            }
            
            metadata_list.append(metadata)
            all_text += f"DOCUMENT: {title}\n\nABSTRACT: {abstract}\n\n{paper_text}\n\n"

        except Exception as e:
            st.error(f"Error processing {file_name}: {str(e)}")
        finally:
            os.unlink(tmp_file_path)

    return all_text.strip(), metadata_list

# Function to refine the research query
def enhanced_query_refinement(query: str) -> Dict[str, Any]:
    """Create more structured and specific research queries"""
    try:
        llm = ChatGroq(
            temperature=0.2,
            model_name="llama-3.3-70b-versatile",
        )
        
        rewrite_prompt = PromptTemplate.from_template(
            """Rewrite the following research question to make it more precise and retrievable:
            
            Original Question: {query}
            
            Please structure the rewritten question to include:
            1. Core scientific concept(s) with precise terminology
            2. Specific variables, effects, or relationships being investigated
            3. Target population, system, or domain (if applicable)
            4. Temporal context (recent developments, historical perspective, etc.)
            
            Format the output as:
            - Primary Question: [focused rewritten question]
            - Key Terms: [3-5 technical terms that would appear in relevant papers]
            - Search Strings: [2-3 alternative phrasings for search engines]
            
            Your structured research query:"""
        )
        
        chain = rewrite_prompt | llm | StrOutputParser()
        structured_query = chain.invoke({"query": query})
        
        # Parse the structured query to extract components
        lines = structured_query.split('\n')
        primary_question = ""
        key_terms = []
        search_strings = []
        
        for line in lines:
            if line.startswith("- Primary Question:"):
                primary_question = line.replace("- Primary Question:", "").strip()
            elif line.startswith("- Key Terms:"):
                terms_text = line.replace("- Key Terms:", "").strip()
                key_terms = [term.strip() for term in terms_text.strip("[]").split(",")]
            elif line.startswith("- Search Strings:"):
                strings_text = line.replace("- Search Strings:", "").strip()
                search_strings = [s.strip() for s in strings_text.strip("[]").split(",")]
        
        # Calculate confidence based on specificity
        specificity_score = 0.5
        if len(key_terms) >= 3:
            specificity_score += 0.2
        if len(search_strings) >= 2:
            specificity_score += 0.2
        if len(primary_question.split()) >= 8:
            specificity_score += 0.1
            
        return {
            "original_query": query,
            "rewritten_query": primary_question,
            "key_terms": key_terms,
            "search_strings": search_strings,
            "confidence": specificity_score
        }
        
    except Exception as e:
        st.error(f"Error refining query: {str(e)}")
        return {
            "original_query": query,
            "rewritten_query": query,
            "key_terms": [],
            "search_strings": [],
            "confidence": 0.5
        }

# Function to search the web for scientific citations
def search_web_citations(query: str, max_results: int = 10) -> List[Document]:
    """Search the web for scientific citations using Tavily and DuckDuckGo"""
    try:
        # Tavily search
        tavily = TavilySearch(api_key=st.secrets("TAVILY_API_KEY"))
        tavily_results = tavily.search(query, max_results=max_results)

        # DuckDuckGo search
        ddg = DuckDuckGoSearchAPIWrapper(region="wt-wt", time="y", max_results=max_results)
        clean_query = " ".join(query.split("\n")[0].split()[:20])  # Simplify query
        academic_query = f"site:.edu OR site:.gov {clean_query} filetype:pdf"
        ddg_results = ddg.results(academic_query, max_results)

        # Combine results
        combined_results = tavily_results + ddg_results

        if not combined_results:
            st.warning("No web results found. Try simplifying your query")
            return []

        # Process results into Document format
        web_docs = []
        for result in combined_results:
            content = f"Title: {result.get('title', '')}\n"
            content += f"URL: {result.get('link', '')}\n"
            content += f"Snippet: {result.get('body', '')}"

            web_docs.append(Document(
                page_content=content[:2000],  # Limit content length
                metadata={
                    "title": result.get('title', 'Untitled'),
                    "link": result.get('link', ''),
                    "source": "web_search"
                }
            ))

        return web_docs

    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

        # Process results into Document format
        web_docs = []
        for result in results:
            content = f"Title: {result.get('title', '')}\n"
            content += f"URL: {result.get('link', '')}\n"
            content += f"Snippet: {result.get('body', '')}"
            
            web_docs.append(Document(
                page_content=content[:2000],  # Limit content length
                metadata={
                    "title": result.get('title', 'Untitled'),
                    "link": result.get('link', ''),
                    "source": "web_search"
                }
            ))

        return web_docs

    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []
    
# Function to retrieve relevant papers from documents
def retrieve_papers(query: str, faiss_store, top_k: int = 5):
    """Retrieve relevant scientific papers from the vector store"""
    try:
        return faiss_store.max_marginal_relevance_search(
            query, k=top_k, fetch_k=min(top_k*3, 15), lambda_mult=0.5
        )
    except Exception as e:
        st.error(f"Error retrieving papers: {str(e)}")
        return []

# Function to generate PDF
def generate_pdf_report(title, summary, sections, references):
    """Generate a formatted research report as a PDF file."""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, title.encode('latin-1', 'replace').decode('latin-1'), ln=True, align="C")
        pdf.ln(5)
        pdf.set_font("Arial", "I", 10)
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        pdf.cell(0, 10, f"Report Generated: {current_date}", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Executive Summary", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 10, summary.encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(5)
        for section in sections:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, section["title"].encode('latin-1', 'replace').decode('latin-1'), ln=True)
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 10, section["content"].encode('latin-1', 'replace').decode('latin-1'))
            pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "References", ln=True)
        pdf.set_font("Arial", "", 10)
        for ref in references:
            pdf.multi_cell(0, 10, f"- {ref}".encode('latin-1', 'replace').decode('latin-1'))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            pdf_path = temp_file.name
        
        pdf.output(pdf_path)
        with open(pdf_path, "rb") as file:
            pdf_bytes = file.read()
        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        os.unlink(pdf_path)
        
        return {
            "pdf_base64": base64_pdf,
            "filename": f"{title.replace(' ', '_')}.pdf"
        }
        
    except Exception as e:
        st.error(f"PDF generation error: {str(e)}")
        return None

# Function to generate the final research synthesis
def generate_attributed_synthesis(query: str, documents: List[Document]) -> Dict[str, Any]:
    """Generate a research synthesis with clear source attribution"""
    try:
        llm = ChatGroq(
            temperature=0.3,
            model_name="llama-3.3-70b-versatile",
        )
        
        # Create numbered context from documents with source information
        context_items = []
        for i, doc in enumerate(documents):
            source_type = "Web" if doc.metadata.get('source') == 'web_search' else "Paper"
            source_title = doc.metadata.get('title', f'Source {i+1}')
            source_url = doc.metadata.get('link', 'No link available') if source_type == "Web" else ""
            
            context_entry = f"[{i+1}] {source_type}: {source_title}\n"
            if source_url:
                context_entry += f"URL: {source_url}\n"
            context_entry += f"Content: {doc.page_content[:800]}..."
            context_items.append(context_entry)
        
        context = "\n\n".join(context_items)
        
        synthesis_prompt = PromptTemplate.from_template(
            """You are a scientific researcher synthesizing information with careful attribution.
            
            RESEARCH QUESTION: {query}
            
            SOURCES:
            {context}
            
            Based on these sources, provide a research synthesis that:
            1. Clearly attributes each key finding to specific sources using numbered citations [1], [2], etc.
            2. Distinguishes between established findings and preliminary or contested results
            3. Rates the strength of evidence for major claims (strong, moderate, limited)
            4. Identifies areas of consensus across multiple sources
            5. Explicitly notes when important aspects of the question lack sufficient evidence
            
            Your synthesis should be comprehensive but prioritize research quality over quantity.
            
            ATTRIBUTED RESEARCH SYNTHESIS:"""
        )
        
        # Generate synthesis with attributions
        synthesis_chain = synthesis_prompt | llm | StrOutputParser()
        research_synthesis = synthesis_chain.invoke({
            "query": query,
            "context": context
        })
        
        # Add a citations section automatically
        citations = []
        for i, doc in enumerate(documents):
            source_type = "Web" if doc.metadata.get('source') == 'web_search' else "Paper"
            source_title = doc.metadata.get('title', f'Source {i+1}')
            source_url = doc.metadata.get('link', 'No link available') if source_type == "Web" else ""
            
            citation = f"[{i+1}] {source_title}"
            if source_url:
                citation += f" Retrieved from: {source_url}"
            citations.append(citation)
        
        research_synthesis += "\n\n## References\n" + "\n".join(citations)
        
        # Calculate confidence based on source quality and alignment
        num_sources = len(documents)
        scientific_sources = sum(1 for doc in documents if ".edu" in doc.metadata.get('link', '') or 
                               ".gov" in doc.metadata.get('link', ''))
        confidence = 0.5 + min(0.3, num_sources * 0.05) + min(0.2, scientific_sources * 0.1)
        
        return {
            "research_synthesis": research_synthesis,
            "confidence": confidence,
            "citations": citations
        }
        
    except Exception as e:
        st.error(f"Error generating attributed synthesis: {str(e)}")
        return {
            "research_synthesis": "Unable to generate synthesis at this time.",
            "confidence": 0.0,
            "citations": []
        }
    
# Function to generate a hypothetical research abstract
def generate_hypothetical_abstract(query: str, existing_knowledge: List[Document]) -> Dict[str, Any]:
    """Generate a hypothetical research abstract when literature is limited"""
    try:
        llm = ChatGroq(
            temperature=0.4,
            model_name="llama-3.3-70b-versatile",
        )
        
        # Extract what limited knowledge we have
        context = ""
        if existing_knowledge:
            context = "\n\n".join([
                f"Partial information {i+1}:\n{doc.page_content[:300]}..." 
                for i, doc in enumerate(existing_knowledge)
            ])
        
        abstract_prompt = PromptTemplate.from_template(
            """You are a scientific researcher tasked with generating a hypothetical research abstract.
            
            RESEARCH QUESTION: {query}
            
            LIMITED EXISTING INFORMATION:
            {context}
            
            Create a hypothetical research abstract that:
            1. Clearly states what is known based on available information
            2. Identifies specific knowledge gaps
            3. Proposes a research methodology that could address these gaps
            4. Speculates on potential findings based on scientific principles
            5. Identifies limitations and ethical considerations
            
            Format this as a proper scientific abstract with clear sections.
            
            ⚠️ IMPORTANT: This is speculative content that must be clearly labeled as "HYPOTHETICAL RESEARCH ABSTRACT"
            
            HYPOTHETICAL RESEARCH ABSTRACT:"""
        )
        
        # Generate abstract
        abstract_chain = abstract_prompt | llm | StrOutputParser()
        hypothetical_abstract = abstract_chain.invoke({
            "query": query,
            "context": context if context else "Limited information available."
        })
        
        # Generate confidence (inherently low due to speculative nature)
        confidence = 0.3 + (0.1 * min(len(existing_knowledge), 3))  # 0.3-0.6 based on available context
        
        return {
            "hypothetical_abstract": hypothetical_abstract,
            "confidence": confidence,
            "is_speculative": True
        }
        
    except Exception as e:
        st.error(f"Error generating hypothetical abstract: {str(e)}")
        return {
            "hypothetical_abstract": "Unable to generate a hypothetical abstract at this time.",
            "confidence": 0.0,
            "is_speculative": True
        }

# Add UI elements for human validation
def add_human_validation_ui(research_synthesis, documents, confidence):
    """Add UI elements for human validation of the research synthesis"""
    st.markdown("<p class='sub-header'>👤 Researcher Validation</p>", unsafe_allow_html=True)
    
    # Show confidence information
    st.markdown(f"**System Confidence: {confidence:.2f}**")
    
    if confidence < 0.6:
        st.warning("⚠️ This synthesis has low confidence and requires human review")
    
    # Key findings extraction for validation
    findings_prompt = PromptTemplate.from_template(
        """Extract 3-5 key findings from this research synthesis:
        
        {synthesis}
        
        For each finding, list:
        1. The finding itself
        2. The source number(s) supporting it
        3. The confidence level (high/medium/low)
        
        KEY FINDINGS:"""
    )
    
    try:
        llm = ChatGroq(temperature=0.1, model_name="llama-3.3-70b-versatile")
        chain = findings_prompt | llm | StrOutputParser()
        findings = chain.invoke({"synthesis": research_synthesis})
        
        # Display findings for validation
        st.markdown("### Key Findings for Validation")
        st.markdown(findings)
        
        # Add validation controls
        st.markdown("### Researcher Feedback")
        accuracy_rating = st.slider("Rate synthesis accuracy", 1, 5, 3)
        missing_aspects = st.text_area("Note any missing aspects or inaccuracies")
        additional_sources = st.text_area("Suggest additional sources")
        
        if st.button("Submit Validation"):
            # Here you would store the validation data
            st.success("Validation submitted. The synthesis will be updated accordingly.")
            
            # Optionally regenerate synthesis with feedback
            if missing_aspects or additional_sources:
                st.info("Regenerating synthesis with your feedback...")
                # This would call the synthesis function again with the feedback
    
    except Exception as e:
        st.error(f"Error in validation process: {str(e)}")

# Main workflow
if original_query:
    with st.spinner("Processing your research question..."):
        # Step 1: Enhanced query refinement
        if use_query_refinement:
            query_info = enhanced_query_refinement(original_query)
            query = query_info["rewritten_query"]
            key_terms = query_info.get("key_terms", [])
            search_strings = query_info.get("search_strings", [])
        else:
            query = original_query
            query_info = {"original_query": original_query, "rewritten_query": original_query, "confidence": 0.7}
            key_terms = []
            search_strings = []
        
        # Initialize combined_documents list
        combined_documents = []
        
        # Step 2: Process uploaded papers if available
        if source_type in ["Uploaded Papers", "Both"] and uploaded_papers:
            with st.spinner("Analyzing uploaded papers..."):
                all_text, metadata_list = process_documents(uploaded_papers)
                
                if all_text:
                    # Chunk documents
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=200
                    )
                    chunks = text_splitter.split_text(all_text)
                    chunk_docs = [Document(page_content=chunk, metadata=metadata_list[i % len(metadata_list)]) 
                                 for i, chunk in enumerate(chunks)]
                    
                    # Generate embeddings
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                    faiss_store = FAISS.from_documents(chunk_docs, embeddings)
                    
                    # Retrieve relevant papers using both original and key terms
                    search_queries = [query] + key_terms[:2]
                    for search_query in search_queries:
                        document_results = retrieve_papers(search_query, faiss_store)
                        combined_documents.extend(document_results)
        
        # Step 3: Perform enhanced web search with multiple queries if selected
        if source_type in ["Web Search", "Both"]:
            with st.spinner("Searching for scientific literature online..."):
                # Use the original query
                web_results = search_web_citations(query, max_web_results)
                combined_documents.extend(web_results)
                
                # Use alternative search strings if available
                if search_strings:
                    for search_string in search_strings[:2]:  # Limit to 2 additional searches
                        additional_results = search_web_citations(search_string, max_web_results // 2)
                        combined_documents.extend(additional_results)
        
        # Step 4: Generate synthesis or hypothetical abstract based on document availability
        if combined_documents:
            with st.spinner("Generating research synthesis..."):
                if len(combined_documents) >= 3:
                    # Sufficient literature - generate attributed synthesis
                    synthesis_info = generate_attributed_synthesis(query, combined_documents)
                    research_synthesis = synthesis_info["research_synthesis"]
                    synthesis_confidence = synthesis_info["confidence"]
                    is_hypothetical = False
                else:
                    # Limited literature - generate hypothetical abstract
                    abstract_info = generate_hypothetical_abstract(query, combined_documents)
                    research_synthesis = abstract_info["hypothetical_abstract"]
                    synthesis_confidence = abstract_info["confidence"]
                    is_hypothetical = True
            
            # Display the results
            st.markdown("<p class='sub-header'>📝 Research Synthesis</p>", unsafe_allow_html=True)
            
            # Show if this is hypothetical content
            if is_hypothetical:
                st.warning("⚠️ LIMITED LITERATURE AVAILABLE: The following is a hypothetical research abstract based on limited available information.")
            
            # Show the query refinement if it was different
            if use_query_refinement and query_info["original_query"] != query_info["rewritten_query"]:
                with st.expander("🔄 Query Refinement", expanded=True):
                    st.markdown("**Original Query:**")
                    st.info(query_info["original_query"])
                    st.markdown("**Refined Query:**")
                    st.info(query_info["rewritten_query"])
                    if key_terms:
                        st.markdown("**Key Terms:**")
                        st.info(", ".join(key_terms))
            
            # Show confidence level
            confidence_level = ""
            if synthesis_confidence >= 0.8:
                confidence_level = "<span class='confidence-high'>High Confidence</span>"
            elif synthesis_confidence >= 0.5:
                confidence_level = "<span class='confidence-medium'>Medium Confidence</span>"
            else:
                confidence_level = "<span class='confidence-low'>Low Confidence</span>"
            
            st.markdown(f"**Confidence Level:** {confidence_level}", unsafe_allow_html=True)
            
            # Display synthesis
            st.markdown(research_synthesis)
            
            # Show sources
            with st.expander("📄 Sources", expanded=False):
                for i, doc in enumerate(combined_documents):
                    source_type = "Web" if doc.metadata.get('source') == 'web_search' else "Paper"
                    st.markdown(f"**Source {i+1} ({source_type}):**")
                    
                    if source_type == "Web":
                        st.markdown(f"*{doc.metadata.get('title', 'Untitled')}*")
                        if doc.metadata.get('link'):
                            st.markdown(f"URL: {doc.metadata.get('link')}")
                    else:
                        st.markdown(f"*{doc.metadata.get('title', 'Uploaded Document')}*")
                        st.markdown(f"Pages: {doc.metadata.get('page_count', 'N/A')}")
                    
                    content_preview = doc.page_content.replace('\n', '  \n')  # Preserve line breaks
                    st.markdown(f"```\n{content_preview[:500]}\n...```" if len(content_preview) > 500 
                                else f"```\n{content_preview}\n```")
                    st.markdown("---")
            # Add download button in display logic
            if combined_documents:
                [...]
                if st.button("📥 Generate PDF Report"):
                    pdf_data = generate_pdf_report(
                        f"Research Synthesis: {query}",
                        research_synthesis,
                        [{"title": "Key Findings", "content": research_synthesis}],
                        [citation for citation in search_web_citations(query, max_web_results) if citation.metadata.get('link') is not None]
                    )
                    st.download_button(...)
            # Add human validation interface
            if not is_hypothetical:
                add_human_validation_ui(research_synthesis, combined_documents, synthesis_confidence)
        else:
            st.error("No relevant documents found. Please try a different query or upload papers.")
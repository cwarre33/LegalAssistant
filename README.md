# LegalAssistant

An AI-powered legal document analysis tool built with Streamlit and LangChain. Upload PDF, Word, or plain text legal documents and ask questions — the assistant answers using advanced RAG techniques with human oversight indicators.

## Features

- **Document Upload**: Supports PDF, `.docx`, and `.txt` legal files
- **Advanced RAG Pipeline**: FAISS vector store + HuggingFace sentence embeddings for semantic retrieval
- **Query Rewriting**: Reformulates questions for better retrieval accuracy
- **Speculative Retrieval**: Generates a hypothetical answer to guide document search
- **Confidence Scoring**: Rates responses as High / Medium / Low confidence
- **Human Review Flags**: Automatically flags low-confidence answers for human review
- **Groq LLM Backend**: Fast inference via `langchain-groq`
- **Web Research Agent** (`ResearchAgent.py`): DuckDuckGo + Tavily search for supplementary legal context

## Stack

- **Frontend**: Streamlit
- **LLM**: Groq (ChatGroq)
- **RAG**: LangChain + FAISS + HuggingFace Embeddings
- **Document Parsing**: PyMuPDF, Unstructured, python-docx
- **Search**: DuckDuckGo Search, Tavily

## Setup

```bash
pip install -r requirements.txt
streamlit run LegalAgent.py
```

> Requires a `GROQ_API_KEY` environment variable.

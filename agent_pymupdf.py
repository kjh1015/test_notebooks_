import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
import pymupdf4llm
import pathlib
from langchain.text_splitter import MarkdownTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
import glob
from tkinter import filedialog
import tkinter as tk
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
import chromadb
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State definition for the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb
# Load environment variables
load_dotenv()

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

if 'directory_selected' not in st.session_state:
    st.session_state.directory_selected = False
    st.session_state.selected_directory = ""
    st.session_state.pdf_files = []
    st.session_state.show_file_select = False

def handle_directory_selection():
    """Callback function for directory selection"""
    try:
        chosen_dir = select_directory()
        if chosen_dir:
            st.session_state.selected_directory = chosen_dir
            st.session_state.directory_selected = True
            # Get PDF files immediately after directory selection
            st.session_state.pdf_files = get_pdfs_from_directory(chosen_dir)
            st.session_state.show_file_select = True
    except Exception as e:
        st.error(f"Error selecting directory: {str(e)}")

def process_pdf(pdf_path):
    # Load PDF using PyMuPDFLoader
    loader = PyMuPDFLoader(pdf_path)
    # Load and split the document into pages
    pages = loader.load()
    return pages

def chunk_markdown(md_text):
    # Initialize the Markdown text splitter
    text_splitter = MarkdownTextSplitter(chunk_size=2000, chunk_overlap=200)
    
    # Split the text into chunks
    chunks = text_splitter.split_text(md_text)
    
    # Convert chunks to Documents
    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents

def process_pdf_with_llm(documents):
    # Initialize ChatOpenAI
    llm = ChatOpenAI(temperature=0.7)
    
    # Process each chunk and combine results
    summaries = []
    for doc in documents:
        prompt_template = PromptTemplate(
            input_variables=["content"],
            template="Please analyze the following document section and provide a concise summary:\n\n{content}"
        )
        prompt = prompt_template.format(content=doc.page_content)
        response = llm.invoke(prompt)
        summaries.append(response.content)
    
    # Create final summary
    final_prompt = PromptTemplate(
        input_variables=["summaries"],
        template="Based on these section summaries, provide a coherent overall summary:\n\n{summaries}"
    )
    final_response = llm.invoke(final_prompt.format(summaries="\n\n".join(summaries)))
    return final_response.content

def get_pdfs_from_directory(directory_path):
    """Get all PDF files from the specified directory"""
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    return pdf_files

def select_directory():
    """Open directory picker dialog and return selected path"""
    try:
        root = tk.Tk()
        root.attributes('-topmost', True)  # Make sure dialog appears on top
        root.withdraw()  # Hide the main window
        directory_path = filedialog.askdirectory(parent=root)
        return directory_path
    finally:
        try:
            root.destroy()
        except:
            pass

def setup_retriever():
    # Get all markdown files from data directory
    markdown_files = glob.glob("data/*.md")
    
    if not markdown_files:
        return None, "No markdown files found in data directory"
    
    # Load all markdown files
    documents = []
    for md_file in markdown_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": md_file}
                    )
                )
        except Exception as e:
            st.error(f"Error loading {md_file}: {str(e)}")
    
    # Adjust text splitter parameters for more meaningful chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,          # Smaller chunks for more precise retrieval
        chunk_overlap=50,        # Smaller overlap to reduce redundancy
        length_function=len,
        separators=[
            "\n## ",            # Split on header level 2
            "\n### ",           # Split on header level 3
            "\n\n",             # Split on paragraphs
            "\n",               # Split on lines
            ". ",               # Split on sentences
            " ",                # Split on words as last resort
            ""
        ]
    )
    doc_splits = text_splitter.split_documents(documents)
    
    # Create persistent client
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Delete existing collection if it exists to refresh the chunks
    try:
        client.delete_collection("pdf-markdown-store")
    except:
        pass
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="pdf-markdown-store",
        embedding=OpenAIEmbeddings(),
        client=client
    )
    
    # Create retriever with adjusted k for smaller chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 6              # Increased k to get more small chunks
        }
    )
    
    # Create retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        "search_pdf_documents",
        "Search and retrieve information from the processed PDF documents that have been converted to markdown.",
    )
    
    return retriever_tool, f"Created retriever tool from {len(markdown_files)} files with {len(doc_splits)} chunks"

# Streamlit UI
st.title("PDF Content Extractor")

# Add option to choose between file upload or directory processing
processing_mode = st.radio(
    "Choose processing mode:",
    ["Single File Upload", "Process Directory"]
)

if processing_mode == "Single File Upload":
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Create data directory if it doesn't exist
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        try:
            # Generate a unique filename based on the uploaded file's name
            md_filename = os.path.join(data_dir, f"cache_{uploaded_file.name}.md")
            
            # Check if we have a cached markdown version
            if not os.path.exists(md_filename):
                # Convert PDF to markdown
                md_text = pymupdf4llm.to_markdown("temp.pdf")
                # Save markdown for future use
                pathlib.Path(md_filename).write_bytes(md_text.encode())
            else:
                # Read from cached markdown file
                md_text = pathlib.Path(md_filename).read_text(encoding='utf-8')
            
            # Display markdown content
            st.markdown(md_text)
            
            # Add a button to analyze with LLM
            if st.button("Analyze with AI"):
                with st.spinner("Analyzing document..."):
                    # Chunk the markdown content
                    documents = chunk_markdown(md_text)
                    st.write(f"Document split into {len(documents)} chunks")
                    
                    # Process chunks through LLM
                    analysis = process_pdf_with_llm(documents)
                    st.write("### AI Analysis")
                    st.write(analysis)
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            
        finally:
            # Clean up the temporary PDF file
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")

else:  # Process Directory mode
    col1, col2 = st.columns([3, 1])
    
    with col1:
        directory_path = st.text_input(
            "Enter directory path containing PDFs:",
            value=st.session_state.selected_directory
        )
    
    with col2:
        st.button("Browse...", on_click=handle_directory_selection)
        
    # Show file selection if we have a valid directory
    if st.session_state.show_file_select and st.session_state.pdf_files:
        st.write(f"Found {len(st.session_state.pdf_files)} PDF files")
        
        # Display list of PDFs with checkboxes
        selected_pdfs = st.multiselect(
            "Select PDFs to process:",
            st.session_state.pdf_files,
            format_func=lambda x: os.path.basename(x)
        )
        
        if selected_pdfs and st.button("Process Selected PDFs"):
            for pdf_path in selected_pdfs:
                with st.expander(f"Processing {os.path.basename(pdf_path)}"):
                    try:
                        # Generate a unique filename for markdown cache
                        md_filename = os.path.join("data", f"cache_{os.path.basename(pdf_path)}.md")
                        
                        # Check for cached markdown version
                        if not os.path.exists(md_filename):
                            # Convert PDF to markdown
                            md_text = pymupdf4llm.to_markdown(pdf_path)
                            # Save markdown for future use
                            pathlib.Path(md_filename).write_bytes(md_text.encode())
                        else:
                            # Read from cached markdown file
                            md_text = pathlib.Path(md_filename).read_text(encoding='utf-8')
                        
                        # Display markdown content
                        st.markdown(md_text)
                        
                        # Analyze with LLM
                        with st.spinner("Analyzing document..."):
                            documents = chunk_markdown(md_text)
                            st.write(f"Document split into {len(documents)} chunks")
                            
                            analysis = process_pdf_with_llm(documents)
                            st.write("### AI Analysis")
                            st.write(analysis)
                            
                    except Exception as e:
                        st.error(f"Error processing {pdf_path}: {str(e)}")
    elif st.session_state.show_file_select:
        st.warning("No PDF files found in the specified directory.")
            
    # Add this to your UI section where you want to set up the retriever
    if st.button("Setup Retriever Tool"):
        with st.spinner("Setting up retriever tool..."):
            retriever_tool, message = setup_retriever()
            if retriever_tool:
                st.session_state.retriever_tool = retriever_tool
                st.session_state.tools = [retriever_tool]
                st.success(message)
            else:
                st.warning(message)

    # Example of using the retriever tool
    if 'retriever_tool' in st.session_state:
        # Initialize agent state if not already done
        if 'agent_state' not in st.session_state:
            st.session_state.agent_state = AgentState(messages=[])
        
        query = st.text_input("Ask a question about the documents:")
        if query:
            with st.spinner("Searching..."):
                try:
                    tool_result = st.session_state.retriever_tool.invoke(query)
                    st.write("### Search Results")
                    if isinstance(tool_result, list):
                        for doc in tool_result:
                            st.write("---")
                            if hasattr(doc, 'page_content'):
                                st.write(doc.page_content)
                                st.write(f"Source: {doc.metadata['source']}")
                                
                                # Add the result to agent state messages
                                st.session_state.agent_state["messages"].append(
                                    BaseMessage(
                                        content=f"Retrieved content: {doc.page_content}\nSource: {doc.metadata['source']}",
                                        type="system"
                                    )
                                )
                            else:
                                st.write(doc)
                    else:
                        st.write(tool_result)
                    
                    # Display current state of messages
                    st.write("### Agent State Messages")
                    for msg in st.session_state.agent_state["messages"]:
                        st.write(f"Type: {msg.type}")
                        st.write(f"Content: {msg.content}")
                        st.write("---")
                        
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")  
                    
              
                    
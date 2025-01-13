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
            
            
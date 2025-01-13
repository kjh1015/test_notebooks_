import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
import pymupdf4llm
import pathlib

# Load environment variables
load_dotenv()

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

def process_pdf(pdf_path):
    # Load PDF using PyMuPDFLoader
    loader = PyMuPDFLoader(pdf_path)
    # Load and split the document into pages
    pages = loader.load()
    return pages

# Streamlit UI
st.title("PDF Content Extractor")
st.write("Upload a PDF file to extract its contents")

# File uploader
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
            
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        
    finally:
        # Clean up the temporary PDF file
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
            
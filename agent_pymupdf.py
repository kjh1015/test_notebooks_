import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader

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
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    try:
        # Process the PDF
        documents = process_pdf("temp.pdf")
        
        # Display content from each page
        for doc in documents:
            st.subheader(f"Page {doc.metadata['page'] + 1}")
            st.text_area(
                label=f"Content (first 500 characters)",
                value=doc.page_content[:500] + "...",
                height=200,
                key=f"page_{doc.metadata['page']}"
            )
            
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        
    finally:
        # Clean up the temporary file
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
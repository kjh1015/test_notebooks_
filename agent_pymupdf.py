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
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain import hub
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langchain_core.output_parsers import StrOutputParser
import logging


class AgentState(TypedDict):
    """State definition for the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb
# https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/
# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/llm-compiler/LLMCompiler.ipynb
# Load environment variables
load_dotenv()

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    text_splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=20)
    
    # Split the text into chunks
    chunks = text_splitter.split_text(md_text)
    
    # Convert chunks to Documents
    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents

def process_pdf_with_llm(documents):
    # Initialize ChatOpenAI
    llm = ChatOpenAI(temperature=0.0)
    
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

def setup_tools(retriever_tool):
    """Setup tools for the agent"""
    tools = [retriever_tool]
    return tools

def agent(state):
    """
    Agent function that decides whether to use tools or not.
    """
    print("---AGENT DECISION---")
    messages = state["messages"]
    
    # Create a system message that explicitly instructs the agent to use the tool
    system_message = """You are a helpful assistant with access to a document search tool.
    ALWAYS use the search_pdf_documents tool first to find relevant information.
    DO NOT explain how to use the tool - just use it directly.
    After getting search results, provide a concise answer based on the retrieved information."""
    
    # Add system message if it's not already there
    if not any(msg.type == "system" for msg in messages):
        messages.insert(0, SystemMessage(content=system_message))
    
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
    model = model.bind_tools(st.session_state.tools)
    response = model.invoke(messages)
    
    print(f"Agent response: {response.content}")
    return {"messages": [response]}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant.
    """
    print("---GRADE DOCUMENTS---")
    messages = state["messages"]
    last_message = messages[-1]
    
    class grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    llm_with_tool = model.with_structured_output(grade)
    
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )
    
    chain = prompt | llm_with_tool
    question = messages[0].content
    docs = last_message.content
    
    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score
    
    print(f"Relevance score: {score}")
    return "generate" if score == "yes" else "rewrite"

def should_generate(state):
    """Determine if we should generate an answer"""
    messages = state["messages"]
    for message in messages:
        if "RELEVANCE_SCORE: yes" in message.content:
            return True
    return False

def rewrite(state):
    """
    Transform the query to produce a better question.
    """
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
    response = model.invoke(msg)
    # Return as a dictionary with messages key
    return {"messages": [response]}

def generate(state):
    """
    Generate answer
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content

    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)

    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    # Convert string response to HumanMessage
    return {"messages": [HumanMessage(content=response)]}

def setup_graph():
    """Setup the processing graph with nodes and edges"""
    from langgraph.graph import END, StateGraph, START
    from langgraph.prebuilt import ToolNode, tools_condition
    
    # Create workflow
    workflow = StateGraph(AgentState)
    
    # Define the tool node for retrieval
    retrieve = ToolNode([st.session_state.retriever_tool])
    
    # Add all nodes
    workflow.add_node("agent", agent)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    
    # Start with agent
    workflow.add_edge(START, "agent")
    
    # Add conditional edges from agent to either retrieve or end
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        }
    )
    
    # Add conditional edges from retrieve based on document relevance
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )
    
    # Add final edges
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")
    
    # Compile
    graph = workflow.compile()
    
    return graph

# Add a function to process retrieval results
def process_retrieval(docs):
    """Process retrieved documents into a readable format"""
    if isinstance(docs, list):
        processed_docs = []
        for doc in docs:
            if hasattr(doc, 'page_content'):
                processed_docs.append(f"Content: {doc.page_content}\nSource: {doc.metadata.get('source', 'unknown')}")
            else:
                processed_docs.append(str(doc))
        return "\n\n---\n\n".join(processed_docs)
    return str(docs)

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
                st.session_state.tools = setup_tools(retriever_tool)
                st.success(message)
            else:
                st.warning(message)

    # Update the UI section
    if 'retriever_tool' in st.session_state:
        if 'agent_state' not in st.session_state:
            st.session_state.tools = setup_tools(st.session_state.retriever_tool)
            st.session_state.agent_state = AgentState(messages=[])
            st.session_state.graph = setup_graph()
            st.session_state.awaiting_clarification = False
            st.session_state.original_query = ""
        
        # Main query input
        if not st.session_state.awaiting_clarification:
            query = st.text_input("Ask a question about the documents:")
            if query:
                # Check if query is too vague
                if len(query.strip()) < 3 or query.strip().lower() in ["a", "aa", "aaa", "test"]:
                    st.session_state.awaiting_clarification = True
                    st.session_state.original_query = query
                    st.rerun()
                else:
                    with st.spinner("Processing query..."):
                        try:
                            initial_state = AgentState(
                                messages=[HumanMessage(content=query)]
                            )
                            
                            process_container = st.container()
                            
                            for output in st.session_state.graph.stream(initial_state):
                                print(f"Raw output: {output}")
                                
                                with process_container:
                                    # Handle retrieve output
                                    if isinstance(output, dict) and 'retrieve' in output:
                                        retrieve_data = output['retrieve']
                                        if isinstance(retrieve_data, dict) and 'messages' in retrieve_data:
                                            for msg in retrieve_data['messages']:
                                                if hasattr(msg, 'content'):
                                                    st.write("📚 Retrieved Content:")
                                                    content = msg.content
                                                    if isinstance(content, str):
                                                        # Try to parse content for source information
                                                        if "Content:" in content and "Source:" in content:
                                                            sections = content.split("Content:")
                                                            for section in sections[1:]:
                                                                if "Source:" in section:
                                                                    doc_content, source = section.split("Source:", 1)
                                                                    source = source.strip()
                                                                    filename = source.split('/')[-1] if '/' in source else source
                                                                    
                                                                    with st.expander(f"📄 {filename}"):
                                                                        st.markdown(doc_content.strip())
                                                                        st.caption(f"Source: {source}")
                                                        else:
                                                            # Display as regular content
                                                            with st.expander("📄 Retrieved Document"):
                                                                st.markdown(content)
                                    
                                    # Handle rewrite output
                                    if isinstance(output, dict) and 'rewrite' in output:
                                        rewrite_data = output['rewrite']
                                        if isinstance(rewrite_data, dict) and 'messages' in rewrite_data:
                                            st.write("🔄 Refining Query:")
                                            for msg in rewrite_data['messages']:
                                                if hasattr(msg, 'content'):
                                                    st.info(msg.content)
                                    
                                    # Handle agent output
                                    if isinstance(output, dict) and 'agent' in output:
                                        agent_data = output['agent']
                                        if isinstance(agent_data, dict) and 'messages' in agent_data:
                                            st.write("🤖 Agent Response:")
                                            for msg in agent_data['messages']:
                                                if hasattr(msg, 'content'):
                                                    st.success(msg.content)
                                    
                                    # Handle any other messages
                                    if isinstance(output, dict) and 'messages' in output:
                                        for msg in output['messages']:
                                            if hasattr(msg, 'content'):
                                                st.write("💬 Message:")
                                                st.markdown(msg.content)
                    
                        except Exception as e:
                            st.error(f"Error processing query: {str(e)}")
                            print(f"Full error: {str(e)}")
        
        # Clarification input when needed
        else:
            st.warning(f"Your query '{st.session_state.original_query}' seems unclear. Could you please provide more details?")
            clarification = st.text_input("Please provide a more specific question:", key="clarification_input")
            
            if clarification:
                st.session_state.awaiting_clarification = False
                st.session_state.original_query = ""
                # Clear the text input by forcing a rerun
                st.rerun()
                    
                    
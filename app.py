import streamlit as st 
from dotenv import load_dotenv
from pdf_process import process_pdf
from rag import RAG
import tempfile
import os
from datetime import datetime
import pytz

# Streamlit UI
st.set_page_config(page_title="Chat with your Document", layout="wide")
st.title("📄 Chat with your Document")

# Load environment variables from the .env file
load_dotenv()

# Get the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize log storage in session state
if "logs" not in st.session_state:
    st.session_state.logs = []

# Function to update the logs in the UI
def log_update(message, level="INFO"):
    ist_timezone = pytz.timezone('Asia/Kolkata')
    timestamp = datetime.now(ist_timezone).strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] [{level}] {message}"
    st.session_state.logs.append(formatted_message)  # Append message to the log

# Reset logs for each new question
def reset_logs():
    st.session_state.logs = []  # Clear logs

# Upload PDF and reset chat history if new file
uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")
if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None

if uploaded_pdf is not None:
    if uploaded_pdf.name != st.session_state.last_uploaded_filename:
        st.session_state.chat_history = []  # Reset on new upload
        st.session_state.last_uploaded_filename = uploaded_pdf.name

# Initialize RAG instance and process PDF
if uploaded_pdf and openai_api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_path = tmp_file.name

    log_update("PDF uploaded. Processing...",level="INFO")

    paragraphs = process_pdf(tmp_path, openai_api_key)

    # Remove temp file
    os.remove(tmp_path)

    log_update("PDF processed and indexed.",level="INFO")

    # Initialize RAG
    rag = RAG(openai_api_key=openai_api_key)

    #----------------------------
    # Step 1: Store vector embeddings 
    log_update("Storing text document in vector embeddings...",level="INFO")
    rag.load_documents(paragraphs)
    #----------------------------   

    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize query input
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    with st.form(key="question_form", clear_on_submit=True):
        user_query = st.text_input("Ask a question about the PDF:", key="query_input", placeholder="Type your question here...")
        submitted = st.form_submit_button("Submit")


    if submitted and user_query:

        # --------------------------- 
        # Step 2: Perform similarity search on documents based on user query
        log_update(f"Searching for Relevant Documents by Performing Similarity Search Based on User Query ....","INFO")
        docs = rag.get_most_relevant_docs(user_query, k=3)
        #----------------------------

        #----------------------------
        # Step 3: Combine the relevant documents
        log_update(f"Combining Relevant Documents ...",level="INFO")
        combined_context = "\n\n".join([doc.page_content for doc in docs])
        #----------------------------

        #----------------------------
        # Step 4: Generate the answer using the LLM
        log_update(f"LLM Generating Answer: ...",level="INFO")
        answer = rag.generate_answer(user_query, combined_context,st.session_state.chat_history)
        #----------------------------
    
        st.session_state.chat_history.append((user_query, answer))
      

    # Clear Chat Button
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.logs = []  # Clear logs
        reset_logs()  # Reset logs for each new question

    # Display chat history
    for q, a in st.session_state.chat_history:
        st.markdown(f"**👤 You:** {q}")
        st.markdown(f"**🤖 Bot:** {a}")

    # Display logs in the UI
    st.markdown("### Logs")
    st.text_area("Process Logs", "\n".join(st.session_state.logs), height=300)
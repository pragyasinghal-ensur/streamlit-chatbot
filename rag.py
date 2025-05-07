import openai
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os
import streamlit as st 
from datetime import datetime
import pytz

# Initialize log storage in session state
if "logs" not in st.session_state:
    st.session_state.logs = []

# Function to update the logs in the UI
def log_update(message,level="INFO"):
    ist_timezone = pytz.timezone('Asia/Kolkata')
    timestamp = datetime.now(ist_timezone).strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] [{level}] {message}"
    st.session_state.logs.append(formatted_message)  # Append message to the log

class RAG:
    def __init__(self, model="gpt-4", openai_api_key=None):
        self.llm = ChatOpenAI(model=model, openai_api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vector_store = None
        self.docs = None

    def load_documents(self, documents):
        """Load documents and compute their embeddings."""
        self.docs = documents
        self.vector_store = FAISS.from_texts(documents, self.embeddings)
        # Log the vector store for search space
        log_update(f"Vector store: {str(self.vector_store)}",level="INFO")

    def get_most_relevant_docs(self, query, k):
        """Find the most relevant document for a given query using FAISS."""
        if not self.vector_store:
            raise ValueError("Documents and their embeddings are not loaded. Call `load_documents` first.")

        # Perform similarity search using FAISS
        relevant_docs = self.vector_store.similarity_search(query, k=k)

        # Log the documents returned from similarity search
        log_update(f"Documents retrieved from vector store from similarity search: {relevant_docs}...",level="INFO")  # Log relevant documents
        
        return relevant_docs

    def generate_answer(self, query, relevant_doc, chat_history=None):
        """Generate an answer for a given query based on the most relevant document."""
        history_prompt = ""
        if chat_history:
            for i, (prev_q, prev_a) in enumerate(chat_history):
                history_prompt += f"Q{i+1}: {prev_q}\nA{i+1}: {prev_a}\n"

        prompt = (
            f"{history_prompt}\n"
            f"Context:\n{relevant_doc}\n\n"
            f"User's current question: {query}"
        )

         # Log the first 500 characters of the prompt
        log_update(f"Prompt to LLM: {prompt}...",level="INFO")  
        
        messages = [
            ("system", "You are a helpful assistant that answers questions based on given documents only."),
            ("human", prompt),
        ]
        ai_msg = self.llm.invoke(messages)

        # Log the LLM response
        log_update(f"LLM response: {ai_msg.content}...",level="INFO")

        return ai_msg.content

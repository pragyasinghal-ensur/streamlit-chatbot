import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# Function to upload and process PDF, split into chunks, and initialize embeddings
def process_pdf(pdf_path, openai_api_key):
    openai.api_key = openai_api_key

    # Load the PDF document
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    full_text = " ".join([doc.page_content for doc in pages])

    # Initialize the CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    # Split the text into chunks
    paragraphs = text_splitter.split_text(full_text)

    return paragraphs

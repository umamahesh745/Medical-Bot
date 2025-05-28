# helper.py - Remove the circular import and define the missing functions

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

def load_pdf_file(data):
    """Load PDF files from a directory"""
    loader = DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_split(extracted_data):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_hugging_face_embeddings():
    """Download and return HuggingFace embeddings"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# Only run the Pinecone setup if this file is executed directly
if __name__ == "__main__":
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

    extracted_data = load_pdf_file(data='Data/')
    text_chunks = text_split(extracted_data)
    embeddings = download_hugging_face_embeddings()

    pc = Pinecone(api_key=PINECONE_API_KEY)

    index_name = "medicalbot"

    # Check if index already exists before creating
    existing_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    # Embed each chunk and upsert the embeddings into your Pinecone index.
    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=embeddings
    )
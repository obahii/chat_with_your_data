from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import hashlib


def create_vector_store(
    file_content: bytes, filename: str, vector_store_path: str
) -> tuple[str, str]:
    """
    Creates a vector store from a PDF document using HuggingFace embeddings.

    Args:
        file_content (bytes): Content of the uploaded PDF file
        filename (str): Original filename of the uploaded document
        vector_store_path (str): Base path where to store the vectors

    Returns:
        tuple[str, str]: (collection_name, vector_store_path)
    """
    try:
        # Create a temporary file to save the uploaded content
        temp_file = "temp.pdf"
        with open(temp_file, "wb") as f:
            f.write(file_content)

        # Generate a unique collection name based on file content
        file_hash = hashlib.md5(file_content).hexdigest()[:10]
        collection_name = f"doc_{file_hash}"

        # Create the specific vector store path for this document
        doc_vector_store_path = os.path.join(vector_store_path, collection_name)
        os.makedirs(doc_vector_store_path, exist_ok=True)

        # Create loader and load the PDF
        loader = PyPDFLoader(temp_file)
        pages = loader.load_and_split()

        # Initialize HuggingFace embeddings
        embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create vector store from documents
        vectordb = Chroma.from_documents(
            documents=pages,
            embedding=embedding_func,
            persist_directory=doc_vector_store_path,
            collection_name=collection_name,
        )

        # Clean up temporary file
        os.remove(temp_file)

        return collection_name, doc_vector_store_path

    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise e

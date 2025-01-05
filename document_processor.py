from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pyspark.ml.feature import Tokenizer
from langchain_core.documents import Document
import os
import hashlib
import tempfile
from spark_config import create_spark_session, cleanup_spark


def create_vector_store(
    file_content: bytes, filename: str, vector_store_path: str
) -> tuple[str, str]:
    """
    Creates a vector store from a PDF document using HuggingFace embeddings and PySpark for processing.
    """
    spark = create_spark_session()
    temp_path = None

    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name

        # Generate collection name from file hash
        file_hash = hashlib.md5(file_content).hexdigest()[:10]
        collection_name = f"doc_{file_hash}"

        # Create vector store path
        doc_vector_store_path = os.path.join(vector_store_path, collection_name)
        os.makedirs(doc_vector_store_path, exist_ok=True)

        # Load PDF and split into pages
        loader = PyPDFLoader(temp_path)
        pages = loader.load_and_split()

        # Convert pages to Spark DataFrame
        pages_data = [(page.page_content, page.metadata) for page in pages]
        pages_df = spark.createDataFrame(pages_data, ["content", "metadata"])

        # Tokenize content
        tokenizer = Tokenizer(inputCol="content", outputCol="tokens")
        tokenized_df = tokenizer.transform(pages_df)

        # Convert back to Langchain document format
        processed_pages = []
        for row in tokenized_df.collect():
            processed_pages.append(
                Document(page_content=row["content"], metadata=row["metadata"])
            )

        # Initialize HuggingFace embeddings
        embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create vector store
        vectordb = Chroma.from_documents(
            documents=processed_pages,
            embedding=embedding_func,
            persist_directory=doc_vector_store_path,
            collection_name=collection_name,
        )

        return collection_name, doc_vector_store_path

    except Exception as e:
        raise e
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        cleanup_spark(spark)

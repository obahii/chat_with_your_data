from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, struct
from pyspark.sql.types import ArrayType, FloatType, StringType, StructType, StructField
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
import hashlib
import tempfile
from typing import List, Tuple
import numpy as np


class DistributedDocumentProcessor:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Register UDF for generating embeddings
        @udf(returnType=ArrayType(FloatType()))
        def generate_embedding(text):
            # Note: The embedding model needs to be available on all workers
            embeddings = self.embedding_model.embed_query(text)
            return embeddings.tolist()

        self.generate_embedding_udf = generate_embedding

        # Define schema for document chunks
        self.chunk_schema = StructType(
            [
                StructField("content", StringType(), True),
                StructField("page_num", StringType(), True),
                StructField("chunk_num", StringType(), True),
            ]
        )

    def process_document(
        self, file_content: bytes, filename: str, vector_store_path: str
    ) -> Tuple[str, str]:
        """
        Processes a document using distributed Spark operations
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name

        try:
            # Generate collection name
            file_hash = hashlib.md5(file_content).hexdigest()[:10]
            collection_name = f"doc_{file_hash}"

            # Create vector store path
            doc_vector_store_path = os.path.join(vector_store_path, collection_name)
            os.makedirs(doc_vector_store_path, exist_ok=True)

            # Load and split document into pages
            loader = PyPDFLoader(temp_path)
            pages = loader.load_and_split()

            # Convert pages to Spark DataFrame
            pages_data = [
                (page.page_content, page.metadata["page"], idx)
                for idx, page in enumerate(pages)
            ]
            pages_df = self.spark.createDataFrame(
                pages_data, ["content", "page_num", "chunk_num"]
            )

            # Generate embeddings in parallel
            embedded_df = pages_df.withColumn(
                "embeddings", self.generate_embedding_udf(col("content"))
            )

            # Collect results and convert to Document format
            processed_docs = []
            for row in embedded_df.collect():
                doc = Document(
                    page_content=row["content"],
                    metadata={
                        "page": row["page_num"],
                        "chunk": row["chunk_num"],
                        "embeddings": row["embeddings"],
                    },
                )
                processed_docs.append(doc)

            # Create vector store
            vectordb = Chroma.from_documents(
                documents=processed_docs,
                embedding=self.embedding_model,
                persist_directory=doc_vector_store_path,
                collection_name=collection_name,
            )

            return collection_name, doc_vector_store_path

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def batch_process_documents(
        self, documents: List[Tuple[bytes, str]], vector_store_path: str
    ) -> List[Tuple[str, str]]:
        """
        Process multiple documents in parallel using Spark
        """
        # Convert documents to RDD for parallel processing
        docs_rdd = self.spark.sparkContext.parallelize(documents)

        def process_single_doc(doc_tuple):
            file_content, filename = doc_tuple
            processor = DistributedDocumentProcessor(self.spark)
            return processor.process_document(file_content, filename, vector_store_path)

        # Process documents in parallel
        results = docs_rdd.map(process_single_doc).collect()
        return results

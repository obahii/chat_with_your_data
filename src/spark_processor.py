from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StringType, ArrayType, FloatType, StructType, StructField
import pandas as pd
import numpy as np
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile


class SparkDocumentProcessor:
    def __init__(self, spark_master="local[*]"):
        """Initialize Spark session."""
        self.spark = (
            SparkSession.builder.appName("DistributedRAG")
            .master(spark_master)
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.python.worker.reuse", "true")
            .getOrCreate()
        )

    def _batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts on the driver."""
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return [embedding_model.embed_query(text) for text in texts]

    def process_pdf(self, file_content: bytes) -> Dict[str, any]:
        """Process PDF file using Spark for distributed computation."""
        # Save content to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(file_content)
            temp_path = temp_pdf.name

        try:
            # Load and split document
            loader = PyPDFLoader(temp_path)
            pages = loader.load_and_split()

            # Create DataFrame from pages
            pages_data = [(str(i), page.page_content) for i, page in enumerate(pages)]
            schema = StructType(
                [
                    StructField("chunk_id", StringType(), False),
                    StructField("content", StringType(), False),
                ]
            )

            df = self.spark.createDataFrame(pages_data, schema)

            # Collect content to driver for embedding generation
            # This is done in batches to handle memory efficiently
            BATCH_SIZE = 10
            all_chunks = []

            for batch in range(0, df.count(), BATCH_SIZE):
                batch_df = df.limit(BATCH_SIZE).offset(batch)
                batch_data = batch_df.collect()

                texts = [row.content for row in batch_data]
                embeddings = self._batch_generate_embeddings(texts)

                for row, emb in zip(batch_data, embeddings):
                    all_chunks.append(
                        {
                            "chunk_id": row.chunk_id,
                            "content": row.content,
                            "embeddings": emb,
                        }
                    )

            return {"num_chunks": len(all_chunks), "chunks": all_chunks}

        finally:
            os.unlink(temp_path)

    def similarity_search(
        self,
        query_embedding: List[float],
        document_embeddings: List[Dict],
        top_k: int = 3,
    ) -> List[Dict]:
        """Perform distributed similarity search."""
        # Create DataFrame of document embeddings
        embeddings_data = [
            (chunk["chunk_id"], chunk["content"], chunk["embeddings"])
            for chunk in document_embeddings
        ]

        schema = StructType(
            [
                StructField("chunk_id", StringType(), False),
                StructField("content", StringType(), False),
                StructField("embeddings", ArrayType(FloatType()), False),
            ]
        )

        df = self.spark.createDataFrame(embeddings_data, schema)

        # Define pandas UDF for similarity computation
        @pandas_udf(FloatType())
        def compute_similarity(embeddings_series):
            def _compute_single_similarity(doc_embedding):
                return float(np.dot(query_embedding, doc_embedding))

            return pd.Series(embeddings_series.apply(_compute_single_similarity))

        # Compute similarities using pandas UDF
        df_with_scores = df.withColumn(
            "similarity_score", compute_similarity(col("embeddings"))
        )

        # Get top-k results
        top_results = (
            df_with_scores.orderBy("similarity_score", ascending=False)
            .limit(top_k)
            .collect()
        )

        return [
            {
                "chunk_id": row.chunk_id,
                "content": row.content,
                "score": float(row.similarity_score),
            }
            for row in top_results
        ]

    def shutdown(self):
        """Stop Spark session."""
        self.spark.stop()

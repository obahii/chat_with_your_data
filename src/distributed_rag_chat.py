from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType, StringType, StructType, StructField
from typing import List, Dict
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
import json


class DistributedRAGChat:
    def __init__(
        self, vector_store_path: str, collection_name: str, spark: SparkSession
    ):
        self.spark = spark
        self.embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = Chroma(
            persist_directory=vector_store_path,
            collection_name=collection_name,
            embedding_function=self.embedding_func,
        )

        self.memory = ConversationBufferMemory(
            return_messages=True, memory_key="chat_history"
        )

        # Define schema for LLM processing
        self.llm_schema = StructType(
            [
                StructField("context", StringType(), True),
                StructField("question", StringType(), True),
                StructField("response", StringType(), True),
            ]
        )

        # Register UDF for Ollama inference
        @pandas_udf(StringType())
        def ollama_inference(contexts, questions):
            def process_single_query(context, question):
                prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say so.

                Context: {context}

                Question: {question}
                
                Answer:"""

                # Make request to Ollama API
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "llama2", "prompt": prompt, "stream": False},
                )
                return response.json()["response"]

            # Process queries in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(process_single_query, contexts, questions))

            return pd.Series(results)

        self.ollama_inference_udf = ollama_inference

        # Register UDF for similarity computation
        @udf(returnType=FloatType())
        def compute_similarity(v1, v2):
            return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        self.compute_similarity_udf = compute_similarity

    def distributed_similarity_search(
        self, query_embedding: List[float], top_k: int = 3
    ):
        """
        Perform similarity search using Spark for parallel processing
        """
        # Convert vector store documents to Spark DataFrame
        docs_data = [
            (doc.page_content, doc.metadata["embeddings"])
            for doc in self.vector_db.get()
        ]
        docs_df = self.spark.createDataFrame(docs_data, ["content", "embeddings"])

        # Compute similarities in parallel
        similarities_df = docs_df.withColumn(
            "similarity",
            self.compute_similarity_udf(
                col("embeddings"), self.spark.sparkContext.broadcast(query_embedding)
            ),
        )

        # Get top_k results
        top_results = (
            similarities_df.orderBy(col("similarity").desc()).limit(top_k).collect()
        )

        return [doc["content"] for doc in top_results]

    def batch_process_questions(
        self, questions: List[str], contexts: List[str]
    ) -> List[str]:
        """
        Process multiple questions in parallel using distributed Ollama inference
        """
        # Create DataFrame for batch processing
        queries_df = self.spark.createDataFrame(
            zip(contexts, questions), ["context", "question"]
        )

        # Process queries in parallel
        results_df = queries_df.withColumn(
            "response", self.ollama_inference_udf(col("context"), col("question"))
        )

        # Collect results
        return [row["response"] for row in results_df.collect()]

    def ask(self, question: str) -> str:
        """
        Process a question using distributed similarity search and LLM inference
        """
        # Generate question embedding
        question_embedding = self.embedding_func.embed_query(question)

        # Get relevant documents using distributed search
        relevant_contexts = self.distributed_similarity_search(question_embedding)

        # Combine contexts
        combined_context = "\n\n".join(relevant_contexts)

        # Process question with distributed LLM
        responses = self.batch_process_questions([question], [combined_context])
        final_response = responses[0]

        # Update conversation memory
        self.memory.save_context({"input": question}, {"output": final_response})

        return final_response

    def batch_ask(self, questions: List[str]) -> List[str]:
        """
        Process multiple questions in parallel
        """
        # Generate embeddings for all questions
        question_embeddings = [self.embedding_func.embed_query(q) for q in questions]

        # Get relevant contexts for each question
        all_contexts = []
        for embedding in question_embeddings:
            contexts = self.distributed_similarity_search(embedding)
            all_contexts.append("\n\n".join(contexts))

        # Process all questions in parallel
        responses = self.batch_process_questions(questions, all_contexts)

        # Update memory with all interactions
        for question, response in zip(questions, responses):
            self.memory.save_context({"input": question}, {"output": response})

        return responses

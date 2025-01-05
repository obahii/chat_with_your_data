import os
import json
from typing import List, Dict
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from concurrent.futures import ThreadPoolExecutor
from spark_processor import SparkDocumentProcessor


class DistributedRAGChat:
    def __init__(
        self,
        collection_path: str,
        collection_name: str,
        spark_processor: SparkDocumentProcessor,
    ):
        """Initialize distributed RAG system."""
        self.collection_path = collection_path
        self.collection_name = collection_name
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.spark_processor = spark_processor

        # Load document chunks and embeddings
        self.chunks = self._load_chunks()

        # Initialize multiple Ollama instances
        self.llm_instances = [
            OllamaLLM(model="llama2", temperature=0)
            for _ in range(3)  # Create 3 instances
        ]

        self.memory = ConversationBufferMemory(
            return_messages=True, memory_key="chat_history"
        )

    def _load_chunks(self) -> List[Dict]:
        """Load processed chunks from storage."""
        chunks_file = os.path.join(
            self.collection_path, f"{self.collection_name}_chunks.json"
        )
        with open(chunks_file, "r") as f:
            return json.load(f)

    def _process_chunk_with_llm(
        self, context: str, question: str, llm: OllamaLLM
    ) -> str:
        """Process a single chunk with LLM."""
        prompt = f"""Based on the following context, answer the question.
        If the context doesn't contain relevant information, say so.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""

        return llm.invoke(prompt)

    def ask(self, question: str) -> str:
        """Process question using distributed computation."""
        # Generate question embedding
        question_embedding = self.embedding_model.embed_query(question)

        # Perform similarity search using the existing spark processor
        relevant_chunks = self.spark_processor.similarity_search(
            question_embedding, self.chunks
        )

        # Process chunks in parallel using multiple Ollama instances
        with ThreadPoolExecutor(max_workers=len(self.llm_instances)) as executor:
            futures = [
                executor.submit(
                    self._process_chunk_with_llm, chunk["content"], question, llm
                )
                for chunk, llm in zip(relevant_chunks, self.llm_instances)
            ]

            responses = [future.result() for future in futures]

        # Combine responses
        combined_response = " ".join(responses)

        # Update conversation memory
        self.memory.save_context({"input": question}, {"output": combined_response})

        return combined_response

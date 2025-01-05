from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings


class RAGChat:
    def __init__(self, vector_store_path: str, collection_name: str):
        self.embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = Chroma(
            persist_directory=vector_store_path,
            collection_name=collection_name,
            embedding_function=self.embedding_func,
        )

        self.llm = OllamaLLM(model="llama2", temperature=0)
        self.memory = ConversationBufferMemory(
            return_messages=True, memory_key="chat_history"
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            memory=self.memory,
            retriever=self.vector_db.as_retriever(
                search_kwargs={"fetch_k": 4, "k": 3}, search_type="mmr"
            ),
            chain_type="refine",
        )

    def ask(self, question: str) -> str:
        response = self.qa_chain.invoke({"question": question})
        return response.get("answer")

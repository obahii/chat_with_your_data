from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import os

class RAGDocumentQA:
    def __init__(self):
        """
        Initialize the RAG-based DocumentQA system using local models.
        """
        # Initialize text splitter with smaller chunks for better retrieval
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        # Initialize language model
        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Create pipeline
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Initialize language model
        self.llm = HuggingFacePipeline(pipeline=pipe)
        self.vector_store = None
        
        # Create RAG prompt template
        self.rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Use the following context to answer the question. 
            If you can't find the answer in the context, say "I cannot find the answer in the document."
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
        )
        
        # Initialize LLM Chain
        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=self.rag_prompt,
            verbose=False
        )
        
    def load_pdf(self, pdf_path):
        """
        Load and process a PDF document.
        Args:
            pdf_path (str): Path to the PDF file
        """
        try:
            # Load the PDF
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # Split text into chunks
            texts = self.text_splitter.split_documents(pages)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            
            return True
        except Exception as e:
            print(f"Error loading PDF: {str(e)}")
            return False
    
    def ask_question(self, question):
        """
        Ask a question about the loaded document using RAG.
        Args:
            question (str): Question to ask about the document
        Returns:
            str: Answer to the question
        """
        if not self.vector_store:
            return "Please load a PDF document first."
        
        try:
            # Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(question, k=3)
            
            # Combine relevant document content
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # Get answer using RAG
            response = self.llm_chain.invoke({
                "context": context,
                "question": question
            })
            
            return response["text"]
            
        except Exception as e:
            return f"Error processing question: {str(e)}"

def main():
    print("RAG-based PDF Q&A System")
    print("-----------------------")
    print("Initializing models (this may take a few minutes)...")
    
    # Initialize system
    qa_system = RAGDocumentQA()
    print("System ready!")
    
    while True:
        print("\nOptions:")
        print("1. Load PDF document")
        print("2. Ask question")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            pdf_path = input("Enter PDF file path: ")
            if qa_system.load_pdf(pdf_path):
                print("PDF document loaded successfully!")
            else:
                print("Failed to load PDF document.")
                
        elif choice == "2":
            question = input("Enter your question: ")
            print("\nProcessing question...")
            answer = qa_system.ask_question(question)
            print("\nAnswer:", answer)
            
        elif choice == "3":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
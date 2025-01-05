# main.py
import streamlit as st
from document_loading_splitting import create_vector_store
from rag_function import RAGChat
import os

# Set page config
st.set_page_config(
    page_title="Document Chat",
    page_icon="📚",
    layout="wide",
)

# Simple styling for chat messages
st.markdown("""
    <style>
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border-left: 3px solid #e5e7eb;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "conversations" not in st.session_state:
    st.session_state.conversations = {}  # Store RAGChat instances
if "current_doc" not in st.session_state:
    st.session_state.current_doc = None
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}
if "doc_names" not in st.session_state:
    st.session_state.doc_names = {}  # Store original filenames

# Sidebar for file upload and document selection
with st.sidebar:
    st.title("📚 Document Chat")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    if uploaded_file:
        # Process the uploaded file
        try:
            file_content = uploaded_file.read()
            vector_store_path = "vector_stores"

            with st.spinner("Processing document..."):
                collection_name, doc_path = create_vector_store(
                    file_content, uploaded_file.name, vector_store_path
                )
                # Store the original filename
                st.session_state.doc_names[collection_name] = uploaded_file.name

                if collection_name not in st.session_state.conversations:
                    st.session_state.conversations[collection_name] = RAGChat(
                        doc_path, collection_name
                    )
                    st.session_state.chat_histories[collection_name] = []
                    st.success("Document processed successfully!")
        except FileNotFoundError:
            st.error("Error: Could not find or access the uploaded file.")
        except Exception as e:
            st.error(f"An error occurred while processing the document: {str(e)}")
            # Clean up any partially created resources if necessary
            if (
                "collection_name" in locals()
                and collection_name in st.session_state.doc_names
            ):
                del st.session_state.doc_names[collection_name]
            if (
                "collection_name" in locals()
                and collection_name in st.session_state.conversations
            ):
                del st.session_state.conversations[collection_name]
            if (
                "collection_name" in locals()
                and collection_name in st.session_state.chat_histories
            ):
                del st.session_state.chat_histories[collection_name]

    # Document selection
    if st.session_state.conversations:
        st.subheader("Select Document")
        docs = list(st.session_state.conversations.keys())
        doc_options = {st.session_state.doc_names[doc]: doc for doc in docs}
        selected_name = st.selectbox(
            "Choose a document to chat with",
            list(doc_options.keys()),
            index=list(doc_options.values()).index(st.session_state.current_doc) if st.session_state.current_doc else 0
        )
        current_doc = doc_options[selected_name]
        st.session_state.current_doc = current_doc

# Main chat interface
if st.session_state.current_doc:
    st.header(f"Chatting with: {st.session_state.doc_names[st.session_state.current_doc]}")
    
    # Display chat history
    for message in st.session_state.chat_histories[st.session_state.current_doc]:
        with st.container():
            st.markdown(
                f"""<div class="chat-message">
                    <strong>{'🤖 Assistant' if message['role'] == 'assistant' else '👤 You'}:</strong><br>
                    {message['content']}
                </div>""",
                unsafe_allow_html=True
            )
    
    # Chat input
    user_input = st.chat_input("Ask a question about the document")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_histories[st.session_state.current_doc].append(
            {"role": "user", "content": user_input}
        )
        
        # Get AI response
        with st.spinner("Thinking..."):
            rag_chat = st.session_state.conversations[st.session_state.current_doc]
            response = rag_chat.ask(user_input)
            
            # Add AI response to history
            st.session_state.chat_histories[st.session_state.current_doc].append(
                {"role": "assistant", "content": response}
            )
        
        # Rerun to update the chat display
        st.rerun()
else:
    st.info("👈 Please upload a PDF document to start chatting!")

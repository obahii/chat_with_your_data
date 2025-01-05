import streamlit as st
from spark_processor import SparkDocumentProcessor
from distributed_rag import DistributedRAGChat
import os
import json
import hashlib
import atexit

# Set page config
st.set_page_config(
    page_title="Distributed Document Chat",
    page_icon="📚",
    layout="wide",
)


# Initialize Spark processor
@st.cache_resource
def get_spark_processor():
    processor = SparkDocumentProcessor()
    # Ensure Spark session is properly stopped when the app exits
    atexit.register(processor.shutdown)
    return processor


# Get or create Spark processor
spark_processor = get_spark_processor()

# Initialize session state
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_doc" not in st.session_state:
    st.session_state.current_doc = None
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}
if "doc_names" not in st.session_state:
    st.session_state.doc_names = {}


def process_document(file_content: bytes, filename: str) -> tuple[str, str]:
    """Process document using distributed computation."""
    # Generate unique collection name
    file_hash = hashlib.md5(file_content).hexdigest()[:10]
    collection_name = f"doc_{file_hash}"

    # Create collection directory
    collection_path = os.path.join("vector_stores", collection_name)
    os.makedirs(collection_path, exist_ok=True)

    # Process document using the existing Spark processor
    processed_data = spark_processor.process_pdf(file_content)

    # Save processed chunks
    chunks_file = os.path.join(collection_path, f"{collection_name}_chunks.json")
    with open(chunks_file, "w") as f:
        json.dump(processed_data["chunks"], f)

    return collection_name, collection_path


# Sidebar for file upload and document selection
with st.sidebar:
    st.title("📚 Distributed Document Chat")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    if uploaded_file:
        try:
            file_content = uploaded_file.read()

            with st.spinner("Processing document using distributed computation..."):
                collection_name, collection_path = process_document(
                    file_content, uploaded_file.name
                )

                # Store filename and initialize conversation
                st.session_state.doc_names[collection_name] = uploaded_file.name
                if collection_name not in st.session_state.conversations:
                    st.session_state.conversations[collection_name] = (
                        DistributedRAGChat(
                            collection_path, collection_name, spark_processor
                        )
                    )
                    st.session_state.chat_histories[collection_name] = []

                st.success("Document processed successfully!")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            # Clean up if necessary
            if "collection_name" in locals():
                for state_dict in [
                    st.session_state.doc_names,
                    st.session_state.conversations,
                    st.session_state.chat_histories,
                ]:
                    if collection_name in state_dict:
                        del state_dict[collection_name]

    # Document selection
    if st.session_state.conversations:
        st.subheader("Select Document")
        docs = list(st.session_state.conversations.keys())
        doc_options = {st.session_state.doc_names[doc]: doc for doc in docs}
        selected_name = st.selectbox(
            "Choose a document to chat with",
            list(doc_options.keys()),
            index=(
                list(doc_options.values()).index(st.session_state.current_doc)
                if st.session_state.current_doc
                else 0
            ),
        )
        st.session_state.current_doc = doc_options[selected_name]

# Main chat interface
if st.session_state.current_doc:
    st.header(
        f"Chatting with: {st.session_state.doc_names[st.session_state.current_doc]}"
    )

    # Display chat history
    for message in st.session_state.chat_histories[st.session_state.current_doc]:
        with st.container():
            role_icon = "🤖" if message["role"] == "assistant" else "👤"
            st.markdown(
                f"""<div class="chat-message">
                    <strong>{role_icon} {'Assistant' if message['role'] == 'assistant' else 'You'}:</strong><br>
                    {message['content']}
                </div>""",
                unsafe_allow_html=True,
            )

    # Chat input
    user_input = st.chat_input("Ask a question about the document")

    if user_input:
        # Add user message to history
        st.session_state.chat_histories[st.session_state.current_doc].append(
            {"role": "user", "content": user_input}
        )

        # Get distributed AI response
        with st.spinner("Processing with distributed computation..."):
            rag_chat = st.session_state.conversations[st.session_state.current_doc]
            try:
                response = rag_chat.ask(user_input)

                # Add AI response to history
                st.session_state.chat_histories[st.session_state.current_doc].append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")

        st.rerun()
else:
    st.info("👈 Please upload a PDF document to start chatting!")

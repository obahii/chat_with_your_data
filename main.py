import streamlit as st
from document_processor import create_vector_store
from rag_chat import RAGChat
from styles import get_styles
from session_state import initialize_session_state
import os


def main():
    # Set page config
    st.set_page_config(
        page_title="Document Chat",
        page_icon="📚",
        layout="wide",
    )

    # Apply styles
    st.markdown(get_styles(), unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state(st)

    # Sidebar for file upload and document selection
    with st.sidebar:
        st.title("📚 Document Chat")
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

        if uploaded_file:
            try:
                file_content = uploaded_file.read()
                vector_store_path = "vector_stores"

                with st.spinner("Processing document..."):
                    collection_name, doc_path = create_vector_store(
                        file_content, uploaded_file.name, vector_store_path
                    )
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

        # Document list selection
        if st.session_state.conversations:
            st.subheader("Available Documents")
            docs = list(st.session_state.conversations.keys())

            for doc in docs:
                doc_name = st.session_state.doc_names[doc]
                is_selected = doc == st.session_state.current_doc

                if st.button(
                    doc_name,
                    key=f"doc_{doc}",
                    help=f"Click to chat with {doc_name}",
                    use_container_width=True,
                    type="secondary" if is_selected else "primary",
                ):
                    st.session_state.current_doc = doc
                    st.rerun()

    # Main chat interface
    if st.session_state.current_doc:
        st.header(
            f"Chatting with: {st.session_state.doc_names[st.session_state.current_doc]}"
        )

        # Display chat history
        for message in st.session_state.chat_histories[st.session_state.current_doc]:
            with st.container():
                st.markdown(
                    f"""<div class="chat-message">
                        <strong>{'🤖 Assistant' if message['role'] == 'assistant' else '👤 You'}:</strong><br>
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
        st.info(
            "👈 Please upload a PDF document and select it from the sidebar to start chatting!"
        )


if __name__ == "__main__":
    main()

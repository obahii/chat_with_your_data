def initialize_session_state(st):
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "current_doc" not in st.session_state:
        st.session_state.current_doc = None
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    if "doc_names" not in st.session_state:
        st.session_state.doc_names = {}

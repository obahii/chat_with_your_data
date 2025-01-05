def get_styles():
    return """
    <style>
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border-left: 3px solid #e5e7eb;
        }
        .doc-list-item {
            cursor: pointer;
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 0.25rem;
            transition: background-color 0.2s;
        }
        .doc-list-item:hover {
            background-color: #f3f4f6;
        }
        .doc-list-item.selected {
            background-color: #e5e7eb;
        }
    </style>
    """

# PDF Document Chat System

A distributed computing-based PDF chat system that allows users to upload PDF documents and interact with their content using natural language queries. The system uses PySpark for distributed processing and implements a RAG (Retrieval Augmented Generation) approach with LLama2.
## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Setup Steps](#setup-steps)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [Troubleshooting](#troubleshooting)
## Prerequisites

Before running the project, ensure you have the following installed:
- Python 3.8 or higher
- Java 8 or higher (required for PySpark)
- pip (Python package installer)

## Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd pdf-document-chat
```
2. Create and activate a virtual environment:
```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For Linux/Mac
python3 -m venv venv
source venv/bin/activate
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

pdf-document-chat/
├── documents/
├── src
    ├── distributed_rag.py
    ├── spark_processor.py
    ├── main.py
    ├── vector_stores/
├── requirements.txt
└── README.md

## Setup Steps
1. Install Ollama (required for LLama2):
Check the official website: [here](https://ollama.com/download)

2. Pull the LLama2 model:
```bash
ollama pull llama2
```

3. Install Java (if not already installed):
```bash
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install default-jdk

# For Mac
brew install java
```

4. Install the requirements:
```bash
pip install -r requirements.txt
```

## Running the Application
1. Start the Ollama service:
```bash
ollama serve
```

2. Run the Streamlit application:
```bash
streamlit run main.py
```

3.Access the application in your web browser at:
``` http://localhost:8501```

## Usage Guide
1. Upload Documents

- Click the "Upload a PDF document" button in the sidebar
- Select a PDF file from your computer
- Wait for the processing to complete

2. Select Documents

- Click on a document from the list in the sidebar
- The chat interface will appear in the main area

3. Chat with Documents

- Type your question in the chat input
- Press Enter or click the send button
- Wait for the response
- Continue the conversation as needed

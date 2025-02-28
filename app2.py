import os
import json  # ✅ Fix: Ensure JSON library is imported
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_text_splitters import TokenTextSplitter

# Load environment variables
load_dotenv()

# Fetch secrets from Streamlit Cloud OR environment variables
AZURE_OPENAI_KEY = st.secrets.get("AZURE_OPENAI_KEY", os.getenv("AZURE_OPENAI_KEY"))
AZURE_OPENAI_ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT"))
AZURE_DEPLOYMENT_NAME = st.secrets.get("AZURE_DEPLOYMENT_NAME", os.getenv("AZURE_DEPLOYMENT_NAME"))

class JioPayChatbot:
    def __init__(self):
        """Initialize the chatbot with Azure OpenAI."""
        try:
            self.embeddings = AzureOpenAIEmbeddings(
                api_key=AZURE_OPENAI_KEY,  # ✅ Fix: Use `api_key` instead of `azure_openai_key`
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                deployment=AZURE_DEPLOYMENT_NAME,
            )
            self.text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=100)
            self.vector_store = None
            self.qa_chain = None
        except Exception as e:
            st.error(f"❌ Error initializing embeddings: {str(e)}")

    def create_knowledge_base(self):
        """Load pre-scraped JSON data + Markdown file"""
        documents = []

        # Load Markdown file
        try:
            with open("jiopay_content1.md", "r", encoding="utf-8") as f:
                md_content = f.read()
                documents.append({"url": "file://jiopay_content1.md", "content": md_content})
        except Exception as e:
            st.error(f"❌ Error loading Markdown file: {e}")

        # Load JSON pre-scraped data
        try:
            with open("scraped_data1.json", "r", encoding="utf-8") as f:
                scraped_data = json.load(f)  # ✅ Fix: JSON library now imported
                documents.extend(scraped_data)
        except Exception as e:
            st.error(f"❌ Error loading JSON: {e}")

        texts = [doc["content"] for doc in documents]
        metadatas = [{"source": doc["url"]} for doc in documents]

        docs = self.text_splitter.create_documents(texts, metadatas=metadatas)
        self.vector_store = FAISS.from_documents(docs, self.embeddings)

    def initialize_qa(self):
        """Initialize Retrieval-Augmented Generation (RAG) with Azure OpenAI"""
        try:
            llm = AzureOpenAI(
                api_key=AZURE_OPENAI_KEY,  # ✅ Fix: Use `api_key`
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                deployment=AZURE_DEPLOYMENT_NAME,
                temperature=0.7,
                max_tokens=512,
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(),
                return_source_documents=True,
            )
        except Exception as e:
            st.error(f"❌ Error initializing QA model: {str(e)}")

    def ask(self, question: str) -> str:
        """Process user query with RAG pipeline"""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized")

        result = self.qa_chain.invoke({"query": question})
        sources = list(set([doc.metadata["source"] for doc in result["source_documents"]]))
        return f"{result['result']}\n\nSources: {sources}"

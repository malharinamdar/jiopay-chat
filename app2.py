import json
import os
from langchain_openai import AzureOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_text_splitters import TokenTextSplitter
from typing import List, Dict
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
try:
    os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"]
    os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_OPENAI_ENDPOINT"]
    os.environ["AZURE_DEPLOYMENT_NAME"] = st.secrets["AZURE_DEPLOYMENT_NAME"]
except (AttributeError, KeyError):
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    os.environ["AZURE_DEPLOYMENT_NAME"] = os.getenv("AZURE_DEPLOYMENT_NAME")

class JioPayChatbot:
    def __init__(self):
        """Initialize the chatbot with Azure OpenAI."""
        self.embeddings = OpenAIEmbeddings(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            deployment_name=os.environ["AZURE_DEPLOYMENT_NAME"]
        )
        self.text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=100)
        self.vector_store = None
        self.qa_chain = None

    def create_knowledge_base(self):
        """Load pre-scraped JSON data + Markdown file"""
        documents = []
        
        # Load Markdown file
        try:
            with open('jiopay_content1.md', 'r', encoding='utf-8') as f:
                md_content = f.read()
                documents.append({
                    "url": "file://jiopay_content1.md",
                    "content": md_content
                })
        except Exception as e:
            print(f"❌ Error loading Markdown file: {e}")

        # Load JSON pre-scraped data
        try:
            with open("scraped_data1.json", "r", encoding="utf-8") as f:
                scraped_data = json.load(f)
                documents.extend(scraped_data)
                print("✅ Loaded scraped data from `scraped_data1.json`")
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")

        texts = [doc["content"] for doc in documents]
        metadatas = [{"source": doc["url"]} for doc in documents]
        
        docs = self.text_splitter.create_documents(texts, metadatas=metadatas)
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
    
    def initialize_qa(self):
        """Initialize RAG with Azure OpenAI"""
        llm = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            deployment_name=os.environ["AZURE_DEPLOYMENT_NAME"],
            temperature=0.7,
            max_tokens=512
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True
        )

    def ask(self, question: str) -> str:
        """Process user query with RAG pipeline"""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized")
        
        result = self.qa_chain.invoke({"query": question})
        sources = list(set([doc.metadata["source"] for doc in result["source_documents"]]))
        return f"{result['result']}\n\nSources: {sources}"

def main():
    chatbot = JioPayChatbot()
    chatbot.create_knowledge_base()
    chatbot.initialize_qa()

    print("Chatbot Ready!")
    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break
        answer = chatbot.ask(question)
        print(f"Assistant: {answer}")

if __name__ == "__main__":
    main()

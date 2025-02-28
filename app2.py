import json
from langchain_openai import AzureOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_text_splitters import TokenTextSplitter
import streamlit as st

class JioPayChatbot:
    def __init__(self):
        """Initialize the chatbot with Azure OpenAI."""
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=st.secrets["AZURE_EMBEDDINGS_DEPLOYMENT"],
            openai_api_version=st.secrets["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
            openai_api_key=st.secrets["AZURE_OPENAI_KEY"],
            openai_api_type="azure"
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
            st.error(f"Error loading Markdown file: {e}")

        # Load JSON pre-scraped data
        try:
            with open("scraped_data1.json", 'r', encoding='utf-8') as f:
                scraped_data = json.load(f)
                documents.extend(scraped_data)
        except Exception as e:
            st.error(f"Error loading JSON: {e}")

        texts = [doc["content"] for doc in documents]
        metadatas = [{"source": doc["url"]} for doc in documents]
        
        docs = self.text_splitter.create_documents(texts, metadatas=metadatas)
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
    
    def initialize_qa(self):
        """Initialize RAG with Azure OpenAI"""
        llm = AzureOpenAI(
            temperature=0.7,
            max_tokens=800,
            deployment_name=st.secrets["AZURE_DEPLOYMENT_NAME"],
            openai_api_version=st.secrets["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
            openai_api_key=st.secrets["AZURE_OPENAI_KEY"],
            openai_api_type="azure"
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

    def ask(self, question: str) -> str:
        """Process user query with RAG pipeline"""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized")
        
        result = self.qa_chain.invoke({"query": question})
        sources = list(set([doc.metadata["source"] for doc in result["source_documents"]]))
        return {
            "answer": result["result"],
            "sources": sources
        }

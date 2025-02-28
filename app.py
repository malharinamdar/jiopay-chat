import streamlit as st
from app2 import JioPayChatbot

def initialize_session_state():
    """Initialize session state variables"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def setup_chatbot():
    """Initialize the JioPay chatbot"""
    try:
        chatbot = JioPayChatbot()
        chatbot.create_knowledge_base()
        chatbot.initialize_qa()
        return chatbot
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return None

def main():
    initialize_session_state()
    st.title("JioPay Customer Support Assistant")
    st.markdown("Ask me anything about JioPay services!")

    if not st.session_state.initialized:
        with st.spinner("Initializing assistant... (this may take a minute)"):
            st.session_state.chatbot = setup_chatbot()
            if st.session_state.chatbot:
                st.session_state.initialized = True
                if not st.session_state.messages:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Welcome to JioPay Support! How can I help you today?"
                    })

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("View Sources"):
                    st.write(message["sources"])

    if st.session_state.initialized:
        user_input = st.chat_input("Ask your question about JioPay...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your question..."):
                    try:
                        response = st.session_state.chatbot.ask(user_input)
                        answer = response["answer"]
                        sources = response["sources"]
                        
                        st.markdown(answer)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        
                        if sources:
                            with st.expander("Reference Sources"):
                                st.write("Sources used for this answer:")
                                for source in sources:
                                    st.write(f"- {source}")
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()

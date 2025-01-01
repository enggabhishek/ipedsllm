# type: ignore
import os
import streamlit as st
from openai import OpenAI
from langchain_utils import chatbot

class UniversityExplorerAIChatbot:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.openai_api_key)
        self.initialize_session_state()

    def initialize_session_state(self):
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-3.5-turbo"
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def display_chat_history(self):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def accept_user_input(self):
        prompt = st.chat_input("Ask About Institutional Demographics, Graduation and Other Resources")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.spinner("Generating response..."):
                with st.chat_message("assistant"):
                    response = chatbot.invoke_chain(prompt, st.session_state.messages)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    def run(self):
        st.title("University Explorer AI Chatbot")
        self.display_chat_history()
        self.accept_user_input()

if __name__ == "__main__":
    chatbot_app = UniversityExplorerAIChatbot()
    chatbot_app.run()

import os
import streamlit as st
from openai import OpenAI
from langchain_utils import chatbot

class UniversityExplorerChatbot:
    def __init__(self):
        self.openai_api_key = os.getenv("API_KEY")
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

    def handle_user_input(self, prompt):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Generating response..."):
            with st.chat_message("assistant"):
                response = chatbot.invoke_chain(prompt, st.session_state.messages)
                if response is None:
                    response = "There are no results based on the given input. Please try to modify few parameters to retrieve the relevant information."
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    def run(self):
        st.title("University Explorer AI Chatbot")
        self.display_chat_history()
        if prompt := st.chat_input("Ask About Institutional Demographics, Graduation and Other Resources"):
            self.handle_user_input(prompt)

def main():
    chatbot = UniversityExplorerChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()

import os
import streamlit as st
from openai import OpenAI
from langchain_utils import invoke_chain


openai_api_key = os.getenv("API_KEY")

def main():
    st.title("University Explorer AI Chatbot")
    # Set OpenAI API key from Streamlit secrets
    client = OpenAI(api_key=openai_api_key)

    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    # Initialize chat history
    if "messages" not in st.session_state:
        # print("Creating session state")
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask About Institutional Demographics, Graduation and Other Resources"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.spinner("Generating response..."):
            with st.chat_message("assistant"):
                response = invoke_chain(prompt, st.session_state.messages)
                if response is None:
                    response = "There are no results based on the given input. Please try to modify few parameters to retrieve the relevant information."
                st.markdown(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response})


if __name__ == "__main__":
    main()

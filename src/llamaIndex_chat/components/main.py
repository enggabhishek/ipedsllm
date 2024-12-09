import streamlit as st
from llamaindex_utils import query_pipeline
qp = query_pipeline()


def main():
    st.set_page_config(page_title="University Explorer Bot",
                       page_icon=':llama:',
                       layout='centered',
                       initial_sidebar_state='collapsed')

    st.header("University Explorer ChatBot")

    input_text = st.text_input("Search the Information you want")

    submit = st.button("Generate")

    if submit:
        response = qp.run(query=input_text)
        st.write(response.message.content)


if __name__ == "__main__":
    main()

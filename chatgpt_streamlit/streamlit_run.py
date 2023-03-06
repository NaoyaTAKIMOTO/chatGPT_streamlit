import streamlit as st
from streamlit_chat import message
from langchain.llms import OpenAIChat
from langchain import PromptTemplate, LLMChain
# chatGPT
template = """Question: {question}

Answer: Let's think step by step in Japanese.
Final answer is given in Japanese simply."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = OpenAIChat()
llm_chain = LLMChain(prompt=prompt, llm=llm)

##
st.write("Welcome")

if "message_history" not in st.session_state:
    st.session_state.message_history = []

for message_ in st.session_state.message_history:
    if "you:" in message_:
        message(message_, is_user=True) # display all the previous message
    else:
        message(message_) # display all the previous message

placeholder = st.empty() # placeholder for latest message
input_ = st.text_input("you:")
st.session_state.message_history.append("you:"+input_)
if len(input_) > 0:
    input2_ = llm_chain.run(input_)
    st.session_state.message_history.append("AI:"+input2_)

with placeholder.container():
    message( st.session_state.message_history[-1]) # display the latest message   
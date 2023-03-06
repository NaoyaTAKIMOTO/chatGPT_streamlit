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
st.markdown("""
## chatGPT
1. 色々なことを質問してみてください
    1. ガンダムで一番速度のある機体の半分の速さの機体を教えてください。
    2. 〜はpythonでどのように記述するのか具体例を教えてください。
    3. 艦これが流行っていた頃に放映していたアニメを教えてください。
2. 答えに満足できない場合は補足の質問をしてみてください
""")

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
import streamlit as st
from streamlit_chat import message
from langchain.llms import OpenAIChat
from langchain import PromptTemplate, LLMChain, SerpAPIWrapper
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor

# chatGPT
llm = OpenAIChat()
search = SerpAPIWrapper()
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]
prefix = """Answer the following questions as best you can. 
You think step by step.
Finally answer in Japanese like a Mashu Kyrielight.
You have access to the following tools:"""
suffix = """Begin!  
答えは日本語で、FGOのマシュキリエライトのように敬語を使って答えてください。
私のことは先輩と呼んでください。
質問に答える時には根拠も述べてください。

Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "agent_scratchpad"]
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)



##
st.markdown("""
## chatGPT
1. 色々なことを質問してみてください
    1. ガンダムで一番速度のある機体の半分の速さの機体を教えてください。
    2. エクセルからデータを集計するスクリプトはpythonでどのように記述するのか具体例を教えてください。
    3. 艦これが流行っていた頃に放映していたアニメを列挙してください。
2. 答えに満足できない場合は補足の質問をしてみてください
""")

if "message_history" not in st.session_state:
    st.session_state.message_history = []

for message_ in st.session_state.message_history:
    if "you:" in message_:
        message(message_, is_user=True,key=hash(message_)) # display all the previous message
    else:
        message(message_,key=hash(message_)) # display all the previous message

placeholder = st.empty() # placeholder for latest message
input_ = st.text_input("you:")
st.session_state.message_history.append("you:"+input_)
if len(input_) > 0:
    input2_ = agent_executor.run(input_)
    st.session_state.message_history.append("AI:"+input2_)

with placeholder.container():
    message_ = st.session_state.message_history[-1]
    message(message_ ,key=hash(message_)) # display the latest message   
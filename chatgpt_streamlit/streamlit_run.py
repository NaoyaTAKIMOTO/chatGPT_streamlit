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
prefix = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:"""
suffix = """Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Args"

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
    input2_ = agent_executor.run(input_)
    st.session_state.message_history.append("AI:"+input2_)

with placeholder.container():
    message( st.session_state.message_history[-1]) # display the latest message   
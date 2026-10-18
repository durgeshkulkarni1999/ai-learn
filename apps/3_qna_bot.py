from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
import streamlit as st
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

llm = ChatGroq(model="openai/gpt-oss-20b", streaming=True)
search = GoogleSerperAPIWrapper()
tools = [search.run]
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

agent = create_agent(
    model = llm,
    tools = tools,
    system_prompt = "You are amazing ai and can search on google as well if any recent information is required. You are a helpful assistant that can perform Google searches and always use the search tool for any queries.",
    checkpointer = st.session_state.memory
)

st.subheader("Qucik QnA chat bot with google search")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask anything")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})
    response = agent.stream({"messages" : [{"role": "user", "content": query}]}, 
                            {"configurable": {"thread_id" : "1"}},
                            stream_mode = "messages")

    ai_container = st.chat_message("assistant")
    with ai_container:
        space = st.empty()
        message = ""

        for chunk in response:
            message = message + chunk[0].content
            space.write(message)    

    st.session_state.history.append({"role": "assistant", "content": message})
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

loader = PyPDFLoader("../data/medical_report.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_db = InMemoryVectorStore.from_documents(documents=docs, embedding=embeddings)

llm = ChatGroq(model = 'openai/gpt-oss-20b')
@tool
def retrieve_context(query: str):
    """Retrieve documents relevant to the query from the knowledge base."""
    context = ""
    docs = vector_db.similarity_search(query=query,k=3)
    for doc in docs:
        context = doc.page_content + "\n\n"

    return context

system_prompt = """You are a helpful assistant that answers questions using retrieved context. 
        My knowledge base consists of the details from the uploaded document. 
        ALWAYS use the `retrieve_context` tool for questions requiring external knowledge."""

memory = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=[retrieve_context],
    system_prompt=system_prompt,
    checkpointer=memory
)

while True:
    query = input("User: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = agent.invoke({"messages": [{"role": "user", "content": query}]},{"configurable":{"thread_id": 1}})
    result = response["messages"][-1].content
    print("AI: ", result)
import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import asyncio

load_dotenv()
VECTOR_STORE = None

def obtain_chat_model():
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    return llm

def embedding_model():
    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
    return embeddings

def init_vector_store():
    global VECTOR_STORE
    if VECTOR_STORE is None:
        embeddings = embedding_model()
        VECTOR_STORE = InMemoryVectorStore(embeddings)
    return VECTOR_STORE

# def vector_store():
#     embeddings = embedding_model()
#     vs = InMemoryVectorStore(embeddings)
#     return vs

async def obtain_docs(file_path):
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

async def splitting(file_path):
    docs = await obtain_docs(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    return all_splits

async def store_to_vectorDB(file_path):
    all_splits = await splitting(file_path)
    vs = init_vector_store()
    vs.add_documents(documents=all_splits)
    return 

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    vs = init_vector_store()
    retrieved_docs = vs.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm = obtain_chat_model()
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}
    # return {"messages": [AIMessage(content=response.content)]}

def tools(state: MessagesState):
    node =  ToolNode([retrieve])
    return node(state)

def generate(state: MessagesState):
    """Generate answer."""
    llm = obtain_chat_model()
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    # docs_content = "\n\n".join(doc.content for doc in tool_messages)
    docs_content = "\n\n".join(str(msg.content) for msg in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

def define_graph():
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    # graph_builder.add_node(tools)
    graph_builder.add_node(ToolNode([retrieve]), name="tools")
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile()
    return graph

class Chatbot:
    def __init__(self, graph):
        self.graph = graph
        self.state = {"messages": []}  # persistent conversation state

    async def ask(self, user_input: str) -> str:
        """Send a message to the chatbot and get response."""
        # self.state["messages"].append({"role": "user", "content": user_input})
        self.state["messages"].append(HumanMessage(content=user_input))
        async for step in self.graph.astream(self.state, stream_mode="values"):
            self.state = step
        # response = self.state["messages"][-1]["content"]
        # return response
        # Find the last AI message
        for msg in reversed(self.state["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content
        return "Sorry, I couldn't generate a response."
    
async def main(file_path: str = "../Hostel_Affidavit_Men_2024-Chennai_Updated.pdf"):
    global GRAPH
    await store_to_vectorDB(file_path)
    GRAPH = define_graph()

    bot = Chatbot(GRAPH)

    print("Chatbot ready! Type 'exit' to quit.")
    while True:
        user_in = input("You: ")
        if user_in.lower() in {"exit", "quit"}:
            break
        answer = await bot.ask(user_in)
        print("Bot:", answer)

asyncio.run(main())
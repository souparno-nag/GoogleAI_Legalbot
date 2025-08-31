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

def obtain_chat_model():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    return llm

def embedding_model():
    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
    return embeddings

def vector_store():
    embeddings = embedding_model()
    vector_store = InMemoryVectorStore(embeddings)
    return vector_store

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
    document_ids = vector_store.add_documents(documents=all_splits)
    return

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
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

def tools():
    return ToolNode([retrieve])

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
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
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
    graph_builder.add_node(tools)
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
        self.state["messages"].append({"role": "user", "content": user_input})
        async for step in self.graph.astream(self.state, stream_mode="values"):
            pass  # we just iterate to final state
        response = self.state["messages"][-1].content
        return response
    
async def main(file_path):
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
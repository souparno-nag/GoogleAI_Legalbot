import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
import operator
from typing import Annotated, List, Literal, TypedDict
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langchain_community.document_loaders import PyPDFLoader

token_max = 1000

def obtain_chat_model():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    return llm

def define_map_prompt(level: str):
    if (level == "expert"):
        system_msg = (
            "Imagine the reader is a lawyer or a paralegal."
            "Write a detailed summary of the following legal text. "
            "Focus on precise legal terminology, case references, and nuanced interpretations."
        )
    elif (level == "moderate"):
        system_msg = (
            "Imagine the reader is not an expert, but also not a complete layman."
            "Summarize the following legal text for someone with basic legal knowledge. "
            "Explain the main points clearly without excessive jargon, but preserve key legal concepts."
        )
    else:  # beginner
        system_msg = (
            "Explain the following legal text in very simple terms, "
            "as if to someone without legal training. Focus only on the main ideas."
        )
    map_prompt = ChatPromptTemplate.from_messages(
        [("system", system_msg + "\\n\\n{context}")]
    )
    return map_prompt

def reduce(level: str):
    reduce_template = """
    The following is a set of summaries:
    {docs}
    Take these and distill it into a final, consolidated summary
    of the main themes.
    """
    if (level == "expert"):
        reduce_template = """
        The following are section summaries:
        {docs}
        Consolidate them into a rigorous legal summary, 
        highlighting legal arguments, precedents, and implications.
        """
    elif (level == "moderate"):
        reduce_template = """
        The following are section summaries:
        {docs}
        Consolidate them into a clear summary for someone with basic legal knowledge.
        Focus on the main points and legal reasoning without too much jargon.
        """
    else:  # beginner
        reduce_template = """
        The following are section summaries:
        {docs}
        Explain them in plain language for a non-lawyer.
        Keep it simple and focus on the overall meaning.
        """

    reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
    return reduce_prompt

def splitting(docs):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"Generated {len(split_docs)} documents.")
    return split_docs

def length_function(documents: List[Document]) -> int:
    llm = obtain_chat_model()
    """Get number of tokens for input contents."""
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

class OverallState(TypedDict):
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str

class SummaryState(TypedDict):
    content: str

async def generate_summary(state: SummaryState, level: str):
    map_prompt = define_map_prompt(level)
    llm = obtain_chat_model()
    prompt = map_prompt.invoke(state["content"])
    response = await llm.ainvoke(prompt)
    return {"summaries": [response.content]}

def map_summaries(state: OverallState):
    return [
        Send("generate_summary", {"content": content}) for content in state["contents"]
    ]

def collect_summaries(state: OverallState):
    return {
        "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
    }

async def _reduce(input: dict, level: str) -> str:
    reduce_prompt = reduce(level)
    llm = obtain_chat_model()
    prompt = reduce_prompt.invoke(input)
    response = await llm.ainvoke(prompt)
    return response.content

async def collapse_summaries(state: OverallState):
    doc_lists = split_list_of_docs(
        state["collapsed_summaries"], length_function, token_max
    )
    results = []
    for doc_list in doc_lists:
        results.append(await acollapse_docs(doc_list, _reduce))

    return {"collapsed_summaries": results}

def should_collapse(
    state: OverallState,
) -> Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = length_function(state["collapsed_summaries"])
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "generate_final_summary"
    
async def generate_final_summary(state: OverallState, level):
    response = await _reduce(state["collapsed_summaries"], level)
    return {"final_summary": response}

def construct_graph(level: str):
    graph = StateGraph(OverallState)
    graph.add_node("generate_summary", lambda s: generate_summary(s, level))
    graph.add_node("collect_summaries", collect_summaries)
    graph.add_node("collapse_summaries", collapse_summaries)
    graph.add_node("generate_final_summary", lambda s: generate_final_summary(s, level))

    graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
    graph.add_edge("generate_summary", "collect_summaries")
    graph.add_conditional_edges("collect_summaries", should_collapse)
    graph.add_conditional_edges("collapse_summaries", should_collapse)
    graph.add_edge("generate_final_summary", END)

    app = graph.compile()
    return app

async def final_summary(file_path, level: str = "beginner"):
    app = construct_graph(level)
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    split_docs = splitting(pages)
    result = None
    async for step in app.astream(
        {"contents": [doc.page_content for doc in split_docs]},
        {"recursion_limit": 10},
    ): result = step
    return result

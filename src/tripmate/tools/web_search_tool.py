from langchain_core.tools import tool

@tool
def web_search(query:str) -> str:
    return f"Result for: {query}"


tools = [
    web_search,
]

llm_with_tools = llm.bind_tools(tools)

graph.add_nodes("tools", ToolNode(tools))
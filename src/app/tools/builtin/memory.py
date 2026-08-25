from langchain_core.tools import BaseTool
from langmem import create_manage_memory_tool, create_search_memory_tool

MEMORY_NAMESPACE = ("memories", "{langgraph_user_id}")


search_memory: BaseTool = create_search_memory_tool(
    namespace=MEMORY_NAMESPACE,
    instructions=(
        "当用户的问题需要参考过去保存的偏好、事实、项目背景、历史约定时，"
        "调用这个工具搜索长期记忆。"
    ),
)

manage_memory: BaseTool = create_manage_memory_tool(
    namespace=MEMORY_NAMESPACE,
    instructions=(
        "当用户明确要求你记住某个偏好、事实、项目背景，或者纠正已有记忆时，"
        "调用这个工具管理长期记忆。不要把普通闲聊都写入长期记忆。"
    ),
)
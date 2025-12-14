import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 加载环境变量
load_dotenv()

# ==========================================
# 0. 初始化与数据加载
# ==========================================
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_NAME = "gemini-2.5-flash" 

# 加载本地数据
with open("memory.json", "r", encoding="utf-8") as f:
    MEMORY_DB = json.load(f)

with open("file_metadata.json", "r", encoding="utf-8") as f:
    FILES_DB = json.load(f)


# ==========================================
# 1. 定义工具 (Tools)
# ==========================================

def get_memory(category: str) -> str:
    """
    Tool 1: 记忆获取工具
    Reads the user's knowledge level from memory based on a category.

    Args:
        category: The category to check (e.g., 'astronomy', 'calculus').

    Returns:
        A string describing the user's knowledge level.
    """
    print(f"\n🔍 [Tool: Memory] Reading memory for: {category}")
    levels = MEMORY_DB.get("knowledge_levels", {})
    # 模糊匹配处理
    for key, val in levels.items():
        if category.lower() in key.lower():
            return f"User Knowledge in {key}: Level={val['level']}. Details: {val['detailed_description']}"
    return "No specific memory found for this category."


def search_files(keywords: str) -> list[str]:
    """
    Tool 2: 本地文件选择工具
    Searches file metadata for relevant files based on keywords.

    Args:
        keywords: Search terms (e.g., 'sun orbit').

    Returns:
        A list of relevant file TITLES (strings).
    """
    print(f"\n📂 [Tool: Files] Searching files for: {keywords}")
    hits = []
    kw_list = keywords.lower().split()

    for filename, data in FILES_DB.items():
        content = data["content"].lower()
        title = filename.lower()
        # 简单的匹配逻辑：只要命中一个关键词就入选
        if any(k in content or k in title for k in kw_list):
            hits.append(filename)

    print(f"   -> Found {len(hits)} files.")
    return hits


def generate_reply_tool(instruction: str, file_titles: list[str], user_context: str) -> str:
    """
    Tool 3: 回复生成工具 (流式输出)
    The FINAL tool to call. It generates the actual response to the user.

    Args:
        instruction: The specific instruction for the reply (e.g., "Summarize the content").
        file_titles: A list of filenames to include in the context.
        user_context: The user's knowledge level or other context found from memory.
    """
    print(f"\n✍️ [Tool: Reply] Generating reply with {len(file_titles)} files...")

    # 1. 组装上下文：根据 Director 传来的文件名，去数据库里把真正的 Content 拿出来
    full_context = ""
    for title in file_titles:
        if title in FILES_DB:
            full_context += f"\n--- File: {title} ---\n{FILES_DB[title]['content']}\n"

    # 2. 构建 Prompt
    final_prompt = f"""
    You are a helpful tutor.

    [User Context]
    {user_context}

    [Reference Materials]
    {full_context}

    [Instruction]
    {instruction}

    Please provide a comprehensive answer based strictly on the materials above.
    Adjust the complexity to match the User Context.
    """

    # 3. 调用生成模型 (这里模拟流式输出)
    # 注意：在 Tool 内部我们再次调用了 client 来生成最终内容
    try:
        response_stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=final_prompt
        )

        print("\n" + "=" * 20 + " Stream Start " + "=" * 20)
        final_text = ""
        for chunk in response_stream:
            print(chunk.text, end="", flush=True)
            final_text += chunk.text
        print("\n" + "=" * 20 + " Stream End " + "=" * 20 + "\n")

        return "Reply generated and streamed to user successfully."

    except Exception as e:
        return f"Error generating reply: {e}"


# ==========================================
# 2. Director Agent (主控)
# ==========================================

def run_hyperknow_agent():
    # 1. 配置 Director
    # 这里的 Director 不负责写内容，只负责“调度”
    director_tools = [get_memory, search_files, generate_reply_tool]

    director_config = types.GenerateContentConfig(
        tools=director_tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=False,
            maximum_remote_calls=5
        ),
        system_instruction="""
        You are the 'Director Agent' of Hyperknow.
        Your goal is to fulfill the user's request by orchestrating tools.

        WORKFLOW:
        1. Analyze user request.
        2. Call `get_memory` to understand the user's level.
        3. Call `search_files` to find relevant file TITLES based on the topic.
        4. CRITICAL: You MUST call `generate_reply_tool` as the final step.
           - Pass the instruction.
           - Pass the list of file titles you found.
           - Pass the user context you found.

        DO NOT generate the final summary yourself. You are just the manager.
        """
    )

    # 2. 启动对话
    director_chat = client.chats.create(
        model=MODEL_NAME,
        config=director_config
    )

    user_query = "给我总结这学期天⽂课上的所有内容"
    print(f"User: {user_query}")

    # 3. 发送指令
    # SDK 会自动进行：Director -> Memory -> Director -> Files -> Director -> Reply Tool
    response = director_chat.send_message(user_query)

    # 4. 结束
    # 因为 generate_reply_tool 已经打印了流式内容，
    # Director 最后的返回值通常是 "Reply generated..." 这种确认信息
    print(f"\n[Director Final Status]: {response.text}")


if __name__ == "__main__":
    run_hyperknow_agent()

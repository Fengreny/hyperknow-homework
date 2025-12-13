import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# ====================== 0. 配置与初始化 ======================
load_dotenv()

# 配置代理地址和 Key
# ⚠️ 注意：这里用的是 OpenAI 的 SDK，但是调用的是 Gemini 模型
client = OpenAI(
    api_key= os.getenv("GPTS_API_KEY"),  # 填入你在 gptsapi 的 key
    base_url="https://api.gptsapi.net/v1"  # 代理地址
)

MODEL_NAME = "gemini-3-pro-preview"  # 或者 gemini-1.5-pro


# ====================== 1. 定义工具函数 ======================

def memory_load_function(subject: str):
    """
    读取memory.json，查询用户指定科目的知识水平。
    """
    print(f"\n🔍 [Tool] 正在查询记忆库: {subject}")
    try:
        # 确保你有 memory.json
        with open('memory.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        knowledge_levels = data.get("knowledge_levels", {})
        if subject.lower() in knowledge_levels:
            return json.dumps(knowledge_levels[subject.lower()])  # 返回字符串
        else:
            return json.dumps({"error": f"未找到 {subject} 的记忆信息"})
    except Exception as e:
        return json.dumps({"error": f"读取记忆文件出错: {str(e)}"})


def select_files(query: str):
    """
    根据关键词搜索本地文件，返回相关文件名列表。
    """
    print(f"\n📂 [Tool] 正在搜索文件关键词: {query}")
    found_files = []
    try:
        # 确保你有 file_metadata.json
        with open('file_metadata.json', 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        for filename, info in file_data.items():
            content = info.get("content", "").lower()
            if query.lower() in content:
                found_files.append(filename)
        print(f"   ✅ 找到 {len(found_files)} 个相关文件")
        return json.dumps(found_files)
    except Exception as e:
        return json.dumps({"error": f"搜索文件出错: {str(e)}"})


def reply_generator(file_titles_json: str, instruction: str):
    """
    生成最终回复。
    注意：OpenAI Function Calling 传回来的是字符串，需要自己 json.loads 一下 file_titles
    """
    try:
        file_titles = json.loads(file_titles_json)
    except:
        file_titles = []  # 容错

    print(f"\n📝 [Tool] 正在调用回复生成器...")
    print(f"   参考文件: {file_titles}")

    context_content = ""
    try:
        with open('file_metadata.json', 'r', encoding='utf-8') as f:
            all_files_data = json.load(f)
        for title in file_titles:
            if title in all_files_data:
                file_text = all_files_data[title]['content']
                context_content += f"\n--- 文件名: {title} ---\n{file_text}\n"
    except Exception as e:
        return f"读取文件内容出错: {str(e)}"

    final_prompt = f"""
    【角色】天文课助教
    【用户指令】{instruction}
    【参考资料】{context_content if context_content else "无"}
    【要求】请根据参考资料和指令，生成详细的总结。
    """

    print("\n💡 [Stream] 开始流式输出回复: \n")

    # 使用 client.chat.completions.create 进行流式生成
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": final_prompt}],
        stream=True
    )

    full_text = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_text += content
    print("\n")

    return "回复已生成完毕。"


# ====================== 2. 定义 Tools Schema (OpenAI 格式) ======================
# OpenAI 需要显式定义 Schema，不像 Gemini SDK 那么智能自动生成
tools = [
    {
        "type": "function",
        "function": {
            "name": "memory_load_function",
            "description": "读取memory.json，查询用户指定科目的知识水平。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "科目名称，例如 'astronomy', 'calculus' 等。"
                    }
                },
                "required": ["subject"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "select_files",
            "description": "根据关键词搜索本地文件，返回相关文件名列表。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，例如 'astronomy'。"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reply_generator",
            "description": "这是生成最终回复的唯一工具。根据提供的文件列表和指令，流式生成回答。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_titles_json": {
                        "type": "string",
                        "description": "需要参考的文件名列表，必须是 JSON 字符串格式，例如 '[\"file1.pdf\", \"file2.pdf\"]'。"
                    },
                    "instruction": {
                        "type": "string",
                        "description": "生成回复的具体指令。"
                    }
                },
                "required": ["file_titles_json", "instruction"]
            }
        }
    }
]

available_functions = {
    "memory_load_function": memory_load_function,
    "select_files": select_files,
    "reply_generator": reply_generator,
}


# ====================== 3. 核心 Agent 逻辑 ======================
class HyperknowAgent:
    def __init__(self):
        self.messages = []  # 手动维护历史记录

    def run(self, user_query: str):
        print(f"👤 用户指令：{user_query}")
        self.messages.append({"role": "user", "content": user_query})

        while True:
            # 1. 发送请求给模型
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=self.messages,
                tools=tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            # 2. 检查是否有函数调用
            tool_calls = response_message.tool_calls

            if tool_calls:
                # 把模型的回复（包含函数调用请求）加入历史
                self.messages.append(response_message)

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)

                    print(f"\n🤖 模型请求调用: {function_name}")
                    print(f"   参数: {function_args}")

                    # 3. 执行函数
                    function_response = function_to_call(**function_args)

                    # 4. 将结果回传给模型
                    print(f"   📤 将结果回传给 Director Agent...")
                    self.messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                    )
            else:
                # 没有函数调用，说明任务结束
                print(f"\n🎉 Director Agent 任务结束: {response_message.content}")
                break


# ====================== 运行测试 ======================
if __name__ == "__main__":
    # 确保目录下有 memory.json 和 file_metadata.json
    agent = HyperknowAgent()
    agent.run("给我总结这学期天文课上的所有内容")

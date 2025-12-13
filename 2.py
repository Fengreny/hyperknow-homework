import json
import os
import time
import google.generativeai as genai
from google.generativeai.types import content_types
from collections import abc
from dotenv import load_dotenv

# ====================== 0. 配置与初始化 ======================
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"], transport='rest')


reply_model = genai.GenerativeModel("gemini-3-pro-preview")


# ====================== 1. 定义工具函数 ======================


def memory_load_function(subject: str):
    """
    读取memory.json，查询用户指定科目的知识水平。

    Args:
        subject: 科目名称，例如 'astronomy', 'calculus' 等。
    """
    print(f"\n🔍 [Tool] 正在查询记忆库: {subject}")
    try:
        with open('memory.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        knowledge_levels = data.get("knowledge_levels", {})
        # 模糊匹配处理
        if subject.lower() in knowledge_levels:
            return knowledge_levels[subject.lower()]
        else:
            return {"error": f"未找到 {subject} 的记忆信息"}
    except Exception as e:
        return {"error": f"读取记忆文件出错: {str(e)}"}


def select_files(query: str):
    """
    根据关键词搜索本地文件，返回相关文件名列表。

    Args:
        query: 搜索关键词，例如 'astronomy'。
    """
    print(f"\n📂 [Tool] 正在搜索文件关键词: {query}")
    found_files = []
    try:
        with open('file_metadata.json', 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        for filename, info in file_data.items():
            content = info.get("content", "").lower()
            if query.lower() in content:
                found_files.append(filename)
        print(f"   ✅ 找到 {len(found_files)} 个相关文件")
        return found_files
    except Exception as e:
        return {"error": f"搜索文件出错: {str(e)}"}


def reply_generator(file_titles: list[str], instruction: str):
    """
    根据提供的文件列表和指令，流式生成回答。

    Args:
        file_titles: 需要参考的文件名列表。
        instruction: 生成回复的具体指令。
    """
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

    # 构建最终生成的 Prompt
    final_prompt = f"""
    【角色】天文课助教
    【用户指令】{instruction}
    【参考资料】{context_content if context_content else "无"}
    【要求】请根据参考资料和指令，生成详细的总结。
    """

    print("\n💡 [Stream] 开始流式输出回复: \n")
    # 使用独立的 reply_model 进行流式生成，避免工具递归调用
    response = reply_model.generate_content(final_prompt, stream=True)

    full_text = ""
    for chunk in response:
        if chunk.text:
            print(chunk.text, end="", flush=True)
            full_text += chunk.text
    print("\n")

    return "回复已生成完毕。"


# 工具字典，用于手动执行
functions = {
    'memory_load_function': memory_load_function,
    'select_files': select_files,
    'reply_generator': reply_generator
}

# ====================== 2. 初始化 Director Agent ======================
# 将工具列表传给模型
tools_list = [memory_load_function, select_files, reply_generator]
director_model = genai.GenerativeModel("gemini-3-pro-preview", tools=tools_list)


# ====================== 3. 核心 Agent 逻辑 ======================
class  directorAgent :
    def __init__(self):
        # 开启自动函数调用设为 False，手动控制流程
        self.chat = director_model.start_chat(enable_automatic_function_calling=False)

    def run(self, user_query: str):
        print(f"👤 用户指令：{user_query}")

        # 发送初始消息
        response = self.chat.send_message(user_query)

        # 循环处理：只要模型想调函数，就一直循环
        while True:
            part = response.parts[0]

            # 1. 检查是否有函数调用请求 (Function Call)
            if part.function_call:
                fc = part.function_call
                func_name = fc.name
                func_args = dict(fc.args)

                print(f"\n🤖 模型请求调用: {func_name}")
                print(f"   参数: {func_args}")

                # 2. 执行函数
                if func_name in functions:
                    api_result = functions[func_name](**func_args)
                else:
                    api_result = {"error": "Function not found"}

                # 3. 构建函数响应 (Function Response)
                # Gemini 需要特殊的格式把结果传回去
                function_response_part = genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=func_name,
                        response={'result': api_result}
                    )
                )

                # 4. 将结果发回给模型，让它决定下一步
                print(f"   📤 将结果回传给 Director Agent...")
                response = self.chat.send_message([function_response_part])

            # 5. 如果是普通文本，说明任务结束（或者模型在自言自语）
            elif part.text:
                print(f"\n🎉 Director Agent 任务结束: {part.text}")
                break
            else:
                break


# ====================== 运行测试 ======================
if __name__ == "__main__":
    agent = directorAgent()
    agent.run("给我总结这学期天文课上的所有内容")

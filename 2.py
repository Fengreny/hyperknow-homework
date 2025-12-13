import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

# ====================== 加载apikey======================
load_dotenv()


# 加上 transport='rest'
genai.configure(api_key=os.environ["GEMINI_API_KEY"], transport='rest')


# 初始化基础模型
model = genai.GenerativeModel("gemini-3-pro-preview")


# ======================工具函数 ======================
def memory_load_function(subject: str):
    """读取memory.json，查询用户指定科目的知识水平"""
    print(f"🔍 正在查询科目: {subject}")
    try:
        with open('memory.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        knowledge_levels = data.get("knowledge_levels", {})
        if subject.lower() in knowledge_levels:
            result = knowledge_levels[subject.lower()]
            return json.dumps(result, ensure_ascii=False)
        else:
            return json.dumps({"error": f"未找到 {subject} 的记忆信息"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"读取记忆文件出错: {str(e)}"}, ensure_ascii=False)


def select_files(query: str):
    """根据关键词搜索本地文件，返回相关文件名列表"""
    print(f"🔍 正在搜索关键词: {query}")
    found_files = []
    try:
        with open('file_metadata.json', 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        for filename, info in file_data.items():
            content = info.get("content", "").lower()
            if query.lower() in content:
                found_files.append(filename)
        print(f"✅ 找到 {len(found_files)} 个相关文件")
        return json.dumps(found_files, ensure_ascii=False)
    except FileNotFoundError:
        return json.dumps({"error": "找不到 file_metadata.json 文件"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"搜索文件出错: {str(e)}"}, ensure_ascii=False)


def reply_generator(file_titles: list, instruction: str):
    """流式生成最终回复"""
    print(f"\n📝 正在生成回复，参考文件: {file_titles}")
    context_content = ""
    try:
        with open('file_metadata.json', 'r', encoding='utf-8') as f:
            all_files_data = json.load(f)
        for title in file_titles:
            if title in all_files_data:
                file_text = all_files_data[title]['content']
                context_content += f"\n--- 文件名: {title} ---\n{file_text}\n"
    except Exception as e:
        return json.dumps({"error": f"读取文件内容出错: {str(e)}"}, ensure_ascii=False)

    # 构建Prompt
    final_prompt = f"""
    【角色】天文课助教，适配初学者水平
    【指令】{instruction}
    【参考资料】{context_content if context_content else "无"}
    【要求】分点总结，通俗易懂，避免专业术语。
    """

    # 流式调用Gemini
    try:
        response = model.generate_content(final_prompt, stream=True)
        print("\n💡 Gemini 回复: ", end="", flush=True)
        full_text = ""
        for chunk in response:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_text += chunk.text
        print("\n")
        return json.dumps({"status": "success", "reply": full_text}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"调用Gemini出错: {str(e)}"}, ensure_ascii=False)


# 工具映射（用于解析后执行）
TOOL_MAP = {
    "memory_load_function": memory_load_function,
    "select_files": select_files,
    "reply_generator": reply_generator
}


# ====================== 核心 Agent 逻辑 ======================
class directorAgent:
    def __init__(self):
        # 启动对话会话（保存上下文）
        self.chat_session = model.start_chat(history=[])

        # 核心 Prompt（告诉模型如何输出函数调用指令，替代 Schema 的作用）
        self.function_call_prompt = """
        你可以调用以下函数来完成用户任务：
        1. 函数名：memory_load_function
           作用：查询用户指定科目的知识水平
           参数：{"subject": "科目名（如astronomy）"}
           返回：JSON字符串

        2. 函数名：select_files
           作用：根据关键词搜索本地天文课PDF文件
           参数：{"query": "搜索关键词（如astronomy）"}
           返回：JSON字符串

        3. 函数名：reply_generator
           作用：生成最终回复（唯一的回复工具）
           参数：{"file_titles": ["文件名1", "文件名2"], "instruction": "生成指令"}
           返回：JSON字符串

        你的输出规则：
        - 如果需要调用函数，输出严格的JSON格式：{"call_function": {"name": "函数名", "args": {"参数名": "参数值"}}}
        - 如果不需要调用函数，直接输出回复内容即可
        - 调用函数时，参数必须严格匹配上述格式，不要添加额外字段
        """

    def _parse_function_call(self, model_response: str) -> dict:
        """解析模型输出的函数调用指令（提取JSON）"""
        try:
            # 提取JSON部分（兼容模型可能加的额外文字）
            start = model_response.find("{")
            end = model_response.rfind("}") + 1
            json_str = model_response[start:end]
            call_data = json.loads(json_str)
            return call_data.get("call_function", {})
        except Exception as e:
            print(f"⚠️ 解析函数调用指令失败: {e}")
            return {}

    def _execute_function(self, func_name: str, func_args: dict) -> str:
        """执行函数并返回结果"""
        if func_name not in TOOL_MAP:
            return json.dumps({"error": f"函数 {func_name} 不存在"}, ensure_ascii=False)
        try:
            # 解包参数执行函数
            result = TOOL_MAP[func_name](**func_args)
            return result
        except Exception as e:
            return json.dumps({"error": f"执行 {func_name} 出错: {str(e)}"}, ensure_ascii=False)

    def run(self, user_query: str):
        """运行Agent（无Schema，纯Prompt引导）"""
        print(f"👤 用户指令：{user_query}\n")

        # 第一步：发送用户指令 + 函数调用引导Prompt
        initial_prompt = f"{self.function_call_prompt}\n用户当前指令：{user_query}"
        response = self.chat_session.send_message(initial_prompt)
        model_output = response.text.strip()

        # 第二步：循环处理模型输出（支持多轮函数调用）
        while True:
            # 解析是否需要调用函数
            func_call = self._parse_function_call(model_output)
            if not func_call:
                # 模型直接返回回复，结束流程
                print(f"\n🎉 最终回复：\n{model_output}")
                break

            # 提取函数名和参数
            func_name = func_call.get("name")
            func_args = func_call.get("args", {})
            print(f"\n📌 模型决定调用函数：{func_name}")
            print(f"📌 调用参数：{func_args}")

            # 执行函数
            func_result = self._execute_function(func_name, func_args)
            print(f"✅ 函数执行结果：{func_result[:100]}...")

            # 第三步：将函数结果回传给模型，继续决策
            follow_up_prompt = f"""
            你之前调用了函数 {func_name}，参数是 {func_args}，执行结果如下：
            {func_result}

            请根据这个结果决定：
            1. 继续调用其他函数（按指定JSON格式输出）；
            2. 直接生成最终回复（输出文本即可）。
            """
            response = self.chat_session.send_message(follow_up_prompt)
            model_output = response.text.strip()


# ====================== 运行测试 ======================
if __name__ == "__main__":
    # 初始化无Schema的Agent
    agent = directorAgent()
    # 运行测试
    agent.run("给我总结这学期天文课上的所有内容")

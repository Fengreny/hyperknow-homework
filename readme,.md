流程

1. Director 接收指令与规划

输入：用户 Query "给我总结这学期天文课上的所有内容"。
决策：Director Agent 根据 System Instruction，判断需要先了解用户水平，再查找资料，最后生成回复。
2. 调用 get_memory (获取上下文)

动作：Director 自动提取关键词 "astronomy"，调用 get_memory(category="astronomy")。
结果：函数返回用户的知识画像（如 "Level=beginner"）。Director 将此信息暂存入对话历史，用于后续指导写作风格。
3. 调用 search_files (检索索引)

动作：Director 再次发起调用 search_files(keywords="astronomy")。
策略：为了节省 Token，该工具只返回文件标题列表（如 ['Sun.pdf', 'Orbits.pdf']），而不是全文。Director 此时只掌握了“有哪些资料可用”，而不知道具体内容。
4. 调用 generate_reply_tool (核心：模型嵌套调用)

动作：Director 将收集到的所有元数据打包，调用最终工具：
generate_reply_tool(
    instruction="总结内容...",
    file_titles=['Sun.pdf', ...],
    user_context="Level=beginner..."
)

最后模型根据prompt输出结果

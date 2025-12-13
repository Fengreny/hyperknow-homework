import os
from google import genai
from dotenv import load_dotenv
# 1. 获取 API Key (确保你的环境变量里确实叫 GEMINI_API_KEY，或者直接在这里填字符串测试)
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    raise ValueError("API Key not found! Please set GEMINI_API_KEY in your environment.")

# 2. 初始化 Client 时传入 api_key
client = genai.Client(api_key=api_key)

try:
    # 3. 使用正确的模型名称 (例如 gemini-1.5-flash 或 gemini-2.0-flash-exp)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="How does AI work?"
    )
    print(response.text)

except Exception as e:
    print(f"An error occurred: {e}")

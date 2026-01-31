import requests
import json
import sys

# 1. 你的公网流式地址 (注意路径里的 stream-predict)
url = "https://jhcyun.com/llm/llm/stream-predict"

# 2. 构造和 README 一样的请求体
payload = {
    "text": "What is my name?",
    "history": [
        {"content": "You are a helpful assistant!", "metadata": None, "role": "system"},
        {"content": "My name is AkagawaTsurunaki.", "metadata": None, "role": "user"},
        {"content": "Hello, AkagawaTsurunaki.", "metadata": None, "role": "assistant"}
    ]
}

headers = {
    "Content-Type": "application/json; charset=utf-8"
}

print(f"🚀 正在连接流式接口: {url} ...")
print("-" * 50)

try:
    # 【关键】stream=True 保持连接不关闭
    with requests.post(url, json=payload, headers=headers, stream=True) as response:
        if response.status_code == 200:
            print("连接成功！接收数据中...\n")

            # 使用 iter_lines() 按行读取，或者 iter_content() 按字节读取
            # 这里使用 iter_content 模拟最原始的接收
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    # 实时解码并打印，flush=True 确保不缓存，立刻显示
                    print(chunk.decode('utf-8', errors='ignore'), end='', flush=True)
        else:
            print(f"❌ 服务器报错: {response.status_code}")
            print(response.text)

    print("\n\n" + "-" * 50)
    print("✅ 测试结束")

except Exception as e:
    print(f"💥 发生错误: {e}")
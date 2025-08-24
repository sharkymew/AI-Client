# AI-Client
# Copyright (C) 2025 SharkyMew
# Licensed under the GNU AGPL v3 or later.

import time
import random
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch

# ================== 配置区 ==================
MODEL_NAME = "./MewyShark" # 更换为本地模型路径或Hugging Face的模型路径
MAX_HISTORY_TOKENS = 8192  # 最大上下文长度（token 数）
MAX_NEW_TOKENS = 512       # 单次回复最大长度
THINKING_MIN = 0         # 最小“思考”时间（秒）「调试用，弃用」
THINKING_MAX = 0        # 最大“思考”时间（秒） 「调试用，弃用」

# ================== 初始化模型 ==================
print(f"正在加载：{MODEL_NAME} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 检查设备，注：仅支持mps或cpu，cuda请在下方修改，mps仅支持Apple M系列SoC，Intel Mac仅可使用cpu
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用设备: {device}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
).to(device) # cuda请在此处将device更换为"cuda"（带引号）

# ================== 系统提示词（基础部分）==================
system_prompt_base = (
    '''你是一个Assistant，会帮助用户解决问题，当前时间是：'''
)

# 初始化对话历史
history = []

def update_system_message():
    """动态更新系统消息中的时间"""
    current_time = time.asctime()
    return {
        "role": "system",
        "content": system_prompt_base + current_time
    }

# 初始加载
history = [update_system_message()]

print("=== 欢迎与 Assistant 聊天！输入内容和它聊天吧（输入 'exit' 退出） ===")

# ================== 上下文裁剪函数 ==================
def trim_history(history, tokenizer, max_tokens=MAX_HISTORY_TOKENS):
    """从头开始删除旧对话，直到总长度小于 max_tokens"""
    while True:
        try:
            total_tokens = len(tokenizer.apply_chat_template(history, tokenize=True))
            if total_tokens <= max_tokens or len(history) <= 2:
                break
            # 删除最早的 user/assistant 对话（跳过 system）
            if len(history) > 2:
                history.pop(1)
                if len(history) > 2:
                    history.pop(1)
        except:
            break
    return history

# ================== 主循环 ==================
while True:
    user_input = input("\n你：").strip()
    if user_input.lower() in ["exit", "quit", "退出"]:
        print("Assistant：哼，不聊啦，下次见！👋")
        break

    if not user_input:
        print("Assistant：……杂鱼，你连话都说不出来吗？")
        continue

    # 更新系统时间
    history[0] = update_system_message()
    history.append({"role": "user", "content": user_input})

    # 裁剪历史
    history = trim_history(history, tokenizer, MAX_HISTORY_TOKENS)

    # 构造 prompt
    try:
        prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        print(f"Assistant：啧，提示词出问题了：{e}")
        continue

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 创建 streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # 生成参数，可根据需要修改
    gen_kwargs = {
        **inputs,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.8,
        "repetition_penalty": 1.3,
        "streamer": streamer,
        "pad_token_id": tokenizer.eos_token_id
    }

    # 模拟“思考中”延迟（调试用，弃用）
    thinking_delay = random.uniform(THINKING_MIN, THINKING_MAX)
    print("\nAssistant：", end="", flush=True)
    # time.sleep(thinking_delay)

    # 启动生成线程
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    # 等待 streamer 准备好第一个 token（弃用）
    # while not streamer.next_tokens_are_ready and thread.is_alive():
    #     time.sleep(0.01)

    # 实时输出
    reply_text = ""
    try:
        for new_text in streamer:
            print(new_text, end="", flush=True)
            reply_text += new_text
        print()  # 换行
    except Exception as e:
        print(f"\nAssistant：啊……头好晕，刚才断片了：{e}")

    # 添加到历史
    if reply_text.strip():
        history.append({"role": "assistant", "content": reply_text.strip()})
    else:
        print("Assistant：……今天不想理你。")
        history.append({"role": "assistant", "content": "……今天不想理你。"})
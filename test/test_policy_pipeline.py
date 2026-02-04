import os
os.environ["HF_ENDPOINT"] = "http://hf-mirror.com"
import sys

from dotenv import load_dotenv


CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from retrieval_pipe import PolicyRetrievalPipeline, POLICY_QUESTIONS

def main():
    # 加载 .env 中的 API Key
    load_dotenv()
    ark_api_key = os.getenv("ARK_API_KEY")

    if not ark_api_key:
        raise RuntimeError("环境变量 ARK_API_KEY 未设置，请在 .env 中配置 ARK_API_KEY=<你的火山方舟密钥>")

    # 1. 构建检索 pipeline（当前使用 doubao + colpali）
    pipeline = PolicyRetrievalPipeline(
        data_dir="./policy_data",          # PDF 统一放在这里
        output_dir="./policy_outputs",     # 检索输出目录
        llm_model="doubao",                # 或 "gpt4" / "qwen"，与 visual_rag 配置一致
        vision_retriever="nemo",        # 或 "nemo" 等
        api_keys={
            "doubao": ark_api_key,
            # 如需 GPT-4，可在此补充 "openai": os.getenv("OPENAI_API_KEY")
        },
        top_k=5,
        force_reindex=False,
        qwen_server_url="http://localhost:8001",
        qwen_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    # 2. 选择 policy_data 下的文件作为输入
    inputs = [
        "policy_data/海南省农机购置与应用补贴实施方案的通知.doc",
        # 你也可以只用 PDF：
        # "policy_data/海南省农机购置与应用补贴实施方案的通知.pdf",
        # 或者中央一号文件：
        # "policy_data/2025年中央一号文件.pdf",
    ]

    # 3. 调用 pipeline，得到 7 个维度的结果
    results = pipeline.retrieve_policy_info(inputs)

    # 4. 按 7 个预设维度打印结果，并将所有维度的回答写入同一个文件方便观察 answer 格式
    print("==== 政策关键信息抽取结果 ====")

    output_path = os.path.join(pipeline.output_dir, "dim_answers.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("==== 政策关键信息抽取结果 ====\n")

        for key in POLICY_QUESTIONS.keys():
            info = results.get(key, {})

            question = info.get("question", "")
            answer = info.get("answer", "")
            analysis = info.get("analysis", "")

            print(f"\n[{key}]")
            print("问题：", question)
            print("答案：", answer)
            print("分析：", analysis)

            f.write(f"\n[{key}]\n")
            f.write(f"问题：{question}\n")
            f.write(f"答案：{answer}\n")
            f.write(f"分析：{analysis}\n")


if __name__ == "__main__":
    main()
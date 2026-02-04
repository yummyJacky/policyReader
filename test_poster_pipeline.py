import os
os.environ["HF_ENDPOINT"] = "http://hf-mirror.com"
import argparse
from typing import List, Dict, Any

from retrieval_pipe import PolicyRetrievalPipeline
from poster_pipeline import (
    build_poster_records_from_answers,
    build_poster_records_from_answers_json,
    concat_poster_images,
    generate_cover_and_tail_from_single_image,
)
from dotenv import load_dotenv

load_dotenv()
ark_api_key = os.getenv("ARK_API_KEY")
if not ark_api_key:
    raise SystemExit("ARK_API_KEY 环境变量未配置，无法调用 doubao")
pipeline = PolicyRetrievalPipeline(
    data_dir="./policy_data",          # PDF 统一放在这里
    output_dir="./policy_outputs",     # 检索输出目录
    llm_model="doubao",
    vision_retriever="nemo",        # 或 "nemo" 等
    api_keys={
        "doubao": ark_api_key,
    },
    top_k=3,
    force_reindex=False,
)

def _run_from_inputs(
    inputs: List[str],
    *,
    max_verify_attempts: int,
) -> Dict[str, Dict[str, Any]]:
    """通过检索 pipeline 从政策文件生成七维度回答并构建海报记录。"""


    results = pipeline.retrieve_policy_info(inputs)

    return build_poster_records_from_answers(
        dim_answers=results,
        max_verify_attempts=max_verify_attempts,
    )


def _run_from_answers_json(
    json_path: str,
    *,
    max_verify_attempts: int,
) -> Dict[str, Dict[str, Any]]:
    """直接从包含七维度回答的 JSON 文件构建海报记录。"""

    return build_poster_records_from_answers_json(
        json_path=json_path,
        max_verify_attempts=max_verify_attempts,
    )


def run_full_poster_pipeline(
    *,
    inputs: List[str] | None,
    answers_json: str | None,
    policy_title: str,
    max_verify_attempts: int,
    long_image_path: str,
) -> None:
    """从政策文件或 answers JSON 到长海报图片的端到端测试流程。

    - 如果提供 ``answers_json``，则跳过检索阶段，直接从 JSON 构建海报；
    - 否则使用 ``inputs`` 中的政策文件跑完整的七维度问答 + 海报生成流程。
    """

    if answers_json:
        print("[test_poster_pipeline] 使用 answers JSON 直接生成海报:", answers_json)
        posters = _run_from_answers_json(
            json_path=answers_json,
            max_verify_attempts=max_verify_attempts,
        )
    else:
        if not inputs:
            raise SystemExit("未提供政策文件路径，也未提供 answers JSON。")

        print("[test_poster_pipeline] 使用政策文件跑检索 + 七维度问答:", inputs)
        posters = _run_from_inputs(
            inputs=inputs,
            max_verify_attempts=max_verify_attempts,
        )

    # 3. 打印每个维度的海报信息
    print("=== Poster records per dimension ===")
    for k, v in posters.items():
        print(f"[{k}] short_text=", v.get("short_text"))
        print(f"[{k}] image=", v.get("image_result"))
        print("-")

    # 4. 基于各维度回答生成一句话总结，用于封面/尾页
    dim_answers_for_summary: Dict[str, Dict[str, str]] = {}
    for k, v in posters.items():
        dim_answers_for_summary[k] = {
            "question": str(v.get("question", "")),
            "answer": str(v.get("original_answer", "")),
            "analysis": "",
        }

    summary_result = pipeline.generate_one_sentence_summary(dim_answers_for_summary)
    summary = summary_result.get("answer", "")
    print("[test_poster_pipeline] 一句话总结:", summary)

    # 5. 调用单次文生图生成上下两栏海报，并裁剪为封面和尾页
    cover_path, tail_path = generate_cover_and_tail_from_single_image(
        title=policy_title,
        summary=summary,
        output_dir=os.path.dirname(long_image_path) or "./policy_outputs/posters",
    )

    # 6. 纵向拼接长图：封面 + 各维度 + 尾页
    merged = concat_poster_images(
        posters,
        long_image_path,
        cover_image=cover_path or None,
        tail_image=tail_path or None,
    )
    print("Long poster image:", merged)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        help="政策文件路径（一个或多个，相对路径或绝对路径）",
    )
    parser.add_argument(
        "--answers-json",
        help="包含七个维度回答的 JSON 文件路径（跳过检索，直接生成海报）",
    )
    parser.add_argument(
        "--max-verify-attempts",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--long-image-path",
        default=os.path.join("./policy_outputs/posters", "policy_poster_long.png"),
    )
    parser.add_argument(
        "--cover-tail-only",
        action="store_true",
        help="仅测试封面+包含 logo 尾页的生成和裁剪功能，不跑完整海报流水线",
    )

    args = parser.parse_args()

    # 如果只想测试封面 + 尾页（包含 logo）生成与裁剪，不跑完整流水线
    if args.cover_tail_only:
        # 交互式获取标题和一句话总结
        try:
            policy_title = input("封面主标题中的政策名称（如“本政策”或具体文件名）? ").strip()
        except EOFError:
            policy_title = ""
        if not policy_title:
            policy_title = "本政策"

        try:
            summary = input("请输入用于封面的一句话总结（将直接印在海报上）: ").strip()
        except EOFError:
            summary = ""
        if not summary:
            summary = "这是一个用于测试的一句话政策总结示例。"

        cover_path, tail_path = generate_cover_and_tail_from_single_image(
            title=policy_title,
            summary=summary,
            output_dir=os.path.dirname(args.long_image_path) or "./policy_outputs/posters",
        )

        print("[test_poster_pipeline] 封面路径:", cover_path)
        print("[test_poster_pipeline] 尾页路径:", tail_path)
    else:
        # 原有完整流水线测试逻辑保持不变
        # 交互式获取政策标题，用于“一图读懂 xxxx”中的 xxxx
        try:
            policy_title = input("你需要分析哪一个政策文件? ").strip()
        except EOFError:
            policy_title = ""
        if not policy_title:
            policy_title = "本政策"

        run_full_poster_pipeline(
            inputs=list(args.inputs) if args.inputs else None,
            answers_json=args.answers_json,
            policy_title=policy_title,
            max_verify_attempts=args.max_verify_attempts,
            long_image_path=args.long_image_path,
        )

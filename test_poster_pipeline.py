import os
os.environ["HF_ENDPOINT"] = "http://hf-mirror.com"
import argparse
from typing import List, Dict, Any

from retrieval_pipe import PolicyRetrievalPipeline
from poster_pipeline import (
    build_poster_records_from_answers,
    build_poster_records_from_answers_json,
    concat_poster_images,
)
from dotenv import load_dotenv

load_dotenv()


def _run_from_inputs(
    inputs: List[str],
    *,
    max_verify_attempts: int,
) -> Dict[str, Dict[str, Any]]:
    """通过检索 pipeline 从政策文件生成七维度回答并构建海报记录。"""

    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise SystemExit("ARK_API_KEY 环境变量未配置，无法调用 doubao")

    pipeline = PolicyRetrievalPipeline(
        data_dir="./policy_data",
        output_dir="./policy_outputs",
        api_keys={"doubao": api_key},
    )

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

    # 4. 纵向拼接长图
    merged = concat_poster_images(posters, long_image_path)
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
        default=3,
    )
    parser.add_argument(
        "--long-image-path",
        default=os.path.join("./policy_outputs/posters", "policy_poster_long.png"),
    )

    args = parser.parse_args()

    run_full_poster_pipeline(
        inputs=list(args.inputs) if args.inputs else None,
        answers_json=args.answers_json,
        max_verify_attempts=args.max_verify_attempts,
        long_image_path=args.long_image_path,
    )

import argparse
import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from retrieval_pipe import POLICY_QUESTIONS, PolicyRetrievalPipeline
from poster_pipeline import generate_cover_and_tail_from_single_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cover and tail posters using Gemini based on a title and a model-generated summary.",
    )
    parser.add_argument(
        "--title",
        required=True,
        help="Policy title, will be used in '一图读懂{title}' on the cover.",
    )
    parser.add_argument(
        "--result_json",
        required=True,
        help=(
            "Path to a JSON file containing model-generated results, "
            "which must include a 'summary' field with an 'answer' value."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="./policy_outputs/posters",
        help="Directory to save the generated cover and tail images.",
    )
    parser.add_argument(
        "--logo_path",
        default="./assets/logo.png",
        help="Path to the logo image used on the tail page.",
    )
    parser.add_argument(
        "--image_input",
        default=None,
        help="Optional image path used as additional visual input/background.",
    )
    parser.add_argument(
        "--max_verify_attempts",
        type=int,
        default=2,
        help="Maximum number of attempts for text verification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 确保可以使用 doubao 文本模型
    load_dotenv()
    ark_api_key = os.getenv("ARK_API_KEY")
    if not ark_api_key:
        print("[generate_cover_tail] ARK_API_KEY 环境变量未配置，无法调用 doubao 生成 summary。")
        return

    # 从检索/问答结果文件中读取各维度回答，调用模型生成一句话总结
    # 支持两种格式：
    # - .json  : 单个 JSON 对象
    # - .jsonl : 多行 JSON，每行一个对象，这里采用最后一条有效记录
    try:
        result_path = args.result_json  # type: ignore[assignment]
        last_obj: Dict[str, Any] | None = None
        with open(result_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    last_obj = obj
        if last_obj is None:
            raise ValueError("no valid JSON object found in JSONL file")
        raw_data = last_obj
    except Exception as exc:  # noqa: BLE001
        print(f"[generate_cover_tail] Failed to load result_json: {exc}")
        return

    # 兼容两种结构：
    # 1) 直接是各维度结果字典；
    # 2) 外层是包含 result 字段的完整 Job 响应。
    data_obj: Dict[str, Any]
    result_field = raw_data.get("result")
    if isinstance(result_field, dict):
        data_obj = result_field
    else:
        data_obj = raw_data

    dim_answers: Dict[str, Dict[str, str]] = {}
    for key in POLICY_QUESTIONS.keys():
        info = data_obj.get(key) or {}
        if not isinstance(info, dict):
            info = {}
        dim_answers[key] = {
            "question": str(info.get("question", "")),
            "answer": str(info.get("answer", "")),
            "analysis": str(info.get("analysis", "")),
        }

    # 使用与 test_poster_pipeline 相同的策略，通过 PolicyRetrievalPipeline 调用模型生成一句话总结
    pipeline = PolicyRetrievalPipeline(
        data_dir="./policy_data",
        output_dir="./policy_outputs",
        llm_model="doubao",
        vision_retriever="nemo",
        api_keys={"doubao": ark_api_key},
        top_k=3,
        force_reindex=False,
    )

    summary_result = pipeline.generate_one_sentence_summary(dim_answers)
    summary_value = summary_result.get("answer", "")

    if not summary_value:
        print("[generate_cover_tail] 模型未能生成有效的一句话 summary，封面/尾页生成已中止。")
        return

    cover_path, tail_path = generate_cover_and_tail_from_single_image(
        title=args.title,
        summary=summary_value,
        output_dir=args.output_dir,
        logo_path=args.logo_path,
        image_input=args.image_input,
        max_verify_attempts=args.max_verify_attempts,
    )

    if not cover_path and not tail_path:
        print("[generate_cover_tail] Failed to generate cover and tail images.")
        return

    if cover_path:
        print(f"[generate_cover_tail] Cover image saved at: {cover_path}")
    if tail_path:
        print(f"[generate_cover_tail] Tail image saved at: {tail_path}")


if __name__ == "__main__":
    main()

import os
import time
import json
import base64
from io import BytesIO
from typing import Any, Callable, Dict, List

from PIL import Image
from google import genai
from google.genai import types
from openai import OpenAI
from dotenv import load_dotenv
from retrieval_pipe import POLICY_QUESTIONS
import logging

load_dotenv()
logger = logging.getLogger("VisDoMRAG")
llm = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=os.getenv("ARK_API_KEY"),
    )

def generate_text_only(prompt) -> str:
    try:
        response = llm.responses.create(
            model="doubao-seed-1-6-flash-250828",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ],
        )
        try:
            return response.output[1].content[0].text  # type: ignore[union-attr,index]
        except Exception:
            return ""
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error in generate_text: {str(e)}")
        return ""

def _judge_answer_has_effective_info(
    answer: str,
) -> bool:
    if not answer or not answer.strip():
        return False

    judge_prompt = (
        "下面是针对某个政策维度的模型回答：\n\n"
        f"{answer}\n\n"
        "请判断该回答是否表明政策文件中**包含**该维度的具体有效信息，"
        "还是明确说明“政策未提及/无法从中提取/未给出具体内容”等情况。\n"
        "如果包含有效信息，请只输出：有信息\n"
        "如果属于未提及或无法提取，请只输出：无信息\n"
        "不要输出任何其它内容。"
    )

    judge_result = generate_text_only(judge_prompt)
    return judge_result.startswith("有信息")


def _summarize_answer_for_poster(
    answer: str,
) -> str:
    refine_prompt = (
        "以下是在政策文件中提取得到的内容，精简以下内容，适合放到海报中。\n\n"
        f"{answer}"
    )

    return generate_text_only(refine_prompt)


def _generate_image_with_verification(
    dim_key: str,
    short_text: str,
    image_prompt: str,
    max_verify_attempts: int = 2,
) -> Any:
    """使用 Gemini 文生图 + Doubao 多模态校验生成单个维度的海报。

    """

    print(f"[poster_pipeline] 开始为维度 {dim_key} 生成海报，最多重试 {max_verify_attempts} 次 (Gemini 文生图 + Doubao 校验)...")

    image_generator = make_gemini_image_generator()
    image_verifier = make_doubao_image_verifier()

    image_result: Any = None
    attempts = 0

    while attempts < max_verify_attempts:
        print(f"  [poster_pipeline] 第 {attempts + 1} 次生成维度 {dim_key} 的图片...")
        candidate = image_generator(dim_key, image_prompt)
        image_result = candidate

        if not candidate or image_verifier is None:
            print("  [poster_pipeline] 图片生成失败或未能创建校验器，结束重试。")
            break

        try:
            print("  [poster_pipeline] 调用多模态模型校验生成的图片是否覆盖精简文案...")
            ok = image_verifier(dim_key, short_text, candidate)
        except Exception:
            print("  [poster_pipeline] 多模态校验过程中发生异常，结束重试。")
            break

        if ok:
            print("  [poster_pipeline] 校验通过，本次生成结果将被采用。")
            break

        attempts += 1

    if image_result:
        print(f"[poster_pipeline] 维度 {dim_key} 海报生成流程结束，最终图片路径: {image_result}")
    else:
        print(f"[poster_pipeline] 维度 {dim_key} 海报生成流程结束，未能生成有效图片。")

    return image_result


def build_poster_records_from_answers(
    dim_answers: Dict[str, Dict[str, str]],
    max_verify_attempts: int = 2,
) -> Dict[str, Dict[str, Any]]:
    """根据七个维度的回答，生成海报所需的精简文案与文生图提示词。

    参数：
    - dim_answers: 结构类似于 policy_api_fastapi._run_job 里的 partial：
        {
            "who": {"question": "...", "answer": "...", "analysis": "..."},
            ...
        }
    - max_verify_attempts: 单个维度在“生成海报 + 校验”上的最大尝试次数。

    返回：
    - 一个字典，key 为维度 key（如 who/what/...），value 包含：
        {
            "question": 原问题,
            "original_answer": 原始回答,
            "short_text": 适合海报的精简文案,
            "image_prompt": 文生图提示词,
            "image_result": 生成的海报图片路径
        }
    """

    results: Dict[str, Dict[str, Any]] = {}

    dim_keys = list(POLICY_QUESTIONS.keys())
    total_dims = len(dim_keys)
    print(f"[poster_pipeline] 开始根据各维度回答生成海报记录，共 {total_dims} 个维度...")

    # 按 POLICY_QUESTIONS 中的 key 顺序遍历，避免依赖 dim_answers 的内部顺序
    for idx, dim_key in enumerate(dim_keys, start=1):
        print(f"[poster_pipeline] ({idx}/{total_dims}) 处理维度: {dim_key} ...")
        info = dim_answers.get(dim_key) or {}
        answer = info.get("answer", "")
        question = info.get("question", POLICY_QUESTIONS.get(dim_key, ""))

        print("  [poster_pipeline] 判断该维度回答是否包含有效信息...")
        if not _judge_answer_has_effective_info(answer):
            print("  [poster_pipeline] 判断结果为无有效信息，跳过该维度。")
            continue

        print("  [poster_pipeline] 生成适合海报的精简文案...")
        short_text = _summarize_answer_for_poster(answer)
        if not short_text:
            print("  [poster_pipeline] 精简文案为空，跳过该维度。")
            continue

        # 为每个维度构造带有明确文本格式约束的文生图提示词，
        # 以便多次生成的海报在标题/正文结构、字体和排版上尽量保持一致。

        # 顶部标题暂时不要，因为现在会自动添加
        # "- 顶部需要一个简短标题，概括该维度的核心含义，控制在 8~12 个字之内。\n"
        image_prompt = (
            "你是一名专业的政策宣传海报设计师，请根据以下信息设计一张单页海报：\n\n"
            f"[政策维度] {dim_key}\n"
            f"[问题] {question}\n"
            "[精简文案]\n"
            f"{short_text}\n\n"
            "请严格遵守以下海报中文本格式要求：\n"
            "- 海报上必须包含清晰可读的中文文字。\n"
            "- 标题和正文在所有海报中保持统一的字体家族和字号层级。\n"
            "- 文本排版简洁，避免在同一张海报中使用过多字体或夸张的字号变化。\n"
            "- 文本颜色保持统一（如深色字体配浅色背景），确保易读性。\n\n"
            "只需生成一张符合上述要求、适合打印和展示的高质量竖版政策宣传海报图片。"
        )

        print("  [poster_pipeline] 调用文生图 + 多模态校验流程生成海报图片...")
        image_result = _generate_image_with_verification(
            dim_key=dim_key,
            short_text=short_text,
            image_prompt=image_prompt,
            max_verify_attempts=max_verify_attempts,
        )

        results[dim_key] = {
            "question": question,
            "original_answer": answer,
            "short_text": short_text,
            "image_prompt": image_prompt,
            "image_result": image_result,
        }

        print("[poster_pipeline] 维度", dim_key, "处理完成。")

    print(f"[poster_pipeline] 所有维度处理完成，共生成 {len(results)} 条海报记录。")
    return results


def build_poster_records_from_answers_json(
    json_path: str,
    max_verify_attempts: int = 2,
) -> Dict[str, Dict[str, Any]]:
    """从包含七个维度回答的 JSON/JSONL 文件中构建海报记录。

    JSONL 文件（多个维度，每行一个 JSON 对象）：
    每一行都是上面的结构，第 1 行 -> who，第 2 行 -> what，依此类推，
    按 POLICY_QUESTIONS 的 key 顺序依次映射。
    """

    print(f"[poster_pipeline] 从 JSON 文件加载七维度回答: {json_path}")

    # JSONL，多行多维度
    if json_path.endswith(".jsonl"):
        dim_keys = list(POLICY_QUESTIONS.keys())
        dim_answers: Dict[str, Dict[str, Any]] = {}

        with open(json_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                if not isinstance(obj, dict):
                    continue

                if idx >= len(dim_keys):
                    break

                dim_key = dim_keys[idx]
                dim_answers[dim_key] = obj

        if not dim_answers:
            raise ValueError("answers JSONL 文件中未解析到任何有效的维度回答")

        return build_poster_records_from_answers(
            dim_answers=dim_answers,
            max_verify_attempts=max_verify_attempts,
        )


def make_gemini_image_generator(
    model: str = "gemini-3-pro-image-preview",
    output_dir: str = "./policy_outputs/posters",
) -> Callable[[str, str], str]:
    """创建一个基于 Google Gemini 文生图服务的 image_generator。

    返回的函数签名为 (dimension_key, prompt) -> image_path
    """
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options=types.HttpOptions(
            base_url="https://api.zhizengzeng.com/google",
        ),
    )
    os.makedirs(output_dir, exist_ok=True)

    def _generate(dim_key: str, prompt: str) -> str:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt],
            )
        except Exception:
            # 出错时返回空字符串，调用方可据此判断是否生成成功
            return ""

        image_path: str = ""
        image_index = 0

        for part in getattr(response, "parts", []) or []:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is None:
                continue

            try:
                image = part.as_image()
                filename = f"{int(time.time())}_{dim_key}_{image_index}.png"
                image_index += 1
                full_path = os.path.join(output_dir, filename)
                image.save(full_path)
                image_path = full_path
            except Exception:
                continue

        return image_path

    return _generate


def make_doubao_image_verifier(
    model: str = "doubao-seed-1-6-flash-250828",
) -> Callable[[str, str, str], bool]:
    """创建一个独立初始化 doubao 客户端的多模态校验函数工厂。

    返回函数签名为 (dimension_key, short_text, image_path) -> bool。
    """

    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise ValueError("ARK_API_KEY 环境变量未配置，无法初始化 doubao 客户端")

    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key,
    )

    def _encode_image_to_base64(path: str) -> str:
        img = Image.open(path).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _verify(dim_key: str, short_text: str, image_path: str) -> bool:
        try:
            b64 = _encode_image_to_base64(image_path)
        except Exception:
            return False

        prompt = (
            "你是一个严格的审核助手。下面是一张海报图片以及对应的精简文案：\n\n"
            f"[维度] {dim_key}\n"
            f"[文案] {short_text}\n\n"
            "请完成以下两项检查：\n"
            "1. 语义覆盖：判断这张图片是否已经完整、准确地体现了上述文案中的所有关键信息，"
            "不会遗漏主要含义，也不会产生明显偏差。\n"
            "2. 文本噪音：检查图片中文字中是否出现 Markdown 标记或明显的技术性/格式控制符号，"
            "例如 ###、##、以 # 开头的标题标记、成对的 **、代码块标记 ```、尖括号 <>、方括号 [ ] 等，"
            "这类字符在正式海报中应当被去除。\n\n"
            "只有当图片既完整覆盖文案内容，又不存在上述无用的 Markdown/技术性标记时，"
            "才认为是合格结果。\n\n"
            "如果认为图片合格，请只输出：通过\n"
            "只要存在语义上的明显遗漏或偏差，或者出现上述任何 Markdown 标记/无用技术性字符，"
            "请只输出：不通过\n"
            "不要输出任何其他内容。"
        )

        try:
            response = client.responses.create(  # type: ignore[attr-defined]
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{b64}",
                            },
                            {
                                "type": "input_text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )
        except Exception:
            return False

        try:
            text = response.output[1].content[0].text  # type: ignore[union-attr,index]
        except Exception:
            return False

        return (text or "").strip().startswith("通过")

    return _verify


def concat_poster_images(
    posters: Dict[str, Dict[str, Any]],
    output_path: str,
) -> str:
    """将各维度生成的海报按维度顺序纵向拼接成一张长图并保存。

    - posters: build_poster_records_from_answers 的返回结果
    - output_path: 拼接后长图的保存路径
    """

    # 按 POLICY_QUESTIONS 的 key 顺序取出对应的图片路径
    image_paths: List[str] = []
    for dim_key in POLICY_QUESTIONS.keys():
        info = posters.get(dim_key) or {}
        img_path = info.get("image_result")
        if isinstance(img_path, str) and img_path:
            image_paths.append(img_path)

    if not image_paths:
        print("[poster_pipeline] 没有可用的单维度海报图片，跳过长图拼接。")
        return ""

    print(f"[poster_pipeline] 开始拼接长图，本次共 {len(image_paths)} 张图片参与拼接...")

    images: List[Image.Image] = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception:
            print(f"[poster_pipeline] 打开图片失败，跳过: {p}")
            continue

    if not images:
        print("[poster_pipeline] 所有图片均无法打开，长图拼接失败。")
        return ""

    widths = [img.width for img in images]
    heights = [img.height for img in images]

    max_width = max(widths)
    total_height = sum(heights)

    # 以白色背景创建长图画布
    long_img = Image.new("RGB", (max_width, total_height), color="white")

    y_offset = 0
    for img in images:
        # 居左粘贴，必要时可以按需缩放或居中
        long_img.paste(img, (0, y_offset))
        y_offset += img.height

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    long_img.save(output_path)
    print(f"[poster_pipeline] 长图拼接完成，已保存至: {output_path}")
    return output_path



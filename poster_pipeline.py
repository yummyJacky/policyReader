import os
import time
import json
import base64
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional
import functools
import logging

from PIL import Image
from google import genai
from google.genai import types
from openai import OpenAI
from dotenv import load_dotenv
from retrieval_pipe import POLICY_QUESTIONS

load_dotenv()
from realtime_logger import get_logger

logger = get_logger("poster_pipeline")
llm = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.getenv("ARK_API_KEY"),
)

# BACKGROUND_IMAGE_PATH = "./assets/background.png"
BACKGROUND_IMAGE_PATH = None

# ────────────────────────── 风格一致性 prompt 片段 ──────────────────────────
_STYLE_CONSISTENCY_INSTRUCTION = (
    "\n\n[风格一致性要求]\n"
    "我已提供了一张或多张参考图片。你生成的海报必须与这些参考图片在以下方面保持高度一致：\n"
    "- 整体配色方案（主色调、辅助色、背景色）；\n"
    "- 插画/图形风格（如扁平、拟物、3D 等）；\n"
    "- 版式布局结构（如标题位置、内容区域划分、留白比例）；\n"
    "- 字体风格与字号层级（标题与正文的对比关系）；\n"
    "- 装饰元素的类型与密度（如圆角卡片、色块、图标等）。\n"
    "请确保生成的图片看起来与参考图片属于同一系列/同一套模板。\n"
)


@functools.lru_cache(maxsize=8)
def _cached_gemini_image_generator(
    model: str = "gemini-3-pro-image-preview",
    output_dir: str = "./policy_outputs/posters",
) -> Callable[[str, str, Any | None], str]:
    return make_gemini_image_generator(model=model, output_dir=output_dir)


@functools.lru_cache(maxsize=8)
def _cached_doubao_image_verifier(
    model: str = "doubao-seed-1-6-flash-250828",
) -> Callable[[str, str, str], bool]:
    return make_doubao_image_verifier(model=model)


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
        "还是明确说明\"政策未提及/无法从中提取/未给出具体内\"等情况。\n"
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
        "你是一名政策解读编辑，将下面的政策回答内容压缩精炼为适合印在政策宣传海报上的中文文案。\n"
        "要求：\n"
        "1. 严格保留政策的关键信息和结论，删除与政策要点无关的解释性话语、套话、提示或模型思考过程。\n"
        "2. 不要包含任何标题、序号、列表符号、Markdown 标记（如 #、**、- 等）或引导语。\n"
        "3. 不要复述任务说明，不要解释你在做什么，只输出最终精简后的内容本身。\n\n"
        "下面是需要精简的原始内容：\n"
        f"{answer}"
    )

    return generate_text_only(refine_prompt)


# ────────────────────────── 辅助：合并风格参照图列表 ──────────────────────────

def _merge_image_inputs(*sources: Any) -> List[Any] | None:
    """将多个图片来源合并为一个扁平列表，去重（按路径字符串去重）。"""
    merged: List[Any] = []
    seen_paths: set[str] = set()

    for src in sources:
        if src is None:
            continue
        items = src if isinstance(src, (list, tuple)) else [src]
        for item in items:
            if isinstance(item, str):
                if item and item not in seen_paths:
                    seen_paths.add(item)
                    merged.append(item)
            elif isinstance(item, Image.Image):
                merged.append(item)

    return merged if merged else None


def _generate_image_with_verification(
    dim_key: str,
    short_text: str,
    image_prompt: str,
    image_input: str | None = None,
    style_reference: Any | None = None,
    max_verify_attempts: int = 2,
) -> Any:
    """使用 Gemini 文生图 + Doubao 多模态校验生成单个维度的海报。

    Parameters
    ----------
    style_reference : 已生成的风格参照图（路径或 PIL Image），将与 image_input/背景图一同
                      传给 Gemini，并在 prompt 中追加风格一致性要求。
    """

    print(
        f"[poster_pipeline] 开始为维度 {dim_key} 生成海报，最多重试 {max_verify_attempts} 次 "
        "(Gemini 文生图 + Doubao 校验)..."
    )

    image_generator = _cached_gemini_image_generator()
    image_verifier = _cached_doubao_image_verifier()

    # ── 合并所有图片输入：背景 + 外部 image_input + 风格参照 ──
    bg_input = image_input or BACKGROUND_IMAGE_PATH
    combined_input = _merge_image_inputs(bg_input, style_reference)

    # ── 如果存在风格参照，给 prompt 追加风格一致性要求 ──
    final_prompt = image_prompt
    if style_reference is not None:
        final_prompt = image_prompt + _STYLE_CONSISTENCY_INSTRUCTION

    image_result: Any = None
    attempts = 0

    while attempts < max_verify_attempts:
        print(f"  [poster_pipeline] 第 {attempts + 1} 次生成维度 {dim_key} 的图片...")
        candidate = image_generator(dim_key, final_prompt, combined_input)
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


def build_poster_for_dimension(
    dim_key: str,
    dim_info: Dict[str, str],
    image_input: str | None = None,
    style_reference: Any | None = None,
    max_verify_attempts: int = 2,
) -> Optional[Dict[str, Any]]:
    """为单个维度生成海报。

    Parameters
    ----------
    style_reference : 已生成的风格参照图（路径、PIL Image 或列表），用于保证本次
                      生成的海报与之前已生成的海报风格保持一致。
    """
    logger.info("[poster_pipeline] Building poster for dimension: %s", dim_key)
    answer = dim_info.get("answer", "")
    question = dim_info.get("question", POLICY_QUESTIONS.get(dim_key, ""))

    print(f"  [poster_pipeline] 判断该维度回答是否包含有效信息: {dim_key} ...")
    if not _judge_answer_has_effective_info(answer):
        print("  [poster_pipeline] 判断结果为无有效信息，跳过该维度。")
        return None

    print("  [poster_pipeline] 生成适合海报的精简文案...")
    short_text = _summarize_answer_for_poster(answer)
    if not short_text:
        print("  [poster_pipeline] 精简文案为空，跳过该维度。")
        return None

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
        "- 文本颜色保持统一（如深色字体配浅色背景），确保易读性。\n"
        "- 海报中只能出现与政策内容直接相关的文字，不得自行添加任何机构名称、部门名称、logo、二维码、网站、联系电话、公章或“设计：XXX”等署名信息。\n"
        "- 不要在海报中出现本提示中的\"[问题]\"\"[精简文案]\"等提示性标记，也不要把问题原文或方括号标签直接照抄到画面，只使用精简文案中的关键信息作为正文。\n\n"
        "只需生成一张符合上述要求、适合打印和展示的高质量竖版政策宣传海报图片。"
    )

    print("  [poster_pipeline] 调用文生图 + 多模态校验流程生成海报图片...")
    image_result = _generate_image_with_verification(
        dim_key=dim_key,
        short_text=short_text,
        image_prompt=image_prompt,
        image_input=image_input,
        style_reference=style_reference,
        max_verify_attempts=max_verify_attempts,
    )

    return {
        "question": question,
        "original_answer": answer,
        "short_text": short_text,
        "image_prompt": image_prompt,
        "image_result": image_result,
    }


def build_poster_records_from_answers(
    dim_answers: Dict[str, Dict[str, str]],
    max_verify_attempts: int = 2,
    style_reference: Any | None = None,
) -> Dict[str, Dict[str, Any]]:
    """根据七个维度的回答，生成海报所需的精简文案与文生图提示词。

    Parameters
    ----------
    style_reference : 外部提供的风格参照图（路径、PIL Image 或列表）。
                      如果提供，则所有维度的海报都会以该图片为风格基准。
                      此外，第一张成功生成的海报也会自动加入风格参照列表，
                      以确保后续维度的海报与之保持一致。
    """

    results: Dict[str, Dict[str, Any]] = {}

    dim_keys = list(POLICY_QUESTIONS.keys())
    total_dims = len(dim_keys)
    logger.info(
        "[poster_pipeline] Starting to generate poster records for %d dimensions...",
        total_dims,
    )

    # ── 用于在维度之间传递风格参照的列表 ──
    # 初始值为外部传入的 style_reference（如果有的话）
    accumulated_style_refs: List[Any] = []
    if style_reference is not None:
        if isinstance(style_reference, (list, tuple)):
            accumulated_style_refs.extend(style_reference)
        else:
            accumulated_style_refs.append(style_reference)

    for idx, dim_key in enumerate(dim_keys, start=1):
        logger.info(
            "[poster_pipeline] Processing dimension %s (%d/%d)...",
            dim_key,
            idx,
            total_dims,
        )
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
            "- 文本颜色保持统一（如深色字体配浅色背景），确保易读性。\n"
            "- 海报中只能出现与政策内容直接相关的文字，不得自行添加任何机构名称、部门名称、logo、二维码、网站、联系电话、公章或“设计：XXX”等署名信息。\n"
            "- 不要在海报中出现本提示中的\"[问题]\"\"[精简文案]\"等提示性标记，也不要把问题原文或方括号标签直接照抄到画面，只使用精简文案中的关键信息作为正文。\n\n"
            "只需生成一张符合上述要求、适合打印和展示的高质量竖版政策宣传海报图片。"
        )

        # ── 将当前已积累的风格参照传给生成函数 ──
        current_style_ref = accumulated_style_refs if accumulated_style_refs else None

        print("  [poster_pipeline] 调用文生图 + 多模态校验流程生成海报图片...")
        image_result = _generate_image_with_verification(
            dim_key=dim_key,
            short_text=short_text,
            image_prompt=image_prompt,
            style_reference=current_style_ref,
            max_verify_attempts=max_verify_attempts,
        )

        # ── 将成功生成的海报图片加入风格参照列表，供后续维度使用 ──
        if image_result and isinstance(image_result, str) and os.path.exists(image_result):
            # 只保留最近一张作为主要风格参照（避免传入过多图片影响生成效果）
            # 但仍保留外部传入的 style_reference
            _external = []
            if style_reference is not None:
                if isinstance(style_reference, (list, tuple)):
                    _external.extend(style_reference)
                else:
                    _external.append(style_reference)
            accumulated_style_refs = _external + [image_result]
            print(
                f"  [poster_pipeline] 已将 {dim_key} 的海报加入风格参照列表，"
                f"当前参照数量: {len(accumulated_style_refs)}"
            )

        results[dim_key] = {
            "question": question,
            "original_answer": answer,
            "short_text": short_text,
            "image_prompt": image_prompt,
            "image_result": image_result,
        }

        logger.info("[poster_pipeline] Dimension %s processed.", dim_key)

    logger.info(
        "[poster_pipeline] All dimensions processed, %d poster records generated.",
        len(results),
    )
    return results


def build_poster_records_from_answers_json(
    json_path: str,
    max_verify_attempts: int = 2,
    style_reference: Any | None = None,
) -> Dict[str, Dict[str, Any]]:
    """从包含七个维度回答的 JSON/JSONL 文件中构建海报记录。"""

    logger.info(
        "[poster_pipeline] Loading seven-dimensional answers from JSON file: %s",
        json_path,
    )

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
            style_reference=style_reference,
        )


def make_gemini_image_generator(
    model: str = "gemini-3-pro-image-preview",
    output_dir: str = "./policy_outputs/posters",
) -> Callable[[str, str, Any | None], str]:
    """创建一个基于 Google Gemini 文生图服务的 image_generator。"""
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options=types.HttpOptions(
            base_url="https://api.zhizengzeng.com/google",
        ),
    )
    os.makedirs(output_dir, exist_ok=True)

    def _make_generate_config_disable_afc() -> Any | None:
        try:
            gen_cfg_type = getattr(types, "GenerateContentConfig", None)
            afc_cfg_type = getattr(types, "AutomaticFunctionCallingConfig", None)
            if gen_cfg_type is None or afc_cfg_type is None:
                return None
            afc_cfg = None
            try:
                afc_cfg = afc_cfg_type(disable=True)
            except Exception:
                try:
                    afc_cfg = afc_cfg_type(enabled=False)
                except Exception:
                    return None
            try:
                return gen_cfg_type(automatic_function_calling=afc_cfg)
            except Exception:
                return None
        except Exception:
            return None

    _gen_config = _make_generate_config_disable_afc()

    def _guess_mime_type(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            return "image/jpeg"
        if ext == ".webp":
            return "image/webp"
        if ext == ".gif":
            return "image/gif"
        return "image/png"

    def _make_image_part(data: bytes, mime_type: str) -> Any | None:
        part_from_bytes = getattr(types.Part, "from_bytes", None)
        if callable(part_from_bytes):
            try:
                return part_from_bytes(data=data, mime_type=mime_type)
            except Exception:
                pass
        blob_type = getattr(types, "Blob", None)
        if blob_type is not None:
            try:
                return types.Part(inline_data=blob_type(data=data, mime_type=mime_type))
            except Exception:
                return None
        return None

    def _coerce_image_items(image_input: Any | None) -> List[Any]:
        if image_input is None:
            return []
        if isinstance(image_input, (list, tuple)):
            return list(image_input)
        return [image_input]

    def _generate(dim_key: str, prompt: str, image_input: Any | None = None) -> str:
        try:
            image_parts: List[Any] = []
            for item in _coerce_image_items(image_input):
                if isinstance(item, str):
                    try:
                        with open(item, "rb") as f:
                            data = f.read()
                        part = _make_image_part(data, _guess_mime_type(item))
                        if part is not None:
                            image_parts.append(part)
                    except Exception:
                        continue
                elif isinstance(item, Image.Image):
                    try:
                        buf = BytesIO()
                        item.save(buf, format="PNG")
                        part = _make_image_part(buf.getvalue(), "image/png")
                        if part is not None:
                            image_parts.append(part)
                    except Exception:
                        continue

            if image_parts:
                response = client.models.generate_content(
                    model=model,
                    contents=[
                        types.Content(
                            role="user",
                            parts=image_parts + [types.Part(text=prompt)],
                        )
                    ],
                    config=_gen_config,
                )
            else:
                response = client.models.generate_content(
                    model=model,
                    contents=[prompt],
                    config=_gen_config,
                )
        except Exception:
            return ""

        image_path: str = ""
        image_index = 0
        parts_list: List[Any] = []

        candidates = getattr(response, "candidates", None)
        if candidates is not None:
            for cand in candidates or []:
                content = getattr(cand, "content", None)
                if content is None:
                    continue
                cand_parts = getattr(content, "parts", None) or []
                parts_list.extend(cand_parts)

        for part in parts_list:
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


def generate_cover_and_tail_from_single_image(
    title: str,
    summary: str,
    output_dir: str = "./policy_outputs/posters",
    logo_path: str = "./assets/logo.png",
    image_input: Any | None = None,
    style_reference: Any | None = None,
    max_verify_attempts: int = 5,
) -> tuple[str, str]:
    """通过一次 Gemini 文生图生成上下两栏图片，并在尾页区域融合指定 logo。

    Parameters
    ----------
    style_reference : 已生成的维度海报图片（路径、PIL Image 或列表），用于确保封面
                      和尾页与各维度海报在整体风格上保持一致。
    """

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[poster_pipeline] 未配置 GEMINI_API_KEY，无法生成封面+尾页。")
        return "", ""

    image_generator = make_gemini_image_generator(
        model="gemini-3-pro-image-preview",
        output_dir=output_dir,
    )

    try:
        text_verifier = make_cover_tail_text_verifier()
    except Exception as e:  # noqa: BLE001
        print("[poster_pipeline] 封面/尾页文字校验器初始化失败，将跳过文字检查:", e)
        text_verifier = None

    prompt = (
        "你是一名专业的政策宣传海报设计师。请在一张竖版图片中设计上下两个区域：\n\n"
        "[上半部分：封面]\n"
        f"- 主标题：一图读懂{title}\n"
        f"- 一句话总结:{summary}\n"
        "- 要求：标题醒目，一句话总结清晰可读，只包含**一句话总结**的内容，不要包含任何如“摘要：”“一句话总结：”等前缀或说明性文字，也不要添加与政策内容无关的机构名称、logo、公章、二维码、联系电话或“设计：XXX”等署名信息。封面的整体配色、插画风格和版式布局应与之前已经生成的各维度政策海报保持同一套风格（如相近的主色调、简洁的卡片式内容区和扁平插画风格），不要使用完全不同的模板或过于花哨的背景。\n\n"
        "[下半部分：尾页]\n"
        "- 作为本次政策解读的结束页，可以包含简短致谢或提示文案，但这些文字必须与本政策的阅读或使用直接相关，整体配色和版式需要与封面及各维度海报保持一致，好像同一套模板的最后一页。\n"
        "- 尾页需要在底部或角落等合适位置添加一行署名文字：国机灵通·农业产业大模型生成。要求该行文字清晰可读、字号适中、与整体风格一致，不要喧宾夺主。\n"
        "- 请将我提供的 logo 图像融入尾页的版式中，例如放置在底部与海报主色调一致的色块/条带或圆角区域中，注意保持 logo 比例和清晰度，不要随意修改其形状；尾页中只允许出现这一处 logo，不得额外添加其他机构 logo、公章、二维码或“设计：XXX”等署名信息。避免让 logo 单独悬浮在与整体风格不协调的背景上。\n"
        "- 整张图片中禁止出现作者姓名、设计团队名称、联系方式、网站地址等与政策内容无关的装饰性文字。\n"
        "- 尾页同样不要出现本提示中的方括号标签或说明性文字，只需使用简洁的中文文案和提供的 logo。\n\n"
        "整张图片为竖版，高度大于宽度，上下两个区域边界清晰，方便后续按照高度中线将图片裁剪为两张独立海报。"
    )

    # ── 合并图片输入：外部 image_input / 背景 / logo / 风格参照 ──
    if image_input is None:
        inputs: list[Any] = []
        if BACKGROUND_IMAGE_PATH and os.path.exists(BACKGROUND_IMAGE_PATH):
            inputs.append(BACKGROUND_IMAGE_PATH)
        if logo_path and os.path.exists(logo_path):
            inputs.append(logo_path)
        image_input = inputs or None

    # 将风格参照图合并进去
    combined_input = _merge_image_inputs(image_input, style_reference)

    # 如果有风格参照，追加风格一致性要求到 prompt
    if style_reference is not None:
        prompt = prompt + _STYLE_CONSISTENCY_INSTRUCTION

    attempts = 0
    while attempts < max_verify_attempts:
        attempts += 1
        print(f"[poster_pipeline] 第 {attempts} 次调用 Gemini 生成封面+尾页...")
        big_path = image_generator("cover_tail", prompt, combined_input)
        if not big_path:
            print("[poster_pipeline] Gemini 生成封面+尾页失败: 未返回图片")
            continue
        print("[poster_pipeline] Gemini 生成封面+尾页完成，图片已保存:", big_path)

        try:
            img = Image.open(big_path).convert("RGB")
        except Exception:
            print("[poster_pipeline] 打开 Gemini 生成的封面+尾页图片失败，重试生成。")
            continue

        width, height = img.size
        mid = height // 2

        overlap = max(0, min(24, height // 20))
        cover_bottom = min(height, mid + overlap)
        tail_top = max(0, mid - overlap)

        cover_img = img.crop((0, 0, width, cover_bottom))
        tail_img = img.crop((0, tail_top, width, height))

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(big_path))[0]
        cover_path = os.path.join(output_dir, f"{base_name}_cover.png")
        tail_path = os.path.join(output_dir, f"{base_name}_tail.png")

        cover_img.save(cover_path)
        tail_img.save(tail_path)

        print(f"[poster_pipeline] 封面海报裁剪完成: {cover_path}")
        print(f"[poster_pipeline] 尾页海报裁剪完成: {tail_path}")

        if text_verifier is None:
            return cover_path, tail_path

        try:
            print("[poster_pipeline] 使用多模态大模型检查封面文字是否为正常简体中文...")
            ok_cover = text_verifier(cover_path)
            print("[poster_pipeline] 使用多模态大模型检查尾页文字是否为正常简体中文...")
            ok_tail = text_verifier(tail_path)
        except Exception:
            print("[poster_pipeline] 封面/尾页文字校验过程中发生异常，将跳过检查并返回本次结果。")
            return cover_path, tail_path

        if ok_cover and ok_tail:
            print("[poster_pipeline] 封面与尾页文字校验均通过。")
            return cover_path, tail_path

        print("[poster_pipeline] 封面或尾页文字校验未通过，将重新生成封面+尾页...")

    print("[poster_pipeline] 达到封面+尾页最大重试次数，未能生成通过文字校验的图片。")
    return "", ""


def make_doubao_image_verifier(
    model: str = "doubao-seed-1-6-flash-250828",
) -> Callable[[str, str, str], bool]:
    """创建一个独立初始化 doubao 客户端的多模态校验函数工厂。"""

    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise ValueError("ARK_API_KEY 环境变量未配置，无法初始化 doubao 客户端")

    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key,
    )

    def _encode_image_to_base64(path: str) -> str:
        with Image.open(path) as opened:
            img = opened.convert("RGB")
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


def make_cover_tail_text_verifier(
    model: str = "doubao-seed-1-6-flash-250828",
) -> Callable[[str], bool]:
    """创建一个用于封面/尾页文字检查的多模态校验器。"""

    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise ValueError("ARK_API_KEY 环境变量未配置，无法初始化 doubao 客户端")

    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key,
    )

    def _encode_image_to_base64(path: str) -> str:
        with Image.open(path) as opened:
            img = opened.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _verify(image_path: str) -> bool:
        try:
            b64 = _encode_image_to_base64(image_path)
        except Exception:
            return False

        prompt = (
            "你是一名负责排查海报中文字质量的审核助手。下面是一张图片（封面或尾页）和一段规则，请你只判断图片中的文字是否满足规则。\n\n"
            "规则：\n"
            "- 图片中的文字应该以正常的简体中文为主，能够被人类清晰阅读。\n"
            "- 不应出现以下情况：\n"
            "  - 大量乱码、无意义的字符组合；\n"
            "  - 主要内容为非简体中文（如大段英文、其他语言或繁体字占主导）；\n"
            "  - 明显不符合正式中文公文/宣传海报常见的用词和排版习惯；\n"
            "  - 出现明显的模板/占位/页面标签类无关文字，例如英文或拼音形式的“cover”“end page”“page”“slide”“template”等，或含义类似的页码/版式说明。\n\n"
            "允许少量常见数字和标点，以及极少量与政策内容直接相关的英文缩写（如 AI、PDF 等），但整体应当以可读的简体中文句子为主，\n"
            "且不应出现任何与本次政策解读无直接关系的模板占位词、页面标签或装饰性说明文字。\n\n"
            "如果整体文字清晰可读、以简体中文为主、没有明显乱码或大段外文，也不存在上述任何模板/占位/页面标签类无关文字，请只输出：通过\n"
            "否则请只输出：不通过\n"
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
    *,
    cover_image: str | None = None,
    tail_image: str | None = None,
) -> str:
    logger.info("[poster_pipeline] Concatenating posters into long image: %s", output_path)
    image_paths: List[str] = []

    if cover_image and isinstance(cover_image, str) and cover_image:
        image_paths.append(cover_image)

    for dim_key in POLICY_QUESTIONS.keys():
        info = posters.get(dim_key) or {}
        img_path = info.get("image_result")
        if isinstance(img_path, str) and img_path:
            image_paths.append(img_path)

    if tail_image and isinstance(tail_image, str) and tail_image:
        image_paths.append(tail_image)

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

    original_widths = [img.width for img in images]
    target_width = min(original_widths)

    resized_images: List[Image.Image] = []
    total_height = 0
    for img in images:
        if img.width == target_width:
            resized = img
        else:
            new_height = int(img.height * target_width / img.width)
            resized = img.resize((target_width, new_height), resample=Image.LANCZOS)
        resized_images.append(resized)
        total_height += resized.height

    long_img = Image.new("RGB", (target_width, total_height), color="white")

    y_offset = 0
    for img in resized_images:
        long_img.paste(img, (0, y_offset))
        y_offset += img.height

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    long_img.save(output_path)
    try:
        long_img.close()
    except Exception:
        pass

    for img in images:
        try:
            img.close()
        except Exception:
            pass
    print(f"[poster_pipeline] 长图拼接完成，已保存至: {output_path}")
    return output_path
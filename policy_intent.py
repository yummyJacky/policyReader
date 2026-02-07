import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


SessionIntent = Literal["text_only", "text_and_poster", "poster_only"]


def _detect_intent_prompt(message: str) -> str:
    return (
        "你是一个接口的意图识别器。请判断用户输入属于以下三类的哪一类，并只输出对应的标签：\n"
        "- text_only: 只需要生成解读文字/问答文字，不需要输出海报图片。\n"
        "- text_and_poster: 需要生成解读文字，并且需要输出海报图片。\n"
        "- poster_only: 只需要输出海报图片，不需要输出解读文字。\n\n"
        "输出要求：只能输出 text_only / text_and_poster / poster_only 其中之一，不要输出其他任何文字、标点或解释。\n\n"
        "示例：\n"
        "用户：请解读这份政策\n"
        "输出：text_only\n\n"
        "用户：请解读这份政策并生成配图\n"
        "输出：text_and_poster\n\n"
        "用户：只需要海报，不要文字\n"
        "输出：poster_only\n\n"
        "用户：门槛指标有哪些？\n"
        "输出：text_only\n\n"
        f"用户：{message}\n"
        "输出："
    )


def _detect_intent_rule(message: str) -> SessionIntent:
    msg = (message or "").strip().lower()
    if not msg:
        return "text_only"

    poster_words = ["海报", "配图", "图片", "生成图", "出图", "画图", "封面", "长图"]
    text_words = ["解读", "解析", "总结", "提取", "说明", "哪些", "怎么", "如何", "是否", "为什么"]

    has_poster = any(w in msg for w in poster_words)
    has_text = any(w in msg for w in text_words)

    if has_poster and has_text:
        return "text_and_poster"
    if has_poster:
        return "poster_only"
    return "text_only"


def detect_intent(message: str) -> SessionIntent:
    llm_model = os.getenv("POLICY_INTENT_MODEL", "doubao-seed-1-6-flash-250828")
    ark_key = os.getenv("ARK_API_KEY")

    if not ark_key:
        return _detect_intent_rule(message)

    client = OpenAI(
        base_url=os.getenv("POLICY_INTENT_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        api_key=ark_key,
    )

    prompt = _detect_intent_prompt(message)

    try:
        response = client.responses.create(
            model=llm_model,
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
            raw = (response.output[1].content[0].text or "").strip().lower()  # type: ignore[union-attr,index]
        except Exception:
            raw = ""

        if "text_and_poster" in raw:
            return "text_and_poster"
        if "poster_only" in raw:
            return "poster_only"
        if "text_only" in raw:
            return "text_only"
        return _detect_intent_rule(message)
    except Exception:
        return _detect_intent_rule(message)

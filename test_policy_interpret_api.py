import argparse
import json
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests
from requests import Session
from requests.exceptions import ChunkedEncodingError


DONE_MARK = "[DONE]"


def call_intent_api(*, sess: Session, base_url: str, message: str, timeout: int = 60) -> str:
    url = base_url.rstrip("/") + "/api/policy-intent"
    resp = sess.post(url, data={"message": message}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    intent = str(data.get("intent") or "").strip()
    if intent not in {"text_only", "text_and_poster", "poster_only"}:
        raise AssertionError(f"Unexpected intent: {intent!r}, raw={data}")
    return intent


def iter_stream_lines(resp: requests.Response) -> Iterator[str]:
    resp.raise_for_status()
    try:
        for line in resp.iter_lines(decode_unicode=True):
            if line is None:
                continue
            line = line.strip()
            if not line:
                continue
            yield line
    except ChunkedEncodingError as e:
        yield json.dumps({"type": "ERROR", "messageType": "TEXT", "content": json.dumps({"code": 599, "message": f"stream broken: {e}"}, ensure_ascii=False)}, ensure_ascii=False)


def parse_line(line: str) -> Tuple[Optional[Dict[str, Any]], bool, bool]:
    """Return (json_obj_or_raw, is_done, is_error)."""
    if line == DONE_MARK:
        return None, True, False
    try:
        obj = json.loads(line)
    except Exception:
        return {"_raw": line}, False, False

    is_error = obj.get("type") == "ERROR"
    return obj, False, is_error


def call_api(
    *,
    sess: Session,
    base_url: str,
    endpoint: str,
    message: str,
    file_path: Optional[str],
    out_dir: str,
    round_name: str,
    round_index: int,
    timeout: int = 600,
) -> Tuple[bool, Dict[str, Any]]:
    url = base_url.rstrip("/") + endpoint
    data = {"message": message}

    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_name = "".join([c if (c.isalnum() or c in "-_") else "_" for c in round_name])
    round_dir = os.path.join(out_dir, f"{round_index:02d}_{safe_name}_{ts}")
    os.makedirs(round_dir, exist_ok=True)
    stream_path = os.path.join(round_dir, "stream.jsonl")
    meta_path = os.path.join(round_dir, "meta.json")

    got_done = False
    got_error = False
    got_text = False
    got_image = False
    doc_tags: List[str] = []
    from_cache_flags: List[bool] = []

    files = None
    fh = None
    if file_path:
        filename = os.path.basename(file_path)
        fh = open(file_path, "rb")
        files = {"file": (filename, fh)}

    try:
        with open(stream_path, "w", encoding="utf-8") as fp:
            with sess.post(url, data=data, files=files, stream=True, timeout=timeout) as resp:
                for line in iter_stream_lines(resp):
                    fp.write(line + "\n")

                    obj, is_done, is_error = parse_line(line)
                    if is_done:
                        got_done = True
                        break

                    if is_error:
                        got_error = True

                    if obj and obj.get("type") == "RESPONSE":
                        if obj.get("messageType") == "TEXT":
                            got_text = True
                        if obj.get("messageType") == "IMAGE":
                            got_image = True
                            content = obj.get("content") or {}
                            if isinstance(content, dict):
                                if content.get("doc_tag"):
                                    doc_tags.append(str(content.get("doc_tag")))
                                if "from_cache" in content:
                                    from_cache_flags.append(bool(content.get("from_cache")))

                    print(line)
    finally:
        if fh is not None:
            fh.close()

    if got_error:
        print("[test] got ERROR line")

    meta: Dict[str, Any] = {
        "round_name": round_name,
        "round_index": round_index,
        "message": message,
        "file_path": file_path,
        "url": url,
        "round_dir": round_dir,
        "stream_path": stream_path,
        "got_text": got_text,
        "got_image": got_image,
        "doc_tags": doc_tags,
        "from_cache_flags": from_cache_flags,
        "got_error": got_error,
        "got_done": got_done,
    }

    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    return got_done, meta


def assert_intent_output(case_name: str, meta: Dict[str, Any]) -> None:
    if case_name == "text_only":
        assert meta["got_text"], "text_only should include TEXT"
        assert not meta["got_image"], "text_only should not include IMAGE"
    elif case_name == "poster_only":
        assert meta["got_image"], "poster_only should include IMAGE"
        assert not meta["got_text"], "poster_only should not include TEXT"
    elif case_name == "text_and_poster":
        assert meta["got_text"], "text_and_poster should include TEXT"
        assert meta["got_image"], "text_and_poster should include IMAGE"
    else:
        raise AssertionError(f"Unknown case: {case_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test /api/policy-interpre streaming API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI base url")
    parser.add_argument("--endpoint", default="/api/policy-interpre", help="API endpoint path")
    parser.add_argument("--file", required=True, help="Policy file path")
    parser.add_argument("--file2", default=None, help="Second policy file path (optional, for cache reset test)")
    parser.add_argument("--out-dir", default="test_runs", help="Directory to save per-round outputs")
    args = parser.parse_args()

    root_ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, root_ts)
    os.makedirs(out_dir, exist_ok=True)

    sess = requests.Session()

    # 1) 意图识别是否正确：只调用意图识别接口，不跑解析流程
    cases = [
        ("text_only", "解读这个政策文件"),
        ("text_and_poster", "解读政策文件并生成海报展示"),
        ("poster_only", "将解读的内容生成海报展示"),
    ]

    for name, msg in cases:
        print(f"\n=== Intent Case: {name} (intent-only endpoint) ===")
        predicted = call_intent_api(sess=sess, base_url=args.base_url, message=msg)
        if predicted != name:
            raise SystemExit(f"Intent case {name} failed: predicted={predicted}")

    # 2) 单轮同时要求文字和海报（with file）：同一轮应同时输出 TEXT + IMAGE
    print("\n=== Single-turn Case: text_and_poster (with file) ===")
    done_tp, meta_tp = call_api(
        sess=sess,
        base_url=args.base_url,
        endpoint=args.endpoint,
        message="解读政策文件并生成海报展示",
        file_path=args.file,
        out_dir=out_dir,
        round_name="single_turn_text_and_poster_with_file",
        round_index=5,
    )
    if not done_tp:
        raise SystemExit("Single-turn text_and_poster failed: did not receive [DONE]")
    assert_intent_output("text_and_poster", meta_tp)

    # 3) 第一轮解析、第二轮生成海报（不传文件）：应基于第一轮回答生成海报
    print("\n=== Multi-turn Case: parse(text_only) then poster_only(no file) ===")
    done1, meta1 = call_api(
        sess=sess,
        base_url=args.base_url,
        endpoint=args.endpoint,
        message="解读这个政策文件",
        file_path=args.file,
        out_dir=out_dir,
        round_name="turn1_text_only_with_file",
        round_index=10,
    )
    if not done1:
        raise SystemExit("Turn1 failed: did not receive [DONE]")
    assert_intent_output("text_only", meta1)

    done2, meta2 = call_api(
        sess=sess,
        base_url=args.base_url,
        endpoint=args.endpoint,
        message="将解读的内容生成海报展示",
        file_path=None,
        out_dir=out_dir,
        round_name="turn2_poster_only_no_file",
        round_index=11,
    )
    if not done2:
        raise SystemExit("Turn2 failed: did not receive [DONE]")
    assert_intent_output("poster_only", meta2)
    if meta2["from_cache_flags"]:
        assert all(meta2["from_cache_flags"]), "Expected poster images generated from_cache in turn2"
    if meta2["doc_tags"]:
        assert len(set(meta2["doc_tags"])) == 1, "Expected all posters to share same doc_tag in turn2"

    # 3) 新文件时：应对新文件解析，并刷新缓存（doc_tag变化）；随后不传文件出海报应仍 from_cache
    file2 = args.file2
    if file2:
        print("\n=== Cache Reset Case: upload new file then poster_only(no file) ===")
        done3, meta3 = call_api(
            sess=sess,
            base_url=args.base_url,
            endpoint=args.endpoint,
            message="解读这个政策文件",
            file_path=file2,
            out_dir=out_dir,
            round_name="newfile_turn_text_only_with_file2",
            round_index=20,
        )
        if not done3:
            raise SystemExit("New-file turn failed: did not receive [DONE]")
        assert_intent_output("text_only", meta3)

        done4, meta4 = call_api(
            sess=sess,
            base_url=args.base_url,
            endpoint=args.endpoint,
            message="将解读的内容生成海报展示",
            file_path=None,
            out_dir=out_dir,
            round_name="newfile_turn_poster_only_no_file",
            round_index=21,
        )
        if not done4:
            raise SystemExit("New-file poster turn failed: did not receive [DONE]")
        assert_intent_output("poster_only", meta4)
        if meta4["from_cache_flags"]:
            assert all(meta4["from_cache_flags"]), "Expected poster images generated from_cache after new upload"
        if meta4["doc_tags"] and meta2["doc_tags"]:
            assert set(meta4["doc_tags"]) != set(meta2["doc_tags"]), "Expected doc_tag to change after uploading new file"

    print("\nAll tests finished.")


if __name__ == "__main__":
    main()

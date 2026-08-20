#!/usr/bin/env python3
"""并行发送中文请求到 PD 代理，保存每个请求的返回结果。

用法示例:
  python3 request_hetero_test.py \
      --proxy-url http://7.246.78.76:9000 \
      --concurrency 8 \
      --num-requests 32 \
      --outdir ./hetero_test_results/hetero
"""
import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

# 中文测试句，覆盖短句/长句/技术/历史/常识等类型。
CHINESE_PROMPTS = [
    "请用一句话介绍杭州西湖。",
    "解释一下什么是人工智能，并给出两个生活中的例子。",
    "用不超过五十个字概括《三国演义》的故事。",
    "请列举三种常见的排序算法，并说明它们的时间复杂度。",
    "为什么天空是蓝色的？请用通俗的语言解释。",
    "写一段关于春天的中文短文，大约一百字。",
    "什么是深度学习？它和传统机器学习有什么区别？",
    "请把下面的意思翻译成更正式的中文表达：我们明天开会讨论这个方案。",
    "请解释“知行合一”的含义。",
    "生成一份周末出游的简短计划，包含时间和地点。",
    "如何看待远程办公的优势和劣势？请分点说明。",
    "请用中文解释量子计算的基本思想，不要超过一百五十字。",
    "写一首五言绝句，主题是秋天。",
    "介绍一下中国高铁的发展历程。",
    "什么是区块链？它主要解决什么问题？",
    "请总结一下《红楼梦》中贾宝玉和林黛玉的关系。",
    "请给出三条提高学习效率的建议。",
    "解释一下大语言模型中的注意力机制。",
    "什么是绿色能源？列举三种常见的绿色能源。",
    "请写一段产品介绍，产品是一款智能手表。",
    "请解释“不积跬步，无以至千里”的意思。",
    "什么是云计算？它有哪些常见的服务模式？",
    "请用中文写一个简短的笑话。",
    "介绍一下太阳系中的八大行星。",
    "什么是气候变化？普通人可以做哪些事情来应对它？",
    "请解释一下深度优先搜索和广度优先搜索的区别。",
    "请写一封简短的感谢信，感谢同事在工作中的帮助。",
    "什么是元宇宙？它和虚拟现实有什么关系？",
    "请解释“塞翁失马，焉知非福”的道理。",
    "请给出一份健康饮食的一日三餐建议。",
    "什么是自然语言处理？举两个常见应用。",
    "请用一句话评价中国航天近年来的发展。",
]


def build_payload(prompt: str, request_id: int) -> dict:
    return {
        "model": "dsv4",
        "messages": [
            {"role": "system", "content": "你是一个乐于助人的中文助手。"},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": False,
        "request_id": f"hetero-test-{request_id}",
    }


def send_request(backend_url: str, payload: dict, timeout: int):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        backend_url.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    begin = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    latency_ms = (time.time() - begin) * 1000.0
    return raw, latency_ms


def parse_choices(raw: str):
    """Best-effort extract generated text from an OpenAI-style response."""
    try:
        obj = json.loads(raw)
        return obj["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-url", default="http://7.246.78.76:9000")
    parser.add_argument(
        "--target-urls", default=None,
        help=("Comma-separated backend base URLs. When set, requests are "
              "sent round-robin directly to these vLLM engine ports instead "
              "of the proxy."),
    )
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600,
                        help="Single request timeout in seconds.")
    parser.add_argument("--outdir", default="hetero_test_results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.target_urls:
        backend_urls = [
            url.strip().rstrip("/")
            for url in args.target_urls.split(",")
            if url.strip()
        ]
    else:
        backend_urls = [args.proxy_url.rstrip("/")]
    if not backend_urls:
        raise SystemExit("No backend URLs configured.")

    prompts = [
        CHINESE_PROMPTS[i % len(CHINESE_PROMPTS)]
        for i in range(args.num_requests)
    ]

    results = []

    def run_one(idx_and_prompt):
        idx, prompt = idx_and_prompt
        payload = build_payload(prompt, idx)
        backend_url = backend_urls[idx % len(backend_urls)]
        begin = time.time()
        try:
            raw, latency_ms = send_request(backend_url, payload, args.timeout)
            content = parse_choices(raw)
            status = "ok"
            error = None
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            latency_ms = (time.time() - begin) * 1000.0
            content = None
            status = f"http_error_{exc.code}"
            error = raw[:2000]
        except Exception as exc:  # noqa: BLE001
            raw = ""
            latency_ms = (time.time() - begin) * 1000.0
            content = None
            status = "error"
            error = f"{type(exc).__name__}: {exc}"

        record = {
            "index": idx,
            "prompt": prompt,
            "backend": backend_url,
            "status": status,
            "latency_ms": round(latency_ms, 1),
            "generated_text": content,
            "error": error,
            "raw_response": raw[:20000],
        }
        (outdir / f"request_{idx:03d}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return record

    print(f"开始向 {len(backend_urls)} 个后端并行发送 "
          f"{args.num_requests} 个中文请求，并发度 {args.concurrency}。")
    for backend_url in backend_urls:
        print(f"  backend: {backend_url}")
    begin = time.time()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        results = list(executor.map(run_one, enumerate(prompts)))
    elapsed = time.time() - begin

    ok_count = sum(1 for r in results if r["status"] == "ok")
    failed_count = len(results) - ok_count
    summary = {
        "proxy_url": args.proxy_url,
        "backend_urls": backend_urls,
        "num_requests": args.num_requests,
        "concurrency": args.concurrency,
        "elapsed_seconds": round(elapsed, 1),
        "ok_count": ok_count,
        "failed_count": failed_count,
        "results": [
            {
                "index": r["index"],
                "status": r["status"],
                "latency_ms": r["latency_ms"],
                "generated_text": r["generated_text"],
            }
            for r in results
        ],
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"完成：成功 {ok_count}/{args.num_requests}，失败 {failed_count}，"
          f"耗时 {elapsed:.1f}s。")
    print(f"结果目录：{outdir.resolve()}")
    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

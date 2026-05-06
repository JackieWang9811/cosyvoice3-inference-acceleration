import os
import time
import json
import math
import queue
import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def prepare_speaker(base_url, prompt_text, prompt_wav, spk_id):
    url = f"{base_url}/prepare_zero_shot_spk"
    data = {
        "prompt_text": prompt_text,
        "prompt_wav": prompt_wav,
        "zero_shot_spk_id": spk_id,
    }
    r = requests.post(url, data=data, timeout=300)
    r.raise_for_status()
    return r.json()


def run_one_request(base_url, tts_text, spk_id, timeout=600):
    url = f"{base_url}/inference_zero_shot_bench"
    data = {
        "tts_text": tts_text,
        "zero_shot_spk_id": spk_id,
        "stream": "true",
    }

    result = {
        "ok": False,
        "status_code": None,
        "elapsed_sec": None,
        "ttft_sec": None,
        "bytes": 0,
        "error": None,
    }

    t0 = time.perf_counter()
    first_chunk_time = None

    try:
        with requests.post(url, data=data, stream=True, timeout=timeout) as r:
            result["status_code"] = r.status_code
            r.raise_for_status()

            total_bytes = 0
            for chunk in r.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                total_bytes += len(chunk)

            t1 = time.perf_counter()
            result["ok"] = True
            result["elapsed_sec"] = t1 - t0
            result["ttft_sec"] = None if first_chunk_time is None else (first_chunk_time - t0)
            result["bytes"] = total_bytes

    except Exception as e:
        result["elapsed_sec"] = time.perf_counter() - t0
        result["error"] = str(e)

    return result


def summarize(results):
    total = len(results)
    oks = [r for r in results if r["ok"]]
    fails = [r for r in results if not r["ok"]]

    elapsed = [r["elapsed_sec"] for r in oks if r["elapsed_sec"] is not None]
    ttfts = [r["ttft_sec"] for r in oks if r["ttft_sec"] is not None]
    sizes = [r["bytes"] for r in oks]

    summary = {
        "total_requests": total,
        "success": len(oks),
        "failed": len(fails),
        "success_rate": 0.0 if total == 0 else len(oks) / total,
        "elapsed_mean": statistics.mean(elapsed) if elapsed else None,
        "elapsed_p50": percentile(elapsed, 50) if elapsed else None,
        "elapsed_p95": percentile(elapsed, 95) if elapsed else None,
        "elapsed_p99": percentile(elapsed, 99) if elapsed else None,
        "ttft_mean": statistics.mean(ttfts) if ttfts else None,
        "ttft_p50": percentile(ttfts, 50) if ttfts else None,
        "ttft_p95": percentile(ttfts, 95) if ttfts else None,
        "bytes_mean": statistics.mean(sizes) if sizes else None,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument("--prompt_text", type=str, required=True)
    parser.add_argument("--prompt_wav", type=str, required=True)
    parser.add_argument("--tts_text", type=str, required=True)
    parser.add_argument("--spk_id", type=str, default="bench_spk")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--num_requests", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print("=" * 80)
    print("1) Prepare zero-shot speaker")
    print("=" * 80)
    resp = prepare_speaker(
        base_url=base_url,
        prompt_text=args.prompt_text,
        prompt_wav=args.prompt_wav,
        spk_id=args.spk_id,
    )
    print(json.dumps(resp, ensure_ascii=False, indent=2))

    print("=" * 80)
    print("2) Run concurrent benchmark")
    print("=" * 80)
    print(f"base_url      : {base_url}")
    print(f"concurrency   : {args.concurrency}")
    print(f"num_requests  : {args.num_requests}")
    print(f"spk_id        : {args.spk_id}")
    print()

    results = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [
            ex.submit(
                run_one_request,
                base_url,
                args.tts_text,
                args.spk_id,
                args.timeout
            )
            for _ in range(args.num_requests)
        ]

        for idx, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            print(
                f"[{idx:03d}/{args.num_requests}] "
                f"ok={r['ok']} "
                f"status={r['status_code']} "
                f"ttft={None if r['ttft_sec'] is None else round(r['ttft_sec'], 4)} "
                f"elapsed={None if r['elapsed_sec'] is None else round(r['elapsed_sec'], 4)} "
                f"bytes={r['bytes']} "
                f"err={r['error']}"
            )

    wall = time.perf_counter() - t0
    summary = summarize(results)
    summary["wall_clock_sec"] = wall
    summary["throughput_req_per_sec"] = 0.0 if wall == 0 else (args.num_requests / wall)

    print("\n" + "=" * 80)
    print("3) Summary")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.out_json:
        payload = {
            "config": vars(args),
            "summary": summary,
            "results": results,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to: {args.out_json}")


if __name__ == "__main__":
    main()
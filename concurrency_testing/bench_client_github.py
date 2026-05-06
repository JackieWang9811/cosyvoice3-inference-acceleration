import argparse
import logging
import requests
import torch
import torchaudio
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def prepare_speaker(args):
    url = f"http://{args.host}:{args.port}/prepare_zero_shot_spk"
    payload = {
        "prompt_text": args.prompt_text,
        "prompt_wav": args.prompt_wav,
        "zero_shot_spk_id": args.zero_shot_spk_id,
    }
    resp = requests.post(url, data=payload, timeout=args.timeout)
    resp.raise_for_status()
    logging.info(f"speaker prepared: {resp.json()}")


def single_inference(index, args, target_sr):
    url = f"http://{args.host}:{args.port}/inference_zero_shot_bench"
    payload = {
        "tts_text": args.tts_text,
        "zero_shot_spk_id": args.zero_shot_spk_id,
        "stream": str(args.stream).lower(),
    }

    start_time = time.time()
    try:
        response = requests.post(url, data=payload, stream=True, timeout=args.timeout)
        response.raise_for_status()

        tts_audio = b""
        first_byte_time = None

        for chunk in response.iter_content(chunk_size=16000):
            if not chunk:
                continue
            if first_byte_time is None:
                first_byte_time = time.time()
            tts_audio += chunk

        end_time = time.time()

        tts_speech = torch.from_numpy(
            np.frombuffer(tts_audio, dtype=np.int16).copy()
        ).unsqueeze(0)

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, f"result_{index}.wav")
            torchaudio.save(save_path, tts_speech, target_sr)

        audio_len = tts_speech.shape[1] / target_sr if tts_speech.shape[1] > 0 else 0.0
        latency = end_time - start_time
        rtf = latency / audio_len if audio_len > 0 else 0.0
        ttfb = first_byte_time - start_time if first_byte_time else 0.0

        return {
            "index": index,
            "success": True,
            "latency": latency,
            "ttfb": ttfb,
            "rtf": rtf,
            "audio_len": audio_len,
        }

    except Exception as e:
        logging.error(f"Request {index} failed: {e}")
        return {
            "index": index,
            "success": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="CosyVoice zero-shot benchmark client")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50000)

    parser.add_argument("--tts_text", type=str, default="你好，你好，往前一步。")
    parser.add_argument("--prompt_text", type=str, default="希望你以后能够做的比我还好呦。")
    parser.add_argument("--prompt_wav", type=str, required=True)
    parser.add_argument("--zero_shot_spk_id", type=str, default="bench_spk")
    parser.add_argument("--stream", action="store_true")

    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--total_requests", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output_dir", type=str, default="test_results")
    parser.add_argument("--target_sr", type=int, default=22050)
    args = parser.parse_args()

    logging.info("preparing zero-shot speaker ...")
    prepare_speaker(args)

    logging.info(
        f"开始并发测试: 总请求={args.total_requests}, 并发数={args.concurrency}"
    )

    results = []
    start_test_time = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(single_inference, i, args, args.target_sr)
            for i in range(args.total_requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    total_duration = time.time() - start_test_time

    success_results = [r for r in results if r["success"]]
    failed_count = len(results) - len(success_results)

    logging.info("-" * 30)
    logging.info(f"测试完成，耗时: {total_duration:.2f}s")
    logging.info(f"成功: {len(success_results)}, 失败: {failed_count}")

    if success_results:
        avg_latency = sum(r["latency"] for r in success_results) / len(success_results)
        avg_ttfb = sum(r["ttfb"] for r in success_results) / len(success_results)
        avg_rtf = sum(r["rtf"] for r in success_results) / len(success_results)
        qps = len(results) / total_duration if total_duration > 0 else 0.0

        logging.info(f"平均延迟 (Latency): {avg_latency:.2f}s")
        logging.info(f"平均首包延迟 (TTFB): {avg_ttfb:.2f}s")
        logging.info(f"平均实时率 (RTF): {avg_rtf:.4f}")
        logging.info(f"吞吐量 (QPS): {qps:.2f} req/s")
    logging.info("-" * 30)


if __name__ == "__main__":
    main()
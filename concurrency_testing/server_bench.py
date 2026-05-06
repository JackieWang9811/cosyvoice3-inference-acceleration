import os
import sys
import time
import uuid
import argparse
import logging
from typing import Optional

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 按你的 CosyVoice 仓库位置调整
# 例如:
#   /home/b24zll/data/wjq_data/CosyVoice
COSYVOICE_ROOT = os.environ.get("COSYVOICE_ROOT", ROOT_DIR)

sys.path.append(COSYVOICE_ROOT)
sys.path.append(os.path.join(COSYVOICE_ROOT, "third_party", "Matcha-TTS"))

# ---- vLLM model registry ----
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

# ---- CosyVoice ----
from cosyvoice.cli.cosyvoice import AutoModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cosyvoice = None
SPEAKER_READY = False
DEFAULT_SPK_ID = "bench_spk"


def pcm16le_header(sample_rate: int, channels: int = 1, bits_per_sample: int = 16):
    """
    如果你后面想换成 wav 封装再扩展。
    当前先纯返回原始音频 bytes，不加 wav header。
    """
    return b""


@app.get("/health")
def health():
    return JSONResponse({
        "ok": True,
        "speaker_ready": SPEAKER_READY,
        "sample_rate": getattr(cosyvoice, "sample_rate", None)
    })


@app.post("/prepare_zero_shot_spk")
def prepare_zero_shot_spk(
    prompt_text: str = Form(...),
    prompt_wav: str = Form(...),
    zero_shot_spk_id: str = Form(DEFAULT_SPK_ID),
):
    """
    预加载一次 speaker，避免每个请求重复处理 prompt wav。
    """
    global SPEAKER_READY
    global DEFAULT_SPK_ID

    if cosyvoice is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    t0 = time.perf_counter()
    try:
        if not os.path.exists(prompt_wav):
            raise FileNotFoundError(f"prompt_wav not found: {prompt_wav}")

        # 关键：复用 zero-shot speaker
        # 官方 issue/用法里是每个请求都直接走 inference_zero_shot；
        # 这里我们改成先缓存 speaker，再做并发测试。
        cosyvoice.add_zero_shot_spk(prompt_text, prompt_wav, zero_shot_spk_id)
        SPEAKER_READY = True
        DEFAULT_SPK_ID = zero_shot_spk_id

        return JSONResponse({
            "ok": True,
            "zero_shot_spk_id": zero_shot_spk_id,
            "elapsed_sec": round(time.perf_counter() - t0, 4)
        })
    except Exception as e:
        logging.exception("prepare_zero_shot_spk failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference_zero_shot_bench")
def inference_zero_shot_bench(
    tts_text: str = Form(...),
    zero_shot_spk_id: Optional[str] = Form(None),
    stream: bool = Form(True),
):
    """
    并发测试接口：
    - 默认复用已缓存 speaker
    - 返回流式音频 bytes
    """
    if cosyvoice is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    spk_id = zero_shot_spk_id or DEFAULT_SPK_ID
    if not SPEAKER_READY:
        raise HTTPException(status_code=400, detail="Speaker not prepared yet")

    request_id = str(uuid.uuid4())[:8]
    logging.info(f"[{request_id}] recv request, len(tts_text)={len(tts_text)}, stream={stream}, spk_id={spk_id}")

    def audio_generator():
        t0 = time.perf_counter()
        first_chunk_sent = False
        chunk_count = 0
        total_bytes = 0

        try:
            # 这里直接走复用音色的 zero-shot 推理
            for out in cosyvoice.inference_zero_shot(
                tts_text,
                "",
                "",
                zero_shot_spk_id=spk_id,
                stream=stream
            ):
                speech = out["tts_speech"]  # torch.Tensor [1, T] 或 [T]
                if speech is None:
                    continue

                if hasattr(speech, "detach"):
                    speech = speech.detach().cpu().float().numpy()

                speech = np.asarray(speech).squeeze()
                if speech.ndim != 1:
                    speech = speech.reshape(-1)

                # float32 [-1,1] -> int16 bytes
                speech = np.clip(speech, -1.0, 1.0)
                pcm16 = (speech * 32767.0).astype(np.int16).tobytes()

                if len(pcm16) == 0:
                    continue

                if not first_chunk_sent:
                    ttft = time.perf_counter() - t0
                    logging.info(f"[{request_id}] first_chunk_sec={ttft:.4f}")
                    first_chunk_sent = True

                chunk_count += 1
                total_bytes += len(pcm16)
                yield pcm16

            elapsed = time.perf_counter() - t0
            logging.info(
                f"[{request_id}] done elapsed={elapsed:.4f}s, chunks={chunk_count}, bytes={total_bytes}"
            )

        except Exception as e:
            logging.exception(f"[{request_id}] generation failed: {e}")
            raise

    return StreamingResponse(
        audio_generator(),
        media_type="application/octet-stream"
    )


def main():
    global cosyvoice

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--load_vllm", action="store_true")
    parser.add_argument("--load_trt", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    logging.info("Initializing CosyVoice model...")
    t0 = time.perf_counter()

    cosyvoice = AutoModel(
        model_dir=args.model_dir,
        load_vllm=args.load_vllm,
        load_trt=args.load_trt,
        fp16=args.fp16
    )

    logging.info(
        f"Model loaded in {time.perf_counter() - t0:.2f}s | "
        f"sample_rate={getattr(cosyvoice, 'sample_rate', None)}"
    )

    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import json
import argparse
import subprocess
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import time
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

INFERENCE_MODES = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
INSTRUCT_DICT = {'预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
                 '3s极速复刻': '1. 选择/录制 prompt 音频（≤30s）\n2. 输入 prompt 文本\n3. 点击生成音频按钮',
                 '跨语种复刻': '1. 选择/录制 prompt 音频（≤30s）\n2. 点击生成音频按钮',
                 '自然语言控制': '1. 选择预训练音色\n2. 输入 instruct 文本\n3. 点击生成音频按钮'}
STREAM_MODE_LIST = [('否', False), ('是', True)]
PROMPT_SR = 16000
MODEL_ACCEL_MODES = ['FP32（原始）', 'AWQ 量化', 'Wanda 剪枝']

# ── helpers ──────────────────────────────────────────────────────────────

default_data = np.zeros(24000)


def generate_seed():
    seed = random.randint(1, 100000000)
    return {"__type__": "update", "value": seed}


def change_instruction(mode):
    return INSTRUCT_DICT[mode]


def load_benchmark_results():
    """Scan common paths for benchmark / model-info results."""
    info = {}
    # quantization model sizes
    for variant in ['cv3_qwen2_backbone_fp32',
                     'cv3_qwen2_backbone_awq',
                     'cv3_qwen2_backbone_awq_v2']:
        d = Path('quantization/quantization_awq') / variant
        if d.exists():
            sf = list(d.glob('*.safetensors'))
            if sf:
                info[variant] = sf[0].stat().st_size / (1024 ** 2)  # MB
    # pruning
    d = Path('pruning/wanda/cv3_qwen2_backbone_wanda_s30')
    if d.exists():
        sf = list(d.glob('*.safetensors'))
        if sf:
            info['cv3_qwen2_backbone_wanda_s30'] = sf[0].stat().st_size / (1024 ** 2)

    # try to run analyze script
    try:
        result = subprocess.run(
            [sys.executable, 'tools/analyze_cosyvoice3_model.py',
             '--model_dir', 'pretrained_models/Fun-CosyVoice3-0.5B'],
            capture_output=True, text=True, timeout=30)
        info['analyze_output'] = result.stdout
    except Exception:
        info['analyze_output'] = None
    return info


# ── TTS inference ────────────────────────────────────────────────────────

def generate_audio(tts_text, mode, sft_dropdown, prompt_text,
                   prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed, model_dir):
    """Generator that yields (sample_rate, audio_array) chunks."""
    try:
        from cosyvoice.cli.cosyvoice import AutoModel
        from cosyvoice.utils.common import set_all_random_seed
    except ImportError:
        gr.Warning('cosyvoice 模块未找到，请确认环境配置正确')
        yield (24000, default_data)
        return

    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None

    # ── model loading ────────────────────────────────────────────────
    if not os.path.exists(model_dir):
        gr.Warning(f'模型目录 {model_dir} 不存在，请确认路径')
        yield (24000, default_data)
        return

    try:
        cosyvoice = AutoModel(model_dir=model_dir)
    except Exception as e:
        gr.Warning(f'模型加载失败: {e}')
        yield (24000, default_data)
        return

    # ── input validation ─────────────────────────────────────────────
    if mode in ['自然语言控制']:
        if instruct_text == '':
            gr.Warning('自然语言控制模式请输入 instruct 文本')
            yield (cosyvoice.sample_rate, default_data)
            return
        if prompt_wav is not None or prompt_text != '':
            gr.Info('自然语言控制模式，prompt 音频/文本将被忽略')

    if mode in ['跨语种复刻']:
        if instruct_text != '':
            gr.Info('跨语种复刻模式，instruct 文本将被忽略')
        if prompt_wav is None:
            gr.Warning('跨语种复刻模式请提供 prompt 音频')
            yield (cosyvoice.sample_rate, default_data)
            return
        gr.Info('跨语种复刻模式，请确保合成文本和 prompt 文本为不同语言')

    if mode in ['3s极速复刻', '跨语种复刻']:
        if prompt_wav is None:
            gr.Warning('prompt 音频为空')
            yield (cosyvoice.sample_rate, default_data)
            return

    if mode in ['预训练音色']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('预训练音色模式，prompt 文本/音频/instruct 文本将被忽略')
        if sft_dropdown == '':
            gr.Warning('没有可用的预训练音色')
            yield (cosyvoice.sample_rate, default_data)
            return

    if mode in ['3s极速复刻']:
        if prompt_text == '':
            gr.Warning('3s极速复刻模式请输入 prompt 文本')
            yield (cosyvoice.sample_rate, default_data)
            return
        if instruct_text != '':
            gr.Info('3s极速复刻模式，预训练音色/instruct 文本将被忽略')

    # ── run inference ────────────────────────────────────────────────
    set_all_random_seed(seed)

    try:
        if mode == '预训练音色':
            for i in cosyvoice.inference_sft(tts_text, sft_dropdown,
                                              stream=stream, speed=speed):
                yield (cosyvoice.sample_rate,
                       i['tts_speech'].numpy().flatten())
        elif mode == '3s极速复刻':
            for i in cosyvoice.inference_zero_shot(tts_text, prompt_text,
                                                    prompt_wav,
                                                    stream=stream, speed=speed):
                yield (cosyvoice.sample_rate,
                       i['tts_speech'].numpy().flatten())
        elif mode == '跨语种复刻':
            for i in cosyvoice.inference_cross_lingual(tts_text, prompt_wav,
                                                        stream=stream,
                                                        speed=speed):
                yield (cosyvoice.sample_rate,
                       i['tts_speech'].numpy().flatten())
        else:
            for i in cosyvoice.inference_instruct(tts_text, sft_dropdown,
                                                   instruct_text,
                                                   stream=stream, speed=speed):
                yield (cosyvoice.sample_rate,
                       i['tts_speech'].numpy().flatten())
    except Exception as e:
        gr.Warning(f'推理失败: {e}')
        yield (cosyvoice.sample_rate, default_data)


# ── benchmark tab ────────────────────────────────────────────────────────

def run_benchmark(model_dir, num_runs):
    """Naive latency benchmark for a given model."""
    try:
        from cosyvoice.cli.cosyvoice import AutoModel
        from cosyvoice.utils.common import set_all_random_seed
    except ImportError:
        return 'cosyvoice 模块未找到'

    if not os.path.exists(model_dir):
        return f'模型目录 {model_dir} 不存在'

    try:
        cosyvoice = AutoModel(model_dir=model_dir)
    except Exception as e:
        return f'模型加载失败: {e}'

    text = '我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。'
    spk = cosyvoice.list_available_spks()
    if len(spk) == 0:
        return '没有可用的预训练音色'
    spk = spk[0]

    latencies = []
    for _ in range(num_runs):
        set_all_random_seed(42)
        start = time.perf_counter()
        for res in cosyvoice.inference_sft(text, spk, stream=False, speed=1.0):
            _ = res['tts_speech']
        latencies.append(time.perf_counter() - start)

    latencies = latencies[1:]  # warmup
    avg = np.mean(latencies)
    std = np.std(latencies)
    audio_len = len(_['tts_speech']) / cosyvoice.sample_rate
    rtf = avg / audio_len if audio_len > 0 else 0

    report = (f'模型: {model_dir}\n'
              f'运行次数: {num_runs}（含 1 次 warmup）\n'
              f'平均耗时: {avg:.3f}s\n'
              f'标准差:   {std:.3f}s\n'
              f'音频时长: {audio_len:.1f}s\n'
              f'实时率:   {rtf:.3f}\n')
    return report


# ── Gradio Layout ────────────────────────────────────────────────────────

def build_ui(model_dir: str, sft_spk: list):
    with gr.Blocks(title='CosyVoice3 推理加速演示',
                   theme=gr.themes.Soft()) as demo:

        # ──────── header ─────────────────────────────────────────────
        gr.Markdown("""
        # 🎙️ CosyVoice3 推理加速演示

        基于 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 的推理加速方案，
        包含 AWQ 量化、Wanda 剪枝、投机解码等多种加速技术。
        """)

        # ──────── Tab 1: TTS 推理 ─────────────────────────────────
        with gr.Tab('🎤 TTS 推理'):
            gr.Markdown(f'**当前模型**: `{model_dir}`')

            with gr.Row():
                mode_radio = gr.Radio(choices=INFERENCE_MODES,
                                      label='推理模式',
                                      value=INFERENCE_MODES[0])
                instruction_text = gr.Text(label='操作步骤',
                                           value=INSTRUCT_DICT[INFERENCE_MODES[0]],
                                           scale=0.5)
                sft_dropdown = gr.Dropdown(choices=sft_spk,
                                            label='预训练音色',
                                            value=sft_spk[0] if sft_spk else '',
                                            scale=0.25)
                stream_radio = gr.Radio(choices=STREAM_MODE_LIST,
                                        label='流式推理',
                                        value=STREAM_MODE_LIST[0][1])
                speed_num = gr.Number(value=1.0,
                                      label='速度 (仅非流式)',
                                      minimum=0.5, maximum=2.0, step=0.1)

            tts_text = gr.Textbox(
                label='合成文本',
                lines=2,
                value='我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。')

            with gr.Row():
                prompt_wav_upload = gr.Audio(sources='upload',
                                              type='filepath',
                                              label='Prompt 音频上传（≥16kHz）')
                prompt_wav_record = gr.Audio(sources='microphone',
                                              type='filepath',
                                              label='录制 Prompt 音频')
            prompt_text = gr.Textbox(label='Prompt 文本',
                                      lines=1,
                                      placeholder='与 prompt 音频内容一致...',
                                      value='')
            instruct_text = gr.Textbox(label='Instruct 文本',
                                        lines=1,
                                        placeholder='用于自然语言控制模式...',
                                        value='')

            with gr.Row():
                seed_btn = gr.Button(value='🎲 随机种子')
                seed_num = gr.Number(value=0, label='随机种子', scale=2)
                generate_btn = gr.Button('🔊 生成音频', variant='primary', scale=3)

            audio_out = gr.Audio(label='合成音频', autoplay=True, streaming=True)

            # events
            seed_btn.click(generate_seed, inputs=[], outputs=seed_num)
            mode_radio.change(fn=change_instruction,
                              inputs=[mode_radio],
                              outputs=[instruction_text])
            generate_btn.click(
                fn=generate_audio,
                inputs=[tts_text, mode_radio, sft_dropdown,
                        prompt_text, prompt_wav_upload, prompt_wav_record,
                        instruct_text, seed_num, stream_radio, speed_num,
                        gr.State(model_dir)],
                outputs=[audio_out])

        # ──────── Tab 2: 加速对比 ────────────────────────────────
        with gr.Tab('⚡ 加速对比'):
            gr.Markdown("""
            ### 模型加速方案对比

            在下方选择模型目录和测试次数，运行基准测试查看性能数据。
            """)
            with gr.Row():
                bench_model_dir = gr.Textbox(
                    label='模型目录',
                    value=model_dir,
                    scale=3)
                bench_runs = gr.Number(
                    label='测试次数',
                    value=5,
                    minimum=2, maximum=20, step=1,
                    scale=1)
                bench_btn = gr.Button('🚀 运行基准测试', variant='primary', scale=2)
            bench_output = gr.Textbox(label='测试结果', lines=12)

            bench_btn.click(
                fn=run_benchmark,
                inputs=[bench_model_dir, bench_runs],
                outputs=[bench_output])

        # ──────── Tab 3: 模型信息 ────────────────────────────────
        with gr.Tab('📊 模型信息'):
            gr.Markdown("""
            ### 模型权重文件大小对比

            以下显示项目中各加速方案的模型权重大小（仅统计 `.safetensors` 文件）。
            """)
            info = load_benchmark_results()
            md_lines = ['| 模型变体 | 权重大小 (MB) |',
                        '|----------|---------------|']
            for name, size_mb in sorted(info.items()):
                if name == 'analyze_output':
                    continue
                tag = name.replace('cv3_qwen2_backbone_', '').replace('_', ' ')
                md_lines.append(f'| {tag} | {size_mb:.1f} MB |')

            if len(md_lines) > 2:
                gr.Markdown('\n'.join(md_lines))
            else:
                gr.Markdown('> 未检测到量化/剪枝后的模型文件，请先运行量化或剪枝脚本。')

            if info.get('analyze_output'):
                with gr.Accordion('模型结构分析', open=False):
                    gr.Textbox(value=info['analyze_output'],
                                label='analyze_cosyvoice3_model.py 输出',
                                lines=20)
            else:
                gr.Markdown(
                    '> 模型结构分析不可用（需 `pretrained_models/Fun-CosyVoice3-0.5B`）')

        # ──────── Tab 4: 项目概览 ────────────────────────────────
        with gr.Tab('📖 项目概览'):
            gr.Markdown("""
            ## 推理加速技术

            ### 1️⃣ AWQ 量化
            对 CosyVoice3 backbone 进行 4-bit 量化，降低显存占用，提升推理速度。
            - 脚本: `quantization/quantize_cv3_backbone_awq.py`
            - 基准测试: `quantization/benchmark_fp32_vs_awq_vllm.py`

            ### 2️⃣ Wanda 剪枝
            基于 Weights-Aware Activation 的结构化剪枝，移除不重要权重。
            - 脚本: `pruning/prune_cosyvoice_qwen_wanda_from_regenerate.py`
            - 基准测试: `pruning/benchmark_fp32_vs_pruned.py`

            ### 3️⃣ 投机解码 (Speculative Decoding)
            使用 Eagle3 draft model 加速自回归生成。
            - 训练: `speculative_decoding/train_eagle3_tts_angleslim.sh`
            - 基准测试: `speculative_decoding/run_spec_benchmark.sh`

            ### 4️⃣ AngelSlim 模型瘦身
            模型压缩与优化框架。

            ---
            **GitHub**: [JackieWang9811/cosyvoice3-inference-acceleration](https://github.com/JackieWang9811/cosyvoice3-inference-acceleration)
            """)

    return demo


# ── entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()

    # load speaker list
    try:
        from cosyvoice.cli.cosyvoice import AutoModel
        cosyvoice = AutoModel(model_dir=args.model_dir)
        sft_spk = cosyvoice.list_available_spks()
        if len(sft_spk) == 0:
            sft_spk = ['']
    except Exception:
        sft_spk = ['']

    demo = build_ui(args.model_dir, sft_spk)
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    main()

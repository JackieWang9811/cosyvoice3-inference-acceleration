# CosyVoice3 推理加速

基于 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 的推理加速方案，包含多种模型压缩与加速技术的实现与评估。

## 项目结构

```
cosyvoice3-inference-acceleration/
├── cosyvoice/                  # CosyVoice 源码（含推理加速相关修改）
├── quantization/               # AWQ 量化
│   ├── quantize_cv3_backbone_awq.py        # AWQ 量化脚本
│   ├── benchmark_fp32_vs_awq_torch.py      # FP32 vs AWQ 性能对比 (PyTorch)
│   ├── benchmark_fp32_vs_awq_vllm.py       # FP32 vs AWQ 性能对比 (vLLM)
│   ├── transfer_awq_vllm.py                # AWQ 模型转 vLLM 格式
│   ├── test_vllm_awq_6cases.py             # AWQ 模型 vLLM 测试
│   └── quantization_awq/                   # 量化后的模型权重（本地）
├── pruning/                    # Wanda 剪枝
│   ├── prune_cosyvoice_qwen_wanda_from_regenerate.py        # Wanda 剪枝
│   ├── prune_cosyvoice_qwen_wanda_from_regenerate_mimic_awq.py # 剪枝+AWQ 混合
│   ├── benchmark_fp32_vs_pruned.py         # FP32 vs 剪枝 性能对比
│   └── wanda/                              # 剪枝后的模型权重（本地）
├── speculative_decoding/       # 投机解码
│   ├── train_eagle3_tts_angleslim.sh       # Eagle3 模型训练
│   ├── run_spec_benchmark.sh               # 投机解码性能测试
│   └── count_eagle3_params.py              # Eagle3 参数量统计
├── AngelSlim/                  # 模型瘦身框架
├── customer_service_finetune/  # 客服场景微调
│   ├── configs/                # 训练配置
│   ├── scripts/                # 训练/推理脚本
│   └── data/                   # 训练数据（本地）
├── evaluation/                 # 评估脚本
│   ├── dnsmos/                 # 语音质量评估 (DNSMOS)
│   ├── speaker_similarity/     # 说话人相似度 & WER 评估
│   ├── emo_eval/               # 情感评估
│   └── scripts/                # 评估启动脚本
├── concurrency_testing/        # 并发推理测试
│   ├── bench_client.py         # 压测客户端
│   └── server_bench.py         # 服务端性能测试
├── tools/                      # 工具脚本
│   ├── analyze_cosyvoice3_model.py  # 模型结构分析
│   ├── check_audio_lengths.py       # 音频长度检查
│   ├── extract_embedding.py         # 说话人嵌入提取
│   ├── extract_speech_token.py      # 语音 Token 提取
│   └── make_parquet_list.py         # 数据列表生成
├── examples/                   # 使用示例
│   ├── example.py              # CosyVoice 推理示例
│   └── vllm_example.py         # vLLM 推理示例
└── webui.py                    # Gradio 可视化推理界面
```

## 加速方案

### 1. AWQ 量化 (Activation-aware Weight Quantization)

对 CosyVoice3 的 backbone 模型进行 AWQ 量化，降低模型精度到 4-bit，显著减少显存占用并提升推理速度。

```bash
# 量化模型
python quantization/quantize_cv3_backbone_awq.py

# 性能对比
python quantization/benchmark_fp32_vs_awq_vllm.py
```

### 2. Wanda 剪枝 (Weights-Aware Activation)

使用 Wanda 方法对模型进行结构化剪枝，移除不重要的权重，实现模型压缩。

```bash
# 执行剪枝
bash pruning/prune_regenerate.sh
```

### 3. 投机解码 (Speculative Decoding)

利用 Eagle3 作为 draft model，通过投机解码加速自回归生成过程。

```bash
# 训练 Eagle3 模型
bash speculative_decoding/train_eagle3_tts_angleslim.sh

# 运行投机解码性能测试
bash speculative_decoding/run_spec_benchmark.sh
```

### 4. 模型瘦身 (AngelSlim)

基于 AngelSlim 框架对模型进行压缩和优化。

## 快速开始

### 环境配置

参考 [CosyVoice 环境配置](https://github.com/FunAudioLLM/CosyVoice) 安装依赖。

### 基础推理

```python
# 查看 examples/example.py
from cosyvoice.cli.cosyvoice import CosyVoice
model = CosyVoice('pretrained_models/CosyVoice-3')
# ... 进行推理
```

### Web UI 演示

```bash
# 启动 Gradio 可视化界面
python webui.py --port 8000 --model_dir pretrained_models/CosyVoice2-0.5B
```

启动后浏览器打开 `http://localhost:8000` 即可体验 TTS 推理和加速对比功能。

## 评估指标

- **DNSMOS**: 语音质量客观评估
- **Speaker Similarity**: 说话人相似度
- **WER**: 字错误率
- **Latency**: 推理延迟
- **Throughput**: 并发吞吐量

## 参考

- [CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer](https://github.com/FunAudioLLM/CosyVoice)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [Wanda: Pruning by Weights and Activations](https://arxiv.org/abs/2306.11695)
- [Speculative Decoding for TTS](https://arxiv.org/abs/2306.00978)
- [AngelSlim: Model Slimming Framework](https://github.com/OpenDFM/AngelSlim)

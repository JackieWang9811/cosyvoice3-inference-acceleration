# CosyVoice3 智能客服微调方案

这个目录是一套可迁移到远程 CosyVoice 项目根目录的微调骨架，目标是把 CosyVoice3 调成更适合智能客服场景的中文 TTS 语气：稳定、清晰、礼貌、短句响应自然。

## 推荐放置位置

把整个 `customer_service_finetune` 目录放到远程项目：

```text
/home/b24zll/data/wjq_data/CosyVoice/customer_service_finetune
```

## 数据格式

先准备一个 CSV 或 JSONL，至少包含：

```text
audio_path,text
```

推荐字段：

```text
audio_path,text,speaker_id,scenario,emotion,language,split
```

示例：

```csv
audio_path,text,speaker_id,scenario,emotion,language,split
wavs/001.wav,您好，请问有什么可以帮您？,cs_001,greeting,polite,zh,train
wavs/002.wav,请您稍等，我正在为您查询订单信息。,cs_001,order_query,polite,zh,train
```

## 可选：导入开源客服小语料

我已经提供了 `MatrixStudio/TTS-SCCusSerFSC` 的导入脚本。这个数据集是中文客服场景 TTS 小语料，Hugging Face 页面标注为 `CC-BY-NC-ND-4.0`，适合研究原型，不建议直接用于商业模型。

如果已经把 parquet 放到服务器：

```bash
/home/b24zll/software/anaconda3/envs/voice/bin/python \
  customer_service_finetune/scripts/import_tts_sccusserfsc.py \
  --parquet customer_service_finetune/downloads/TTS-SCCusSerFSC-train.parquet \
  --output-root customer_service_finetune/data/customer_service/raw_tts_sccusserfsc
```

然后把导出的 manifest 接到准备脚本：

```bash
/home/b24zll/software/anaconda3/envs/voice/bin/python \
  customer_service_finetune/scripts/prepare_customer_service_data.py \
  --input customer_service_finetune/data/customer_service/raw_tts_sccusserfsc/raw_manifest.csv \
  --audio-root customer_service_finetune/data/customer_service/raw_tts_sccusserfsc \
  --output-dir customer_service_finetune/data/customer_service/processed
```

## 生成训练清单

在 CosyVoice 根目录执行：

```bash
python3 customer_service_finetune/scripts/prepare_customer_service_data.py \
  --input data/customer_service/raw_manifest.csv \
  --audio-root data/customer_service \
  --output-dir data/customer_service/processed \
  --dev-ratio 0.05
```

输出既包含便于检查的 JSONL，也包含 CosyVoice3 原生训练需要的 Kaldi 风格文件：

```text
data/customer_service/processed/train.jsonl
data/customer_service/processed/dev.jsonl
data/customer_service/processed/metadata.list
data/customer_service/processed/stats.json
data/customer_service/processed/train/wav.scp
data/customer_service/processed/train/text
data/customer_service/processed/train/utt2spk
data/customer_service/processed/train/spk2utt
data/customer_service/processed/train/instruct
data/customer_service/processed/dev/...
```

## 生成 CosyVoice3 parquet

CosyVoice3 的训练脚本读取的是 `parquet/data.list`，所以数据清单准备完以后继续执行：

```bash
bash customer_service_finetune/scripts/make_customer_service_parquet.sh \
  . \
  customer_service_finetune/data/customer_service/processed
```

这会生成：

```text
customer_service_finetune/data/customer_service/train.data.list
customer_service_finetune/data/customer_service/dev.data.list
```

## 启动微调

先 dry-run 看命令：

```bash
python3 customer_service_finetune/scripts/launch_finetune.py \
  --cosyvoice-root . \
  --config customer_service_finetune/configs/cosyvoice3_customer_service_sft.json \
  --dry-run
```

确认无误后去掉 `--dry-run`。当前配置默认只训练 CosyVoice3 的 `llm`，和仓库 `examples/libritts/cosyvoice3/run.sh` 保持一致。

## 设计思路

- 先做 CosyVoice3 官方链路里的 LLM SFT，避免直接动 flow/hift 导致声学质量不稳定。
- 数据上强调客服短句、确认句、道歉句、等待句、转人工句、订单/售后/预约等高频场景。
- 推理时在文本前加稳定的客服风格控制前缀，由 `infer_customer_service.py` 统一管理。
- 训练产物建议独立保存到 `exp/customer_service_cosyvoice3_lora`，不要覆盖原模型。

## 已对接的原项目入口

- 训练入口：`cosyvoice/bin/train.py`
- 模型配置：`pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml`
- 初始权重：`pretrained_models/Fun-CosyVoice3-0.5B/llm.pt`
- Qwen tokenizer/pretrain 路径：`pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN`
- ONNX 在线特征路径：`pretrained_models/Fun-CosyVoice3-0.5B`

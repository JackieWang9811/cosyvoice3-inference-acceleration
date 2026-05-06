from pathlib import Path
import sys, traceback, torchaudio
ROOT=Path('/home/b24zll/data/wjq_data/CosyVoice')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'third_party/Matcha-TTS'))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed
model=AutoModel(model_dir=str(ROOT/'pretrained_models/Fun-CosyVoice3-0.5B'), fp16=False)
out=ROOT/'customer_service_finetune/infer_outputs/diagnose'; out.mkdir(parents=True, exist_ok=True)
prompt=str(ROOT/'asset/zero_shot_prompt.wav')
text='??????????????????????????'
methods=[
 ('zero_true', lambda: model.inference_zero_shot(text, 'You are a helpful assistant.<|endofprompt|>???????????????', prompt, stream=False, text_frontend=True)),
 ('zero_false', lambda: model.inference_zero_shot(text, 'You are a helpful assistant.<|endofprompt|>???????????????', prompt, stream=False, text_frontend=False)),
 ('cross_false', lambda: model.inference_cross_lingual('You are a helpful customer service assistant. Speak in Mandarin clearly and warmly.<|endofprompt|>'+text, prompt, stream=False, text_frontend=False)),
 ('cross_true', lambda: model.inference_cross_lingual('You are a helpful customer service assistant. Speak in Mandarin clearly and warmly.<|endofprompt|>'+text, prompt, stream=False, text_frontend=True)),
 ('instruct2_false', lambda: model.inference_instruct2(text, 'You are a helpful customer service assistant. Speak in Mandarin clearly and warmly.<|endofprompt|>', prompt, stream=False, text_frontend=False)),
]
for idx,(name, fn) in enumerate(methods):
    print('METHOD', name, flush=True)
    set_all_random_seed(500+idx)
    try:
        n=0
        for j, r in enumerate(fn()):
            path=out/f'{name}_{j}.wav'
            torchaudio.save(str(path), r['tts_speech'].cpu(), model.sample_rate)
            print('SAVED', path, r['tts_speech'].shape, flush=True)
            n+=1
        print('COUNT', n, flush=True)
    except Exception:
        traceback.print_exc()

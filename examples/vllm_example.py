# import sys
# sys.path.append('third_party/Matcha-TTS')
# from vllm import ModelRegistry
# from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
# ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

# from cosyvoice.cli.cosyvoice import AutoModel
# from cosyvoice.utils.common import set_all_random_seed
# from tqdm import tqdm
# import torchaudio
# import os 
# def cosyvoice2_example():
#     """ CosyVoice2 vllm usage
#     """
#     cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=True, fp16=True)
#     for i in tqdm(range(100)):
#         set_all_random_seed(i)
#         for _, _ in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav', stream=False)):
#             continue


# def cosyvoice3_example():
#     """ CosyVoice3 vllm usage """
#     # 1. Create the output directory if it doesn't exist
#     output_dir = './vllm_generated_examples'
#     os.makedirs(output_dir, exist_ok=True)
    
#     cosyvoice = AutoModel(model_dir='./pretrained_models/Fun-CosyVoice3-0.5B', load_trt=True, load_vllm=True, fp16=False)
    
#     for i in tqdm(range(10)):
#         set_all_random_seed(i)
#         for _, j in enumerate(cosyvoice.inference_zero_shot(
#             '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', 
#             'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
#             './asset/zero_shot_prompt.wav', 
#             stream=False
#         )):
#             # 2. Now the directory is guaranteed to exist
#             save_path = os.path.join(output_dir, 'instruct_{}.wav'.format(i))
#             torchaudio.save(save_path, j['tts_speech'], cosyvoice.sample_rate)


# def main():
#     # cosyvoice2_example()
#     cosyvoice3_example()


# if __name__ == '__main__':
#     main()




import os
import sys
import torchaudio
from tqdm import tqdm

sys.path.append('third_party/Matcha-TTS')

from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed


def cosyvoice3_vllm_test_6cases():
    """CosyVoice3 vLLM usage: test the 6 example cases"""

    output_dir = './vllm_generated_examples_6cases'
    os.makedirs(output_dir, exist_ok=True)

    cosyvoice = AutoModel(
        model_dir='./pretrained_models/Fun-CosyVoice3-0.5B',
        load_trt=True,
        load_vllm=True,
        fp16=False
    )

    prompt_wav = './asset/zero_shot_prompt.wav'

    test_cases = [
        {
            "case_id": 1,
            "mode": "inference_zero_shot",
            "target_text": "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
            "prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
            "description": "zero_shot_tongue_twister",
        },
        {
            "case_id": 2,
            "mode": "inference_cross_lingual",
            "target_text": "You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]",
            "prompt_text": None,
            "description": "fine_grained_control_chinese",
        },
        {
            "case_id": 3,
            "mode": "inference_instruct2",
            "target_text": "好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。",
            "prompt_text": "You are a helpful assistant. 请用广东话表达。<|endofprompt|>",
            "description": "instruct_cantonese",
        },
        {
            "case_id": 4,
            "mode": "inference_instruct2",
            "target_text": "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
            "prompt_text": "You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>",
            "description": "instruct_fast_speech",
        },
        {
            "case_id": 5,
            "mode": "inference_zero_shot",
            "target_text": "高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。",
            "prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
            "description": "zero_shot_hotfix_polyphone",
        },
        {
            "case_id": 6,
            "mode": "inference_cross_lingual",
            "target_text": "You are a helpful assistant.<|endofprompt|>レキシ テキ セカイ ニ オイ テ ワ、カコ ワ タンニ スギサッ タ モノ デ ワ ナイ、プラトン ノ イウ ゴトク ヒ ユー ガ ユー デ アル。",
            "prompt_text": None,
            "description": "cross_lingual_japanese",
        },
    ]

    for case in tqdm(test_cases, desc="Running 6 CosyVoice3 vLLM test cases"):
        case_id = case["case_id"]
        mode = case["mode"]
        target_text = case["target_text"]
        prompt_text = case["prompt_text"]
        description = case["description"]

        set_all_random_seed(case_id)

        if mode == "inference_zero_shot":
            generator = cosyvoice.inference_zero_shot(
                target_text,
                prompt_text,
                prompt_wav,
                stream=False
            )

        elif mode == "inference_cross_lingual":
            generator = cosyvoice.inference_cross_lingual(
                target_text,
                prompt_wav,
                stream=False
            )

        elif mode == "inference_instruct2":
            generator = cosyvoice.inference_instruct2(
                target_text,
                prompt_text,
                prompt_wav,
                stream=False
            )

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        for idx, result in enumerate(generator):
            save_name = f"case{case_id}_{mode}_{description}_{idx}.wav"
            save_path = os.path.join(output_dir, save_name)
            torchaudio.save(save_path, result['tts_speech'], cosyvoice.sample_rate)

            print("=" * 80)
            print(f"Saved: {save_path}")
            print(f"case_id    : {case_id}")
            print(f"mode       : {mode}")
            print(f"description: {description}")
            print(f"target_text: {target_text}")


def main():
    cosyvoice3_vllm_test_6cases()


if __name__ == '__main__':
    main()
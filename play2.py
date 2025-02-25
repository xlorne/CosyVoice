import os
import sys
import time

sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice2('./pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
prompt_text = '希望你以后能够做的比我还好呦。'

# mkdir out
if not os.path.exists('./out'):
    os.mkdir('./out')

t1 = time.time()
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', prompt_text, prompt_speech_16k, stream=False)):
    torchaudio.save('./out/zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
t2 = time.time()

print('time',(t2-t1) * 1000)

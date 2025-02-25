import os
import sys
import time

sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice
import torchaudio


cosyvoice = CosyVoice('./pretrained_models/CosyVoice-300M-SFT', load_jit=False, load_trt=False, fp16=False)
# sft usage
print(cosyvoice.list_available_spks())

# mkdir out
if not os.path.exists('./out'):
    os.mkdir('./out')

t1 = time.time()
# change stream=True for chunk stream inference
for i, j in enumerate(cosyvoice.inference_sft('你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？', '中文女', stream=False)):
    torchaudio.save('./out/sft_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
t2 = time.time()

print('time',(t2-t1) * 1000)

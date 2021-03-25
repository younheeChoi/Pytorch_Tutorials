# pytorch data load 방법
# pip install torchaudio 필요

import torch
import torchaudio

# =======데이터 다운 및 잘 불러왔는지 테스트 용도========#
# download = Ture: 다운로드시 사용 / download = False: 다운로드 후 데이터 불러올때 사용
yesno_data_trainset = torchaudio.datasets.YESNO('./', download=False)

# 예시데이터 들을때 사용 tuple (waveform, sample_rate, labels) where labels 로 나옴
n = 3
waveform, sample_rate, labels = yesno_data_trainset[n]
print("Waveform: {}\nSample rate: {}\nLabels: {}".format(waveform, sample_rate, labels))

# =======데이터 로드========#


# Yes Laughing

## Env
```bash
conda create -y -n realtime python=3.7
conda activate yes_laughing
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cpuonly -c pytorch
pip install streamlit streamlit-webrtc 
pip install Pillow opencv-python-headless

pip install imutils
```

```bash
wget http://efrosgans.eecs.berkeley.edu/gangealing/pretrained_stn_only/celeba.pt -o assets/pretrained/celeba.pt
# wget http://efrosgans.eecs.berkeley.edu/gangealing/video/elon/data.mdb -o assets/elon/data.mdb
# wget http://efrosgans.eecs.berkeley.edu/gangealing/video/elon/lock.mdb -o assets/elon/lock.mdb
```

## Run
```
streamlit run app.py
```
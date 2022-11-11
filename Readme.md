# Yes Laughing

## Env
```bash
conda create -y -n yes_laughing python=3.7
conda activate yes_laughing
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cpuonly -c pytorch
pip install streamlit streamlit-webrtc 
pip install Pillow opencv-python-headless

pip install imutils
```

```bash
wget http://efrosgans.eecs.berkeley.edu/gangealing/pretrained_stn_only/celeba.pt -P assets/pretrained/
# wget http://efrosgans.eecs.berkeley.edu/gangealing/video/elon/data.mdb -P assets/elon/
# wget http://efrosgans.eecs.berkeley.edu/gangealing/video/elon/lock.mdb -p assets/elon/
```

## Run
```
streamlit run app.py
```

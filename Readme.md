# Yes Laughing
ref: https://github.com/wpeebles/gangealing

## Env
```bash
conda create -y -n yes_laughing python=3.8
conda activate yes_laughing

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
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

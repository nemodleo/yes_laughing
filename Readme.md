# Yes Laughing

ref: https://github.com/wpeebles/gangealing

![yes_laughing](assets/results/yes_laughing.gif)


## Env

```bash
donvr -p 4000 -g 0 -n yes_laughing
```

```bash
pip install streamlit streamlit-webrtc \
            Pillow opencv-python-headless \
            ninja
```

```bash
wget http://efrosgans.eecs.berkeley.edu/gangealing/pretrained_stn_only/celeba.pt -P assets/pretrained/
```


## Run

```
streamlit run app.py --server.port 4000
```

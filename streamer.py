import threading
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
from PIL import Image

import av
import cv2

from gangealing import option, GanGealing

def streamer_webcam_gg():
    # st.header("Webcam Live")

    class GGTransformer(VideoTransformerBase):

        def __init__(self, model=None) -> None:
            self._model_lock = threading.Lock()
            self.model = model
            if self.model:
                self.args = option()
                self.gangealing = GanGealing(self.args)
        
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            # np.ndarray, RGB, [480 640 3]
            img = frame.to_ndarray(format="bgr24")
            # assert img.shape == (480, 640, 3), f'[!] {img.shape} != (480, 640, 3)'
            H, W, C = img.shape
            img = img[:, ::-1, :]
            # [480 480 3]
            img = img[:, (W-H)//2:(W+H)//2,:]
            # assert img.shape == (480, 480, 3)
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            # assert img.shape == (512, 512, 3)
            if self.model:
                # [H W 3]
                img = self.gangealing.forward(img)
            else:
                # [H W]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # [H W 3]
                img = gray[:,:,None].repeat(3, axis=2)
                img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
                img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_AREA)
            a = av.VideoFrame.from_ndarray(img, format="bgr24")
            return a

    ctx = webrtc_streamer(
        client_settings=ClientSettings(
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "video": True, 
                "audio": False
            },
        ),
        video_transformer_factory=GGTransformer,
        key="realtime",
    )

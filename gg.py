import threading
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
from PIL import Image

import av
import cv2
# import imutils
# from neural_style_transfer import get_model_from_path, style_transfer
# from data import *


def webcam_gg():#style_model_name):
    # st.header("Webcam Live")
    # WIDTH = st.select_slider('QUALITY (May reduce the speed)', list(range(150, 501, 50)))

    class TestTransformer(VideoTransformerBase):
        # _width = WIDTH
        # _model_name = #style_model_name
        _model = None

        def __init__(self) -> None:
            self._model_lock = threading.Lock()

            # self._width = WIDTH
            # self._update_model()

        # def set_width(self, width):
        #     update_needed = self._width != width
        #     self._width = width
        #     if update_needed:
        #         self._update_model()

        # def update_model_name(self, model_name):
        #     update_needed = self._model_name != model_name
        #     self._model_name = model_name
        #     if update_needed:
        #         self._update_model()

        # def _update_model(self):
        #     style_model_path = style_models_dict[self._model_name]
        #     with self._model_lock:
        #         self._model = get_model_from_path(style_model_path)

        
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

            # [H W]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # [H W 3]
            img = gray[:,:,None].repeat(3, axis=2)
            img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
            # img = cv2.resize(img, (1024, 768), interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_AREA)

            # img_white = np.ones((480, 640, 3), dtype=np.uint8) * 255
            # img_white[:, (W-H)//2:(W+H)//2,:] = img
            # img = img_white

            # faces = self._face_cascade.detectMultiScale(
            #     gray, scaleFactor=1.11, minNeighbors=3, minSize=(30, 30)
            # )

            # overlay = self._filters[self.filter_type]

            # for (x, y, w, h) in faces:
            #     # Ad-hoc adjustment of the ROI for each filter type
            #     if self.filter_type == "ironman":
            #         roi = (x, y, w, h)
            #     elif self.filter_type == "laughing_man":
            #         roi = (x, y, int(w * 1.15), h)
            #     elif self.filter_type == "cat":
            #         roi = (x, y - int(h * 0.3), w, h)
            #     overlay_bgra(img, overlay, roi)

            #     if self.draw_rect:
            #         img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            a = av.VideoFrame.from_ndarray(img, format="bgr24")
            return a


        # def transform(self, frame):
        #     # np.ndarray, RGB, [480 640 3]
        #     image = frame.to_ndarray(format="bgr24")
        #     assert image.shape == (480, 640, 3)

        #     if self._model == None:
        #         image = image[..., ::-1]  ###
        #         image = image[:, ::-1, :]  ###
        #         return image

        #     orig_h, orig_w = image.shape[0:2]
        #     # cv2.resize used in a forked thread may cause memory leaks
        #     # input = np.asarray(Image.fromarray(image).resize((self._width, int(self._width * orig_h / orig_w))))

        #     with self._model_lock:
        #         # transferred = style_transfer(input, self._model)
        #         transferred = input

        #     result = Image.fromarray((transferred * 255).astype(np.uint8))

        #     # result = result[..., [2,1,0]]
        #     return np.asarray(result.resize((orig_w, orig_h)))

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
        video_transformer_factory=TestTransformer,
        key="realtime",
    )
    # if ctx.video_transformer:
    #     ctx.video_transformer.set_width(WIDTH)
    #     ctx.video_transformer.update_model_name(style_model_name)






# import cv2
# import imutils
# def get_model_from_path(style_model_path):
#     model = cv2.dnn.readNetFromTorch(style_model_path)
#     return model

# def style_transfer(image, model):
#     (h, w) = image.shape[:2]
#     # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) #PIL Jpeg to Opencv image

#     blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
#     model.setInput(blob)
#     output = model.forward()

#     output = output.reshape((3, output.shape[2], output.shape[3]))
#     output[0] += 103.939
#     output[1] += 116.779
#     output[2] += 123.680
#     output /= 255.0
#     output = output.transpose(1, 2, 0)
#     output = np.clip(output, 0.0, 1.0)
#     output = imutils.resize(output, width=500)
#     return output
#setting.py

import av
import os
import sys
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from aiortc.contrib.media import MediaRecorder
from streamlit_lottie import st_lottie
import json
import requests
import pandas as pd


BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
sys.path.append(BASE_DIR)

# from utils import get_mediapipe_pose
# from process_frame import ProcessFrame
# from thresholds import get_thresholds_beginner, get_thresholds_pro




def get(path: str):
    with open(path, "r") as p:
        return json.load(p)


# lot_sehat = get(".appfiles/sehat_lottie.json")
# -------------- SETTINGS --------------

joints = ["left_shoulder", "right_shoulder", "left_hip",
          "right_hip", "left_knee", "right_knee",
          "left_elbow", "right_elbow"]
page_title = "مقدار زوایای مورد نظر را برای هر مفصل انتخاب کنید "
page_icon = "📝"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
layout = "centered"
# --------------------------------------

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)


# lot = st_lottie(animation_data=lot_sehat,
#                 speed=.5,
#                 reverse=False,
#                 loop=True,
#                 quality="low",
#                 height=225,
#                 width=400,
#                 key="lot_t")

values = []


options = st.multiselect('مفاصل',
                         ["گردن",
                          "شانه چپ",
                          "شانه راست",
                          "آرنج چپ",
                          "آرنج راست",
                          "مچ چپ",
                          "مچ راست",
                          "ران جپ",
                          "ران راست",
                          "زانو چپ",
                          "زانو راست",
                          "مچ پای چپ",
                          "مچ پای راست"]
                         )
for option in options:
    value = st.slider(option, 0, 180, (50, 100))
    values.append(value)

with st.form("entry_form", clear_on_submit=True):
    with st.expander("راهنما"):
        st.text("""باتوجه به ورزش مورد نظر خود بازه زوایا را مشخص کنید""")
    submitted = st.form_submit_button("ذخیره داده‌ها")
    if submitted:
        data = []
        for i in range(len(options)):
            row = [options[i], values[i][0], values[i][1]]
            data.append(row)
        df = pd.DataFrame(data, columns=['مفصل', 'حداقل زاویه', 'حداکثر زاویه'])
        st.write(df)
        df.to_csv('data.csv', index=False)


connected_joints = {
    "گردن": ["شانه چپ", "شانه راست"],
    "شانه چپ": ["لگن چپ", "آرنج چپ"],
    "شانه راست": ["لگن راست", "آرنج راست"],
    "آرنج چپ": ["شانه چپ", "مچ چپ"],
    "آرنج راست": ["شانه راست", "مچ راست"],

    "لگن جپ": ["شانه چپ","زانو چپ"],
    "لگن راست": ["شانه راست","زانو راست"],
    "زانو چپ": ["لگن جپ", "مچ پای جپ"],
    "زانو راست": ["لگن راست", "مچ پای راست"]}


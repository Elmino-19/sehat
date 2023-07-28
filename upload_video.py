import os
import tempfile
import time
import pandas as pd
import cv2
import streamlit as st

import class_process

# حذف فایل جهت جلوگیری از خطاهای محاسباتی
if os.path.exists("upload.csv"):
    os.remove("upload.csv")

# اطلاعات مربوط به هر مفصل شامل شماره هر اندام
dict_angle = {
    "right_elbow": [16, 14, 12, 0, 0, 0, "ff", 180],
    "left_elbow": [15, 13, 11, 0, 0, 0, "ff", 180],
    "right_shoulder": [14, 12, 24, 0, 0, 0, "ff", 180],
    "left_shoulder": [13, 11, 23, 0, 0, 0, "ff", 180],
    "right_hip": [12, 24, 26, 0, 0, 0, "ff", 180],
    "left_hip": [11, 23, 25, 0, 0, 0, "ff", 180],
    "right_knee": [24, 26, 28, 0, 0, 0, "ff", 180],
    "left_knee": [23, 25, 27, 0, 0, 0, "ff", 180],
    "right_ankle": [26, 28, 30, 0, 0, 0, "ff", 180],
    "left_ankle": [25, 27, 29, 0, 0, 0, "ff", 180],
}

# تبدیل دیکشنری زاویه‌ها به لیست
angle_list = list(dict_angle.keys())

# ایجاد دیتافریم جدید برای ذخیره زوایای مفاصل
df_angle = pd.DataFrame(columns=angle_list)

def main():
    with st.container():
        #تعیین انتخاب عملیات مورد نظر توسط کاربر 
        radio = st.radio(
            "توقف/اجرا",
            ("اجرا", "ذخیره و نمایش زوایای مفاصل", "توقف"),
            help="پس از بارگذاری با زدن دکمه اجرا، آنالیز شروع می‌شود"
        )

        # دریافت ویدیو از کاربر
        video_file_buffer = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        # پردازش و نشان دادن ویدیو
        if video_file_buffer:
            temp_file.write(video_file_buffer.read())
            
        if radio == "اجرا":
            frame_win = st.image([])
            cap = cv2.VideoCapture(temp_file.name)
            with st.spinner('در حال محاسبه'):
                time.sleep(5)

            past_time = 0
            detector = class_process.PoseDetector()
            # انجام پردازش بر روی هر فریم
            try:
                while cap.isOpened():
                    success, img = cap.read()
                    # محاسبه محل قرار گیری اندام‌های بدن
                    img = detector.find_pose(img)
                    landmark_list = detector.find_position(img, draw=False)
                    if not success:
                        break

                    if len(landmark_list) != 0:
                        angles = []
                        # محاسبه زاویه هر مفصل
                        for joint in dict_angle:
                            angle = detector.find_angle(img, dict_angle[joint][0], dict_angle[joint][1], dict_angle[joint][2])
                            angles.append(angle)
                        # وارد شدن زوایا هر مفصل در دریتافریم
                        df_angle.loc[len(df_angle)] = angles
                        df_angle.to_csv("upload.csv", index=False)
                        if 'df_angle' not in st.session_state:
                            st.session_state.df_angle = df_angle
                    # شمارش تعداد فریم دریافت شده
                    current_time = time.time()
                    fps = 1 / (current_time - past_time)
                    past_time = current_time
                    frame_win.image(img)
                cap.release()
            except:
                pass
        if radio == "توقف":
            if 'df_angle' not in st.session_state:
                st.error("لطفاً ابتدا ویدیو را بارگذاری کنید")
            else:
                with st.spinner('در حال وارد کردن اطلاعات'):
                    time.sleep(5)
                st.info("محاسبات مورد نیاز انجام شده است")

        if radio == "ذخیره و نمایش زوایای مفاصل":
            if 'df_angle' not in st.session_state:
                st.error("لطفاً ابتدا ویدیو را بارگذاری کنید")
            else:
                with st.spinner('در حال وارد کردن اطلاعات'):
                    time.sleep(5)
                st.dataframe(st.session_state.df_angle)

if __name__ == "__main__":
    main()

import os
import copy
import time
import pandas as pd
import cv2
import streamlit as st
import class_process

# اطلاعات مربوط به هر مفصل شامل شماره هر اندام
dict_angle = {
    "right_elbow": [16, 14, 12, 0, 0, 0, "ff", 180, 0],
    "left_elbow": [15, 13, 11, 0, 0, 0, "ff", 180, 1],
    "right_shoulder": [14, 12, 24, 0, 0, 0, "ff", 180, 2],
    "left_shoulder": [13, 11, 23, 0, 0, 0, "ff", 180, 3],
    "right_hip": [12, 24, 26, 0, 0, 0, "ff", 180, 4],
    "left_hip": [11, 23, 25, 0, 0, 0, "ff", 180, 5],
    "right_knee": [24, 26, 28, 0, 0, 0, "ff", 180, 6],
    "left_knee": [23, 25, 27, 0, 0, 0, "ff", 180, 7],
    "right_ankle": [26, 28, 30, 0, 0, 0, "ff", 180, 8],
    "left_ankle": [25, 27, 29, 0, 0, 0, "ff", 180, 9],
}

lst = []
dict_counter = copy.deepcopy(dict_angle)

if not os.path.exists("upload.csv"):
    st.error("ابتدا ویدیوی مورد نظر خود را تحلیل کنید")
else:
    df = pd.read_csv("upload.csv")

    # محاسبه مقدار بیشینه و کمینه برای هر مفصل
    for k in dict_counter.keys():
        mn = df[k].min()
        mx = df[k].max()
        if mx - mn > 30:
            dict_counter[k][3] = mx
            dict_counter[k][4] = mn
            dict_counter[k][5] = 0
            dict_counter[k][6] = "start"
        else:
            lst.append(k)
    for k in lst:
        del (dict_counter[k])

def main():
    with st.container():
        #تعیین انتخاب عملیات مورد نظر توسط کاربر 
        radio = st.radio(
            "توقف/اجرا",
            ("اجرا", "ذخیره و نمایش زوایای مفاصل", "توقف"),
            help="پس از بارگذاری با زدن دکمه اجرا آنالیز شروع می‌شود"
        )
        st.warning("لطفا به صورت فردی جلوی دوربین به طوری که کل بدنتان در تصویر باشد قرار گیرید")
        if radio == "اجرا":
            with st.spinner("درحال دریافت تصویر"):
                time.sleep(10)
            frame_win = st.image([])
            cap = cv2.VideoCapture(0)

            past_time = 0
            detector = class_process.PoseDetector()
            while cap.isOpened():
                success, img = cap.read()
                # محاسبه محل قرار گیری اندام‌های بدن
                img = detector.find_pose(img)
                landmark_list = detector.find_position(img, draw=False)
                if not success:
                    break
                if len(landmark_list) != 0:
                    # محاسبه زاویه‌های مفاصل و بررسی شرایط مربوط به آن‌ها
                    for k in dict_counter:
                        # محاسبه زوایای مفاصل
                        angle = detector.find_angle(img, dict_counter[k][0], dict_counter[k][1], dict_counter[k][2])
                        # شمارش تعداد صحیح حرکات و موقعیت آنها
                        dict_counter[k][7] = angle
                        if dict_counter[k][7] > dict_counter[k][4]:
                            dict_counter[k][6] = "start"
                        if dict_counter[k][7] < dict_counter[k][4] and dict_counter[k][6] == 'ff':
                            dict_counter[k][6] = "return"
                            dict_counter[k][5] += 1
                
                #نمایش تعداد و موقعیت حرکات هر مفصل
                img = detector.put_text(img, dict_counter)
                current_time = time.time()
                fps = 1 / (current_time - past_time)
                past_time = current_time
                frame_win.image(img)
            cap.release()

    if radio == "توقف":
        if not os.path.exists("upload.csv"):
            st.error("ابتدا ویدیوی مورد نظر خود را تحلیل کنید")
        else:
            with st.spinner("در حال محاسبه اطلاعات"):
                time.sleep(3)
            st.info("محاسبات مورد نیاز انجام شده است")
        
    if radio == "نمایش تحلیل حرکت مفاصل":
        if not os.path.exists("upload.csv"):
            st.error("ابتدا ویدیوی مورد نظر خود را تحلیل کنید")
        else:
            with st.spinner("در حال محاسبه اطلاعات"):
                time.sleep(5)
    
            df = pd.read_csv("upload.csv")
            # محاسبه آمار و مشخصات مربوط به زوایا و نمایش آن‌ها در دیتافریم
            statistics = df.describe()
            first_value = df.iloc[0]
            last_value = df.iloc[-1]
            new_row = pd.concat([first_value, last_value], axis=1).T
            statistics = pd.concat([statistics, new_row])
            st.dataframe(statistics)

if __name__ == "__main__":
    main()

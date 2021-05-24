import streamlit as st
import joblib
from skimage import io
from sklearn.cluster import KMeans
import numpy as np
import warnings
import os
from PIL import Image
import time
import pandas as pd
from plt_show import data_visualization
from collections import defaultdict
from keras import backend as K
from PIL import Image


DATE_COLUMN = 'date/time'
DATA_URL = 'https://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz'


@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data


def load_prediction_models(model_file):
    loaded_models = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_models


def main():
    image1 = Image.open('C:/Users/PZEZ/Desktop/1.jpeg')
    """创客松大赛Royal never give up  队主赛题作品"""
    st.title("欢迎来到**AI战疫前线**，我们等待你的加入")

    sideway = ["AI战'疫'前线", "诊断空间",  "联系我们"]
    choice = st.sidebar.selectbox("选择你想穿梭到达的时空页面", sideway)

    if choice == "AI战'疫'前线":
        st.info("AI识别X-ray胸片，加速肺炎患者诊断")
        st.image(image1, caption=' ', use_column_width=True)
        no1 = st.checkbox('前言')
        no2 = st.checkbox('该AI产品的使用方法')
        if no1:
            st.markdown("    ##### 2020庚子鼠年开年，人们没有料到的‘黑天鹅’--新型冠状病毒降临。我们无法奋战在医疗前线，"
                        " 却不甘心无所事事等待春天，这个冬天需要我们主动去逾越。")
            st.markdown("    ##### 疫情在当下，AI却可以收集过去的数据，预测未来之走向。基于当下疫情重要诊断方法之一----"
                        "** CT/X-ray肺片检测阳性**，我们选择了收集过去的数据这一稳妥的方式，以**减轻放射科医生庞大的阅片量**为主要目的，"
                        "进行了AI产品的开发")
        elif no2:
            st.markdown("##### 1.若要进行‘X光胸片诊断’，则点击左上角，展开侧边栏，进入‘诊断空间’，并按照提示操作")
            st.markdown("#####2.其余产品正在开发中，敬请期待")

    if choice == '诊断空间':
        st.info('如果人类注定要与瘟疫共存，那么，我们至少让那些席卷城市的波浪不再血红:heartbeat:')
        text_one = st.text_area("请再此输入X光胸片路径", "")
        path1 = str(text_one)
        K.clear_session()
        if st.button('点击此处开始检测 '):
            st.markdown("    正在为您检测胸片X光图片，请稍等约五分钟。")
            st.markdown(" 接下来将会为您呈现以下内容：")
            st.markdown(" 1.根据AI检测，该X光胸片是否被诊断为**肺炎**。")
            st.markdown(" 2.该AI模型经过训练后进行检测的**损失率**与**准确率**。")
            data_visualization(path1)

    if choice == "联系我们":
        st.markdown()

if __name__ == '__main__':
    # try:
    main()
    # except:
    #     st.markdown(" ##### ERROR: "
    #                 "\nThe picture cannot be opened, here are some tips for you "
    #                 "\nTips:1. Do **not** press *'ENTER'* after typing the path"
    #                 "     2. Do check the path of the picture.")

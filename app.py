# app.py

import streamlit as st
import numpy as np
from PIL import Image
import os
from run_prediction import run_ai4stem

st.set_page_config(page_title="AI4STEM Web App", layout="centered")
st.title("🔬 AI4STEM - STEM图像结构预测")
st.markdown("上传 `.npy` 格式的 STEM 图像，自动识别结构类型，并返回不确定度图谱。")

uploaded_file = st.file_uploader("📂 上传STEM图像（格式：.npy）", type=["npy"])

if uploaded_file is not None:
    try:
        npy_data = np.load(uploaded_file)
        npy_data = np.squeeze(npy_data)

        st.image(npy_data, caption="上传图像（灰度显示）", use_column_width=True, clamp=True)

        if st.button("🔍 运行预测"):
            with st.spinner("正在调用AI4STEM模型进行预测..."):
                assignments, uncertainty, output_path = run_ai4stem(npy_data, image_name="streamlit_result")

            st.success("🎉 预测完成！结果如下：")
            st.image(output_path, caption="预测结果图", use_column_width=True)

            # 可选下载按钮
            with open(output_path, "rb") as f:
                st.download_button("📥 下载结果图像", f, file_name="ai4stem_result.png", mime="image/png")

    except Exception as e:
        st.error(f"❌ 解析图像失败：{e}")

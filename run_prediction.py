# run_ai4stem_prediction.py

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging

from ai4stem.utils.utils_data import load_pretrained_model, load_class_dicts
from ai4stem.utils.utils_prediction import predict

def run_ai4stem(image_array, pixel_to_angstrom=0.1245, window_size_angstrom=12.0,
                stride_size=[36, 36], results_folder='./results', image_name="uploaded_image"):

    # 保证图像为2D灰度图
    input_image = np.squeeze(image_array)
    assert input_image.ndim == 2, "图像应为二维灰度图像"

    # 窗口大小：从 Å 转为像素
    window_size = int(round(window_size_angstrom / pixel_to_angstrom))

    # 输出路径准备
    os.makedirs(results_folder, exist_ok=True)

    # FFT 描述符参数
    descriptor_params = {'sigma': None, 'thresholding': True}

    # 加载模型和类别
    model = load_pretrained_model()
    n_iter = 100
    sliced_images, fft_descriptors, prediction, uncertainty = predict(
        input_image, model, n_iter, stride_size, window_size, descriptor_params
    )

    assignments = prediction.argmax(axis=-1)
    numerical_to_text_labels, text_to_numerical_labels = load_class_dicts()

    # 可视化
    fig, axs = plt.subplots(1, 3, figsize=(10, 10))

    im1 = axs[0].imshow(input_image, cmap='gray')
    axs[0].set_title('Input image')
    fig.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.05)

    im2 = axs[1].imshow(assignments, cmap='tab10')
    axs[1].set_title('Assigned label')

    all_colors = plt.cm.tab10.colors
    unique_assignments = np.unique(assignments.flatten())
    my_colors = [all_colors[idx] for idx in unique_assignments]
    patches = [mpatches.Patch(facecolor=c, edgecolor=c) for c in my_colors]
    axs[1].legend(patches, sorted(text_to_numerical_labels.keys()), handlelength=0.8, loc='lower right')

    im3 = axs[2].imshow(uncertainty, cmap='hot', vmin=0.0)
    axs[2].set_title('Bayesian uncertainty\n(mutual information)')
    fig.colorbar(im3, ax=axs[2], orientation='vertical', fraction=0.05)

    for ax in axs:
        ax.axis('off')
    fig.tight_layout()

    # 保存图像
    png_path = os.path.join(results_folder, f'{image_name}_summary.png')
    plt.savefig(png_path)
    plt.close()

    return assignments, uncertainty, png_path

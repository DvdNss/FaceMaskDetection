# coding:utf-8
"""
Filename: app.py
Author: @DvdNss

Created on 12/10/2021
"""
import csv
import os.path
import time

import cv2
import gdown
import numpy as np
import streamlit as st
import torch


def load_classes(csv_reader):
    """
    Load classes from csv.

    :param csv_reader: csv
    :return:
    """
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise (ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def draw_caption(image, box, caption):
    """
    Draw caption and bbox on image.

    :param image: image
    :param box: bounding box
    :param caption: caption
    :return:
    """

    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


@st.cache
def load_labels():
    """
    Loads labels.

    :return:
    """

    with open("dataset/labels.csv", 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    return labels


def download_models(ids):
    """
    Download all models.

    :param ids: name and links of models
    :return:
    """

    # Download model from drive if not stored locally
    with st.spinner('Downloading models, this may take a minute...'):
        for key in ids:
            if not os.path.isfile(f"model/{key}.pt"):
                url = f"https://drive.google.com/uc?id={ids[key]}"
                gdown.download(url=url, output=f"model/{key}.pt")


@st.cache(suppress_st_warning=True)
def load_model(model_path, prefix: str = 'model/'):
    """
    Load model.

    :param model_path: path to inference model
    :param prefix: model prefix if needed
    :return:
    """

    # Load model
    if torch.cuda.is_available():
        model = torch.load(f"{prefix}{model_path}.pt").to('cuda')
    else:
        model = torch.load(f"{prefix}{model_path}.pt", map_location=torch.device('cpu'))
    model.training = False
    model.eval()

    return model


def process_img(model, image, labels, caption: bool = True):
    """
    Process img given a model.

    :param caption: whether to use captions or not
    :param image: image to process
    :param model: inference model
    :param labels: given labels
    :return:
    """

    image_orig = image.copy()
    rows, cols, cns = image.shape

    smallest_side = min(rows, cols)

    # Rescale the image
    min_side = 608
    max_side = 1024
    scale = min_side / smallest_side

    # Check if the largest side is now greater than max_side
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # Resize the image with the computed scale
    image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
    rows, cols, cns = image.shape

    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    image = new_image.astype(np.float32)
    image /= 255
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))

    with torch.no_grad():

        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda()

        st = time.time()
        scores, classification, transformed_anchors = model(image.float())
        elapsed_time = time.time() - st
        idxs = np.where(scores.cpu() > 0.5)

        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]

            x1 = int(bbox[0] / scale)
            y1 = int(bbox[1] / scale)
            x2 = int(bbox[2] / scale)
            y2 = int(bbox[3] / scale)
            label_name = labels[int(classification[idxs[0][j]])]
            colors = {
                'with_mask': (0, 255, 0),
                'without_mask': (255, 0, 0),
                'mask_weared_incorrect': (190, 100, 20)
            }
            cap = '{}'.format(label_name) if caption else ''
            draw_caption(image_orig, (x1, y1, x2, y2), cap)
            cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=colors[label_name], thickness=2)
            cv2.putText(image_orig,
                        f"{'{:.1f}'.format(1 / float(elapsed_time))}{'  cuda:' + str(torch.cuda.is_available()).lower()}",
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN, org=(10, 20), color=(0, 255, 0))
    return image_orig


# Page config
st.set_page_config(layout="centered")
st.title("Face Mask Detection")
run = st.checkbox('Webcam mode (not working on Streamlit Cloud)')
labels = load_labels()

# Models drive ids
ids = {
    'resnet50_20': '17c2kseAC3y62IwaRQW4m1Vc-7o3WjPdh',
    'resnet50_29': '1E_IOIuE5OpO4tQgTbXjdAmXR-9BCxxmT',
    'resnet152_20': '1oUHqE_BgXehopdicuvPCGOxnwAdlDkEY',
}

download_models(ids)

# Model selection
model_path = st.selectbox('Model selection', ('resnet50_20', 'resnet50_29', 'resnet152_20'), index=1)
model = load_model(model_path=model_path)

if run:
    camera = cv2.VideoCapture(0)
    video = st.image([])

    # Process camera imgs
    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.image(process_img(model, frame, labels, caption=True))

else:
    index = st.number_input('', min_value=0, max_value=852, value=373)
    image = cv2.imread(f'dataset/validation/image/maksssksksss{str(index)}.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    left, right = st.columns([3, 1])
    left.image(process_img(model, image, labels, caption=False))
    right.write({
        'green': 'with_mask',
        'orange': 'mask_weared_incorrect',
        'red': 'without_mask'
    })
    device = 'CPU' if not torch.cuda.is_available() else 'GPU'
    right.write(f"CUDA: {torch.cuda.is_available()} ({device})")

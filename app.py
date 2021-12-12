# coding:utf-8
"""
Filename: app.py
Author: @DvdNss

Created on 12/10/2021
"""
import csv
import time

import cv2
import numpy as np
import streamlit as st
import torch


def load_classes(csv_reader):
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


@st.cache
def load_model(model_path):
    """
    Load model.

    :param model_path: path to inference model
    :return:
    """

    # Load model
    if torch.cuda.is_available():
        model = torch.load(f"model/{model_path}.pt")
        model.cuda()
    else:
        model = torch.load(f"model/{model_path}.pt", map_location=torch.device('cpu'))
    model.training = False
    model.eval()

    return model


def process_img(model, image, labels):
    """
    Process img given a model.

    :param image: image to process
    :param model: inference model
    :param labels: given labels
    :return:
    """

    image_orig = image.copy()
    rows, cols, cns = image.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    min_side = 608
    max_side = 1024
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
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
        scores, classification, transformed_anchors = model(image.cuda().float())
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
            caption = '{}'.format(label_name)
            draw_caption(image_orig, (x1, y1, x2, y2), caption)
            cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=colors[label_name], thickness=2)
            cv2.putText(image_orig,
                        f"{'{:.1f}'.format(1 / float(elapsed_time))}{'  cuda:' + str(torch.cuda.is_available()).lower()}",
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN, org=(10, 20), color=(0, 255, 0))
    return image_orig


# Page config
st.title("Face Mask Detection")
run = st.checkbox('Start the camera')
labels = load_labels()

# Model selection
model_path = st.selectbox('', ('resnet50_20', 'resnet50_29', 'resnet152_20'), index=1,
                          help='Select a model for inference. ')
model = load_model(model_path=model_path)

# Load camera
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

# Process camera imgs
while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(process_img(model, frame, labels))

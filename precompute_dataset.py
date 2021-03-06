# coding:utf-8
"""
Filename: process_dataset.py
Author: @DvdNss

Created on 12/15/2021
"""

import csv
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm


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
        model = model.module.cpu()
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
            cv2.putText(image_orig, f"{'{:.1f}'.format(1 / float(elapsed_time))}"
                                    f"{'  cuda:' + str(torch.cuda.is_available()).lower()}", fontScale=1,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, org=(10, 20), color=(0, 255, 0))
    return image_orig


def main():
    """
    Precompute images and save them.

    :return:
    """
    # Models drive ids
    ids = {
        'resnet50_20': '17c2kseAC3y62IwaRQW4m1Vc-7o3WjPdh',
        'resnet152_20': '1oUHqE_BgXehopdicuvPCGOxnwAdlDkEY',
    }

    # Model selection
    labels = load_labels()
    for model_name in ids:
        model = load_model(model_path=model_name)
        for n in tqdm(range(0, 852)):
            image = cv2.imread(f"dataset/validation/image/maksssksksss{str(n)}.jpg")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = process_img(model, image, labels, caption=False)
            cv2.imwrite(f"dataset/validation/{model_name.split('_')[0]}/maksssksksss{str(n)}.jpg", image)


if __name__ == '__main__':
    main()

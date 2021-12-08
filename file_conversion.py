# coding:utf-8
"""
Filename: file_conversion.py
Author: @DvdNss

Created on 12/8/2021
"""

import argparse
import os

from PIL import Image
from tqdm import tqdm


def png_to_jpg(args=None):
    parser = argparse.ArgumentParser(description='Convert .png files to .jpg files. ')
    parser.add_argument('--img_dir', help='Directory containing images. ')

    parser = parser.parse_args(args)

    for filename in tqdm(os.listdir(parser.img_dir)):
        if ".png" in filename:
            im1 = Image.open(parser.img_dir + filename)
            im1 = im1.convert('RGB')
            im1.save(f"{parser.img_dir + filename[:-3]}jpg")


if __name__ == '__main__':
    png_to_jpg()

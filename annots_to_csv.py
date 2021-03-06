# coding:utf-8
"""
Filename: annots_to_csv.py
Author: @DvdNss

Created on 12/7/2021
"""

import argparse
import glob
import os
import xml.etree.ElementTree as ET

import pandas as pd


def xml_to_csv_and_ground_truth(path: str, output_path: str, path_to_dataset: str = ""):
    """
    Returns a csv format from a given xml format dataset path.

    """

    # Annotations and labels lists
    xml_list = []
    labels = []

    for xml_file in glob.glob(path + 'annotation/*.xml'):
        # Parse the directory to find all annotations
        tree = ET.parse(xml_file)

        # Get file path
        root = tree.getroot()

        # Parse xml document to find the bbox and labels
        for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = int(float(bbx.find('xmin').text))
            ymin = int(float(bbx.find('ymin').text))
            xmax = int(float(bbx.find('xmax').text))
            ymax = int(float(bbx.find('ymax').text))
            label = member.find('name').text

            # If label not in labels then add it
            if label not in labels:
                labels.append(label)

            img_path = os.path.abspath(path + "/image/" + root.find('filename').text.replace(".png", ".jpg"))
            if path_to_dataset != "":
                img_path = path_to_dataset + img_path.split('\\dataset')[1]
                img_path = img_path.replace("\\", "/")

            value = (img_path,
                     xmin,
                     ymin,
                     xmax,
                     ymax,
                     label
                     )
            # Append values to the xml list
            xml_list.append(value)

    # Create dataframe from generated lists
    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
    xml_df = pd.DataFrame(xml_list, columns=column_name)

    # Create label file
    f = open(output_path + "labels.csv", "w+")

    # Write labels
    for i, label in enumerate(labels):
        f.write(label + "," + str(i) + "\n")
    f.close()

    return xml_df


def generate_csv_and_ground_truth(args=None):
    """
    Generates csv files and directories for all datasets mentioned in the config file.

    """

    # Create parser and its args
    parser = argparse.ArgumentParser(description='Generate csv and ground truth files given datasets. ')
    parser.add_argument('--train_dataset', help='Path to training dataset. ', default='dataset/train/')
    parser.add_argument('--valid_dataset', help='Path to validation dataset. ', default='dataset/validation/')
    parser.add_argument('--output_path', help='Output path for generated csv files. ', default='dataset/')
    parser.add_argument('--path_to_dataset', help='Replace image path by given path. ', default='')

    # Call parser args
    parser = parser.parse_args(args)

    for set in [parser.train_dataset, parser.valid_dataset]:
        # Creates a dataframe from all xml files of a dataset
        xmlDf = xml_to_csv_and_ground_truth(set, output_path=parser.output_path,
                                            path_to_dataset=parser.path_to_dataset)

        # Name csv file
        csvName = f"{parser.output_path}{'train' if 'train' in set else 'validation'}.csv"

        # Convert dataframe to csv and save it
        xmlDf.to_csv(f"{csvName}", index=False, header=False)
        print(csvName, f"file has been added to {parser.output_path} directory. ")


if __name__ == '__main__':
    generate_csv_and_ground_truth()

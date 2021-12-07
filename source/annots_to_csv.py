import configparser
import glob
import os
import xml.etree.ElementTree as ET
from typing import List

import pandas as pd


def xml_to_csv_and_ground_truth(path, output_path):
    """Returns a csv format from a given xml format dataset path. """

    xml_list = []  # List of all annotations
    labels = []  # List of all labels

    for xml_file in glob.glob(path + 'annotation/*.xml'):
        tree = ET.parse(xml_file)  # Parsing the directory to find all annotations
        root = tree.getroot()  # Getting the path of the file
        lines = []

        for member in root.findall('object'):  # Parsing the xml document to find the bbox and labels
            bbx = member.find('bndbox')
            xmin = int(float(bbx.find('xmin').text))
            ymin = int(float(bbx.find('ymin').text))
            xmax = int(float(bbx.find('xmax').text))
            ymax = int(float(bbx.find('ymax').text))
            label = member.find('name').text

            if label not in labels:  # If label not in labels then adds it
                labels.append(label)

            value = (os.path.abspath(path + "/image/" + root.find('filename').text),
                     xmin,
                     ymin,
                     xmax,
                     ymax,
                     label
                     )
            xml_list.append(value)  # Appends values to the xml list
            lines.append(label + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax))

        gtFile = open(xml_file.replace('annotation', 'ground_truth').replace('.xml', '.txt'), 'w+')
        for i, line in enumerate(lines):
            if i != len(lines) - 1:
                gtFile.write(line + '\n')
            else:
                gtFile.write(line)
        gtFile.close()

    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'label']  # Column names of the dataframe
    xml_df = pd.DataFrame(xml_list, columns=column_name)  # Creates a dataframe from colummns

    f = open(output_path + "labels.csv", "w+")  # Opens the labels file
    for i, label in enumerate(labels):  # For each label
        f.write(label + "," + str(i) + "\n")  # Print the labels in the labels file
    f.close()

    return xml_df  # Returns the dataframe of the given directory path


def generate_csv_and_ground_truth(paths: List[str], output_path: str = 'dataset/'):
    """Generates csv files and directories for all datasets mentioned in the config file. """

    datasets = paths

    for ds in datasets:
        if not os.path.exists(ds + 'ground_truth'):  # If ground_truth folder doesn't exist
            os.makedirs(ds + 'ground_truth')  # Creates ground_truth folder
            print(ds + 'ground_truth has been created. ')
        if not os.path.exists(ds + 'result'):
            os.makedirs(ds + 'result')  # Creates ground_truth folder
            print(ds + 'result has been created. ')
        if not os.path.exists(ds + 'image_with_predictions'):
            os.makedirs(ds + 'image_with_predictions')  # Creates ground_truth folder
            print(ds + 'image_with_predictions has been created. ')
        xmlDf = xml_to_csv_and_ground_truth(ds,
                                            output_path=output_path)  # Creates a dataframe from all xml files of a dataset
        csvName = output_path + ds.replace("dataset/", "").replace("/", "") + ".csv"  # Names the csv
        xmlDf.to_csv(f"{csvName}", index=False, header=False)  # Adds the csv file to the output path
        print(csvName, "file has been added to", output_path,
              "directory and ground_truth has been filled. ")  # Prints the result of the operation


if __name__ == '__main__':
    generate_csv_and_ground_truth(paths=['dataset/train/', 'dataset/validation/', 'dataset/train_bis/'],
                                  output_path='dataset/')

<!-- PROJECT LOGO -->
<br />
<p align="center">
<h3 align="center">Face Mask Detection using Retinanet</h3>
<p align="center">
  <img src="https://raw.githubusercontent.com/DvdNss/FaceMaskDetection/main/resources/ex.jpg?token=AL7HZ2RLFPIQ4FFSAEUGMADBWQDVK" />
</p>

<!-- ABOUT THE PROJECT -->

## About The Project

This project aims to create a Face Mask Detection model to visually detect facemasks on images and videos. We operate
with 3 labels:
* _with_mask_
* _without_mask_
* _mask_weared_incorrect_

The dataset contains approximately 2500 hand-collected and hand-labelled images.

[__Test it here!__]()

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <ul>
      <li><a href="#structure">Structure</a></li>
      <li><a href="#example">Example</a></li>
    </ul>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- GETTING STARTED -->

## Getting Started

### Installation

1. Clone the repo

```shell
git clone https://github.com/DvdNss/FaceMaskDetection
```

2. Install requirements

```shell
pip install -r requirements.txt
```

3. Clone PyTorch-Retinanet
```shell
git clone https://github.com/yhenon/pytorch-retinanet.git
```

<!-- USAGE EXAMPLES -->

## Usage

### Structure

* `dataset/`: contains datasets files
* `pytorch-retinanet/`: contains scripts for retinanet training/evaluation
* `annots_to_csv.py`: script for datasets conversion to csv
* `device.py`: script for device detection (gpu or cpu)
* `file_conversion.py`: script for png conversion to jpg

### Example

1. Convert datasets to csv file using `annots_to_csv.py`
```shell
python annots_to_csv.py --train_dataset path_to_train_dataset --valid_dataset path_to_valid_dataset --output_path path_of_outputs
```

2. Train a given model using `pytorch-retinanet/train.py`
```shell
cd pytorch-retinanet
python train.py --dataset csv --csv_train path_to_train_csv  --csv_classes path_to_class_csv  --csv_val path_to_valid_csv --depth depth_of_resnset --epochs number_of_epochs
```

3. Evaluate a given model using `pytorch-retinanet/csv_evaluation.py`
```shell
cd pytorch-retinanet
python csv_validation.py --csv_annotations_path path_to_val_annots --model_path model_path --images_path path_to_val_img --class_list_path path_to_labels
```

4. Visualize result using `pytorch-retinanet/visualize_single_image.py`
```shell
cd pytorch-retinanet
python visualize_single_image.py --image_dir image_dir_path --model_path model_path --class_list labels_path
```

<!-- CONTACT -->

## Contact

David NAISSE - [@LinkedIn](https://www.linkedin.com/in/davidnaisse/) - private.david.naisse@gmail.com

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/sunwaee/PROJECT_NAME.svg?style=for-the-badge

[contributors-url]: https://github.com/Sunwaee/PROJECT_NAME/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/sunwaee/PROJECT_NAME.svg?style=for-the-badge

[forks-url]: https://github.com/Sunwaee/PROJECT_NAME/network/members

[stars-shield]: https://img.shields.io/github/stars/sunwaee/PROJECT_NAME.svg?style=for-the-badge

[stars-url]: https://github.com/Sunwaee/PROJECT_NAME/stargazers

[issues-shield]: https://img.shields.io/github/issues/sunwaee/PROJECT_NAME.svg?style=for-the-badge

[issues-url]: https://github.com/Sunwaee/PROJECT_NAME/issues

[license-shield]: https://img.shields.io/github/license/sunwaee/PROJECT_NAME.svg?style=for-the-badge

[license-url]: https://github.com/Sunwaee/PROJECT_NAME/blob/master/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[linkedin-url]: https://www.linkedin.com/in/davidnaisse/
<!-- PROJECT LOGO -->
<br />
<p align="center">
<h3 align="center">Face Mask Detection using Retinanet</h3>
<p align="center">
  <img src="" />
</p>

<!-- ABOUT THE PROJECT -->

## About The Project

This project aims to create a Face Mask Detection model to visually detect facemasks on images and videos. We operate
with 3 labels:
* _with_mask_
* _without_mask_
* _mask_weared_incorrect_

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

* `example/`: contains inference outputs
* `model/`: contains model .pt file
* `resources/`: contains repo home image
* `source/`:
    * `mc_autoencoder.py`: model structure (structure, forward pass...)
    * `model.py`: model methods (train, eval, save...)
    * `train.py`: training script
    * `inference.py`: eval and inference script
    * `app.py`: project GUI
* `utils/`:
    * `device.py`: fast script for device availability (cpu or gpu -- just run `device.py`)

### Example

1. Run the `train.py` script. Feel free to edit parameters like `channel sizes`, `epochs` or `learning rate`.

```python
import torch.nn
from torchvision.transforms import ToTensor

from source.model import Model

# Load data & dataloader
train_data, test_data, train_dataloader, test_dataloader = Model.load_mnist(transform=ToTensor(), batch_size=1)

# Load model
model = Model(device='cuda', img_chan_size=100, global_chan_size=50)
print(model.model)

# Loss & Optim
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-3)

epoch = 15
mask_prob = 25

# Training model
for _ in range(0, epoch):
    model.train(dataloader=train_dataloader, loss=loss, optimizer=optimizer, mask_prob=mask_prob, log_iter=60000)
    model.eval(dataloader=test_dataloader, loss=loss)
    model.eval(dataloader=test_dataloader, loss=loss, mask=True)
    mask_prob += 25 if mask_prob < 100 else 0
    print(f'Mask probability is now {mask_prob}%. ')

# Save model
model.save('model/model.pt')
```

2. Run the `inference.py` script (examples will be stored in example/ as .png)

```python
from torchvision.transforms import ToTensor

from source.model import Model

# Load data & dataloader
train_data, test_data, train_dataloader, test_dataloader = Model.load_mnist(transform=ToTensor(), batch_size=1)

# Load model
model = Model(load_model='model/model.pt', img_chan_size=100, global_chan_size=50)
print(f'Trainable parameters: {sum(p.numel() for p in model.model.parameters())}. ')

# Quick inference
model.infer(eval_data=test_data, random=True)
```

3. Examples of input/output/label

input : ![](example/target1.png) \
output : ![](example/output1.png) \
label : 8

4. Use the model with GUI

```shell
cd source/
streamlit run app.py
```

<p align="center">
  <img src="https://raw.githubusercontent.com/DvdNss/mnist_encoder/main/resources/img.PNG" />
</p>

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
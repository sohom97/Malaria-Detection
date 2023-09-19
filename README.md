### Detecting Malaria cells using Convolutional Neural Network

### Overview

The data is collected from an authorized portal and the main objective is to design and develop a neural network model for future malaria prediction. The dataset contains 27558 images categorized into two folders. The parasitized folder contains images of malaria infected cells and uninfected folder contains images of clear blood cells.

### How to run the code

For training of the model you need to run the train.py by passing the path for infected and uninfected cell images.

python3 train.py -i {path of infected malaria cell images} -u {path of uninfected malaria cell images}

To evaluate the performance of the model you need to have the weight files of the model saved in tmp/modelchkpt/ folder. Run this command:

python3 evaluate.py

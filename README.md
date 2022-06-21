# The HDIN dataset: A Real-world Indoor UAV Dataset with Multi-task Labels for Visual-based Navigation
This repository contains the data preprocessing code for the original HDIN dataset 
## Introduction
Due to limitations that the current public datasets are mostly collected from outdoor environments which lead to the limitations of indoor generalization capabilities, we proposed an HDIN indoor dataset by collecting data only based on the UAV and its onboard sensors with scaling factor labeling methods to overcome the sensor accumulative errors and unidentical label units. This repository contains the code to preprocess the original HDIN dataset for visual-based navigation based on Multi-task supervised learning. 
## Running the code
### Software requirements
This code has been tested on Ubuntu 18.04, and on Python 3.6.

Dependencies:
* TensorFlow 1.5.0
* Keras 2.1.4 (Make sure that the Keras version is correct!)
* NumPy 1.12.1
* OpenCV 3.1.0
* scikit-learn 0.18.1
* Python gflags
* Python matplotlib
* h5py 2.10.0

### Data preparation
The dataset can be downloaded from here: 
#### Dataset processing
Once Udacity dataset is downloaded, extract the contents. After extraction, the original processed structure of the entire dataset should look like this:
```
HDIN
    training/
        collision001/
            images/
            labels.txt
        steer001
            images/
            label.txt
        ...
    validation
        collision006/
            images/
            labels.txt
        steer009
            images/
            label.txt
        ...
    testing
        collision011
            images/
            labels.txt
        steer010
            images/
            label.txt
        ...
```
Download this repository 

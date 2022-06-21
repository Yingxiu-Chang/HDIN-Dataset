# The HDIN dataset: A Real-world Indoor UAV Dataset with Multi-task Labels for Visual-based Navigation
This repository contains the data preprocessing code for the original HDIN dataset.

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
Download this repository ```https://github.com/Yingxiu-Chang/HDIN-Dataset.git``` and copy the ```HDIN/``` folder into this repository path.

There are three scaling factor labeling methods to process the original dataset:
1. Expected Steering labeling method
```
python dataset_preprocessing.py [methods]
```
Use ```[methods]``` to set the labeling method. 
Example:
```
python dataset_preprocessing.py expected_steering
```
2. Fitting Angular Velocity labeling method
Example:
```
python dataset_preprocessing.py fitting_angular_velocity
```
3. Scalable Angular Velocity labeling method
Example:
```
python dataset_preprocessing.py scalable_angular_velocity
```

The final structure of the steering dataset should look like this:
```
HDIN
    training/
        collision001/
            images/
            labels.txt
        steer001
            images/
            (label.txt)
            sync_steering.txt
        ...
    validation
        collision006/
            images/
            labels.txt
        steer009
            images/
            (label.txt)
            sync_steering.txt
        ...
    testing
        collision011
            images/
            labels.txt
        steer010
            images/
            (label.txt)
            sync_steering.txt
        ...
```

### Train DroNet to evaluate the dataset
1. Download the DroNet repository.
```
git clone https://github.com/uzh-rpg/rpg_public_dronet.git
```
2. Copy the training, validation and testing sets from the processed final HDIN dataset to the DroNet path.
3. Modify the 118-122 lines of ```cnn.py``` in DroNet:
```
Generate training data with real-time augmentation
train_datagen = utils.DroneDataGenerator(rotation_range = 0.2,
                                         rescale = 1./255,
                                         width_shift_range = 0.2,
                                         height_shift_range=0.2)
```
to 
```
train_datagen = utils.DroneDataGenerator(rescale = 1./255)
```
4. Following the instructions of DroNet to train the dataset.
Example (RGB images):
```
python cnn.py --experiment_rootdir='./model/xxx' --train_dir='./training' --val_dir='./validation' --batch_size=16 --epochs=150 --log_rate=25 --img_mode='rgb'
```
Example (Gray images):
```
python cnn.py --experiment_rootdir='./model/xxx' --train_dir='./training' --val_dir='./validation' --batch_size=16 --epochs=150 --log_rate=25
```
5. Evaluate model performances.
Example (RGB images):
```
python evaluation.py --experiment_rootdir='./model/xxx' --weights_fname='model_weights_xxx.h5' --test_dir='./testing' --img_mode='rgb'
```
Example (Gray images):
```
python evaluation.py --experiment_rootdir='./model/xxx' --weights_fname='model_weights_xxx.h5' --test_dir='./testing'
```

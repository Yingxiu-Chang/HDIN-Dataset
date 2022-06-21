# ------------------------------------------------------------------
# File Name:        dataset_preprocessing for HDIN dataset
# Author:           Hugh-Chang
# Version:          v1
# Created:          2022/05/27
# ------------------------------------------------------------------
import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import mean_squared_error

# Get all paths of steering subfolders
def get_steering_paths(dataset, img_folder):
    all_img_paths = []
    subsets = os.listdir(dataset)
    for set_name in subsets:
        set_path = os.path.join(dataset, set_name) # Get the path of training or validation or testing
        folders = os.listdir(set_path)
        for subfolder in folders:
            if 'collision' in subfolder: # No need to process collision data
                pass
            else:
                image_folder = os.path.join(set_path, subfolder, img_folder)
                all_img_paths.append(image_folder)
    all_img_paths.sort()
    return all_img_paths

# Load original unprocessed steering text
def load_label(path, text):
    label_path = os.path.join(path.strip('/images'), text)
    # Load original label text
    timestamps = np.loadtxt(label_path, dtype=int, delimiter=',', usecols=(0))  # timestamps of unprocessed steering text
    steering = np.loadtxt(label_path, dtype=float, delimiter=',', usecols=(1))  # steering values of unprocessed steering text
    return timestamps, steering

# Load images from Image path
def load_image(path):
    # Load images
    images = [os.path.basename(x) for x in glob.glob(path + "/*.jpg")]
    # Extract timestamps of images
    im_stamps = []
    for im in images:
        stamp = int(re.sub(r'\.jpg$', '', im))
        im_stamps.append(stamp)
    im_stamps.sort()
    return im_stamps

# Matching function
def Matching(s_stamps, im_stamps, raw_S):
    matching_S = np.zeros(len(im_stamps))
    # time matching raw steering values with image timestamps
    for i in range(len(im_stamps)):
        idx = np.where((s_stamps - im_stamps[i]) >= 0)
        matching_S[i] = raw_S[idx[0][0]]
    return matching_S

# Transformation function
def Transformation(raw_S):
    S_r = raw_S - raw_S[0] # Sr=S-S0
    trans_S = np.zeros(len(S_r)) # initialization
    for i in range(len(S_r)):
        if (raw_S[0] > 0 and S_r[i] < -180):
            trans_S[i] = S_r[i] + 360 # St = Sr+360 when S0>0 and Sr<-180
        elif (raw_S[0] < 0 and S_r[i] > 180):
            trans_S[i] = S_r[i] - 360 # St = Sr-360 when S0<0 and Sr>180
        else:
            trans_S[i] = S_r[i]
    return trans_S

# Rotation function
def rotation(raw_S):
    expected_S = np.zeros(len(raw_S))
    for i in range(len(raw_S)):
        if i != len(raw_S) - 1:
            expected_S[i] = raw_S[i+1] - raw_S[i] # Se = St[i+1] - St[i]
        else:
            expected_S[i] = 0
    return expected_S

# Low-pass filter
def low_pass(raw_S, alpha):
    theta_S = np.zeros(len(raw_S))
    initial_theta = 0
    for i in range(len(raw_S)):
        if i == 0:
            theta_S[i] = (1 - alpha) * initial_theta + alpha * raw_S[i] # theta[i] = (1-alpha)*Se[i-1]+alpha*Se[i]
        else:
            theta_S[i] = (1 - alpha) * theta_S[i-1] + alpha * raw_S[i]
    return theta_S

# Write synchronization text
def write_label(path, text, values):
    write_path = os.path.join(path.strip('/images'), text)
    with open(write_path, 'w') as f:
        f.write('# steering angle\n')
        for index in range(len(values)):
            f.write(str(values[index]) + '\n')
        f.close()

# Fitting function (Polynomial Fitting)
def Fitting(s_stamps, raw_S):
    x_label = (s_stamps - s_stamps[0]) * 10 ** (-9) # seconds as X label
    loss = [] # Loss between fitting and original steering
    fit = [] # List of fitting formular parameter
    pred = [] # fitting values
    for deg in range(0, 21): # Find the minimum deg for the best fitting
        fit.append(np.polyfit(x_label, raw_S, deg))
        pred.append(np.polyval(fit[-1], x_label))
        loss.append(mean_squared_error(pred[-1], raw_S))
    # Get min deg
    min_loss_deg = loss.index(min(loss))
    return fit, min_loss_deg, x_label

# First Derivative function for angular velocity
def derivative(fit, min_deg, x):
    fx = np.poly1d(fit[min_deg])
    dfx = fx.deriv()
    grad = dfx(x)
    return grad

# Algorithm 1: Expected Steering
def expected_steering(paths, read_text, write_text):
    path_num = len(paths)
    alpha = 0.1 # Set alpha=0.1 for low-pass filter
    for i in range(path_num):
        image_path = paths[i] # image and label paths
        L_ts, S = load_label(image_path, read_text) # Get steering values and timestamps
        I_ts = load_image(image_path) # Load images
        S_m = Matching(L_ts, I_ts, S) # Time-matching steering values
        S_t = Transformation(S_m) # Steering values transformation
        S_e = rotation(S_t) # Calculated expected steering values
        theta = low_pass(S_e, alpha) # Get steering values from low-pass fillter
        write_label(image_path, write_text, theta) # Write final label text
        # Uncomment next line to delete the original unprocessed label text (unnecessary)
        # os.remove(os.path.join(image_path.strip('/images'), read_text))

# Algorithm 2: Fitting Angular Velocity
def fit_ang_vel(paths, read_text, write_text):
    path_num = len(paths)
    for i in range(path_num):
        image_path = paths[i] # image and label paths
        L_ts, S = load_label(image_path, read_text) # Get steering values and timestamps
        I_ts = load_image(image_path) # Load images
        S_t = Transformation(S) # Steering values transformation
        S_f, min_deg, X = Fitting(L_ts, S_t) # Polynomial fitting steering
        S_d = derivative(S_f, min_deg, X) # Derivative steering values
        S_m = Matching(L_ts, I_ts, S_d) # Time-matching steering values
        theta = np.radians(S_m) # Get radians
        write_label(image_path, write_text, theta) # Write final label text
        # Uncomment next line to delete the original unprocessed label text (unnecessary)
        # os.remove(os.path.join(image_path.strip('/images'), read_text))

# Algorithm 3: Scalable Angular Velocity
def scalable_ang_vel(paths, read_text, write_text):
    path_num = len(paths)
    max_angvel = 40 # Set the max angular velocity as 40
    for i in range(path_num):
        image_path = paths[i] # image and label paths
        L_ts, S = load_label(image_path, read_text) # Get steering values and timestamps
        I_ts = load_image(image_path) # Load images
        S_t = Transformation(S) # Steering values transformation
        S_f, min_deg, X = Fitting(L_ts, S_t) # Polynomial fitting steering
        S_d = derivative(S_f, min_deg, X) # Derivative steering values
        S_m = Matching(L_ts, I_ts, S_d) # Time-matching steering values
        theta = S_m/max_angvel # Get scalable steering values
        write_label(image_path, write_text, theta) # Write final label text
        # Uncomment next line to delete the original unprocessed label text (unnecessary)
        # os.remove(os.path.join(image_path.strip('/images'), read_text))

# Main function
def main(argv):
    # Paths and label names
    root = './HDIN' # dataset root
    image_folder = 'images' # image folder name
    read_label = 'label.txt' # original unprocessed steering label text
    write_label = 'sync_steering.txt' # Synchronization steering text

    # Correctly enter scaled steering algorithm
    try:
        if 'expected' in argv[1]:
            all_image_paths = get_steering_paths(root, image_folder) # Get all image paths
            expected_steering(all_image_paths, read_label, write_label)
            print('Alg1: Expected Steering Done')
        elif 'fitting' in argv[1]:
            all_image_paths = get_steering_paths(root, image_folder)  # Get all image paths
            fit_ang_vel(all_image_paths, read_label, write_label)
            print('Alg2: Fitting Angular Velocity Done')
        elif 'scalable' in argv[1]:
            all_image_paths = get_steering_paths(root, image_folder)  # Get all image paths
            scalable_ang_vel(all_image_paths, read_label, write_label)
            print('Alg3: Scalable Angular Velocity Done')
        else:
            print('Enter wrong, please check algorithm spelling')
    except IndexError:
        print('Please Reenter algorithm')

if __name__ == "__main__":
    main(sys.argv)
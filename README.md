# Fast and Compact Deep Gabor Network for Micro-Doppler Signal Processing and Human Motion Classification
## Descriptions
This project proposes a new light-weight Bayesian Gabor Network for camera-based detection of pedestrian lanes in unstructured scenes. The proposed method is fast, compact, and
suitable for real-time operations on edge computers.

![alt_text](/image/no_arm_swinging.jpg) ![alt_text](/image/stft_0_arm.jpg) ![alt_text](/image/smethod_0_arm.jpg) ![alt_text](/image/cwt_0_arm.jpg)

![alt_text](/image/one_arm_swinging.jpg) ![alt_text](/image/stft_1_arm.jpg) ![alt_text](/image/smethod_1_arm.jpg) ![alt_text](/image/cwt_1_arm.jpg)

![alt_text](/image/two_arm_swinging.jpg) ![alt_text](/image/stft_2_arm.jpg) ![alt_text](/image/smethod_2_arm.jpg) ![alt_text](/image/cwt_2_arm.jpg)
**Figure 1.** A subject walking with three motion types towards the radar system and the corresponding signal representations.

## Quick start
### Installation
1. Install Tensorflow=1.13.1 and Keras=2.2.4 following [the official instructions](https://www.tensorflow.org/install/pip)

2. git clone https://github.com/hthanhle/Gabor-Networks/

3. Install dependencies: `pip install -r requirements.txt`

### Train and test

Please specify the configuration file. 

1. To train a Gabor Network, run the following command: `python train.py`

2. To test a pretrained Gabor Detector, run the following command:
**Example 1:** `python test.py`
**Example 3:** `python test.py --weight ./checkpoints/acc_0.9543_loss_0.0868.h5 --block 3`

## Citation
If you find this work or code is helpful for your research, please cite:


## Reference


# Fast and Compact Deep Gabor Network for Micro-Doppler Signal Processing and Human Motion Classification
## Descriptions
We propose a new light-weight Bayesian Gabor Network for camera-based detection of pedestrian lanes in unstructured scenes. The proposed method is fast, compact, and
suitable for real-time operations on edge computers.

![alt_text](/image/no_arm_swinging.jpg) ![alt_text](/image/stft_0_arm.png) ![alt_text](/image/smethod_0_arm.png) ![alt_text](/image/cwt_0_arm.png)

![alt_text](/image/one_arm_swinging.jpg) ![alt_text](/image/stft_1_arm.png) ![alt_text](/image/smethod_1_arm.png) ![alt_text](/image/cwt_1_arm.png)

![alt_text](/image/two_arm_swinging.jpg) ![alt_text](/image/stft_2_arm.png) ![alt_text](/image/smethod_2_arm.png) ![alt_text](/image/cwt_2_arm.png)

**Figure 1.** A subject walking with three motion types (no-arm swinging, one-arm swinging, and two-arms swinging) towards the radar system and the corresponding signal representations. Column 2: Short-time Fourier Transform. Column 3: Smethod. Column 4: Continous Wavelet Transform.

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
```
@ARTICLE{9519700,
  author={Le, Hoang Thanh and Phung, Son Lam and Bouzerdoum, Abdesselam},
  journal={IEEE Sensors Journal}, 
  title={A Fast and Compact Deep Gabor Network for Micro-Doppler Signal Processing and Human Motion Classification}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JSEN.2021.3106300}}
```

## Reference


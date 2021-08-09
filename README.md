# Fast and Compact Deep Gabor Network for Micro-Doppler Signal Processing and Human Motion Classification
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


# SincNet in Tensorflow
An Implementation of <a href="https://github.com/mravanelli/SincNet">SincNet</a> using Tenorflow 2.x.
- Models are converted from original torch networks.
- The main implementation of the <a href="https://github.com/mravanelli/SincNet">sinc_conv</a> layer is non-optimal. Instead of using loops in the call section, we used matrix multiplication and a few programming tricks that allow the hardware to run more efficiently (25 times faster).

# SincNet
SincNet is a neural architecture for processing **raw audio samples**. It is a novel Convolutional Neural Network (CNN) that encourages the first convolutional layer to discover more **meaningful filters**. SincNet is based on parametrized sinc functions, which implement band-pass filters. [Arxiv](http://arxiv.org/abs/1808.00158)


## Install

```bash
$ pip install sincnet-tensorflow
`


## Usage
### Demo
Training on a dummy database to check for error-free execution 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AryaAftab/sincnet-tensorflow/blob/master/demo/sincnet_tensorflow_demo.ipynb)

### A layer for Keras Functional
```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Flatten, MaxPooling1D, Input

from sincnet_tensorflow import SincConv1D, LayerNorm


out_dim = 50 #number of classes

sinc_layer = SincConv1D(N_filt=64,
                        Filt_dim=129,
                        fs=16000,
                        stride=16,
                        padding="SAME")


inputs = Input((32000, 1)) 

x = sinc_layer(inputs)
x = LayerNorm()(x)

x = LeakyReLU(alpha=0.2)(x)
x = MaxPooling1D(pool_size=2)(x)


x = Conv1D(64, 3, strides=1, padding='valid')(x)
x = BatchNormalization(momentum=0.05)(x)
x = LeakyReLU(alpha=0.2)(x)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(64, 3, strides=1, padding='valid')(x)
x = BatchNormalization(momentum=0.05)(x)
x = LeakyReLU(alpha=0.2)(x)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(128, 3, strides=1, padding='valid')(x)
x = BatchNormalization(momentum=0.05)(x)
x = LeakyReLU(alpha=0.2)(x)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(128, 3, strides=1, padding='valid')(x)
x = BatchNormalization(momentum=0.05)(x)
x = LeakyReLU(alpha=0.2)(x)
x = MaxPooling1D(pool_size=2)(x)

x = Flatten()(x)

x = Dense(256)(x)
x = BatchNormalization(momentum=0.05, epsilon=1e-5)(x)
x = LeakyReLU(alpha=0.2)(x)

x = Dense(256)(x)
x = BatchNormalization(momentum=0.05, epsilon=1e-5)(x)
x = LeakyReLU(alpha=0.2)(x)

prediction = Dense(out_dim, activation='softmax')(x)
model = tf.keras.models.Model(inputs=inputs, outputs=prediction)

model.summary()
```


## References
```bibtex
@inproceedings{ravanelli2018speaker,
  title={Speaker recognition from raw waveform with sincnet},
  author={Ravanelli, Mirco and Bengio, Yoshua},
  booktitle={2018 IEEE Spoken Language Technology Workshop (SLT)},
  pages={1021--1028},
  year={2018},
  organization={IEEE}
}

@misc{SincNet,
    title   = {SincNet}, 
    author  = {Mirco Ravanelli (mravanelli)},
    year    = {2018},
    url  = {https://github.com/mravanelli/SincNet},
    publisher = {Github},
}
```

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
# A port of [SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd) to [Keras](https://keras.io) framework.
For more details, please refer to [arXiv paper v5](http://arxiv.org/abs/1512.02325).
For normal classification of the 512x512 model, use `SSD.ipynb` for examples. 

For training procedure for 512x512 model, follow `SSD_training.ipynb`.


VGG_coco_SSD_512x512_iter_360000.caffemodel is hosted at [weilius original repository](https://github.com/weiliu89/caffe/blob/ssd/README.md#models).

Weights are ported from the original models and are available [here](https://mega.nz/#F!7RowVLCL!q3cEVRK9jyOSB9el3SssIA). You need `weights_SSD300.hdf5`, `weights_300x300_old.hdf5` is for the old version of architecture with 3x3 convolution for `pool6`.

This code was tested with `Keras` v1.2.2 and `Tensorflow` v1.0.0

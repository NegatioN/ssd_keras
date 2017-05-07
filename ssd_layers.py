"""Some special pupropse layers for SSD."""

import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
import logging

logging.basicConfig()
logger = logging.getLogger("SSD_LOGGER")
logger.setLevel(logging.INFO)
smin = 0.15
smax = 0.9


class Normalize(Layer):
    """Normalization layer as described in ParseNet paper.

    # Arguments
        scale: Default feature scale.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        Same as input

    # References
        http://cs.unc.edu/~wliu/papers/parsenet.pdf

    #TODO
        Add possibility to have one scale for all features.
    """

    def __init__(self, scale, **kwargs):
        if K.image_dim_ordering() == 'tf':
            self.axis = 3
        else:
            self.axis = 1
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = K.variable(init_gamma, name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        output *= self.gamma
        return output


class PriorBox(Layer):
    """Generate the prior boxes of designated sizes and aspect ratios.
    # Arguments
        img_size: Size of the input image as tuple (w, h).
        aspect_ratios: List of aspect ratios of boxes.
        flip: Whether to consider reverse aspect ratios.
        variances: List of variances for x, y, w, h.
        clip: Whether to clip the prior's coordinates
            such that they are within [0, 1].

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        3D tensor with shape:
        (samples, num_boxes, 8)

    # References
        https://arxiv.org/abs/1512.02325
    """

    def __init__(self, img_size, k, aspect_ratios, variances=[0.1], clip=True, **kwargs):
        self.k = k
        if K.image_dim_ordering() == 'tf':
            self.waxis, self.haxis = 2, 1
        else:
            self.waxis, self.haxis = 3, 2
        self.img_size = img_size
        self.aspect_ratios = aspect_ratios
        self.variances = np.array(variances)
        self.clip = clip
        super(PriorBox, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        num_priors_ = len(self.aspect_ratios) + 1
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]
        num_boxes = num_priors_ * layer_width * layer_height
        return (input_shape[0], num_boxes, 8)

    def call(self, x, mask=None):
        input_shape = find_input_shape(x)
        logger.debug(".call input-shape: {}".format(input_shape))
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]
        step = step_k(self.k)
        # because we add special handling for ratio 1 TODO fix pretty
        num_priors = len(self.aspect_ratios) + 1 if 1.0 in self.aspect_ratios else len(self.aspect_ratios)

        box_widths, box_heights = box_height_widths(self.aspect_ratios, step)
        prior_boxes = make_prior_boxes(layer_width=layer_width,
                                       layer_height=layer_height,
                                       step=step,
                                       box_heights=box_heights,
                                       box_widths=box_widths,
                                       num_priors=num_priors,
                                       clip=self.clip)
        return reshape_for_output(x=x,
                                  prior_boxes=prior_boxes,
                                  variances=self.variances)


def box_height_widths(aspect_ratios, step_k):
    box_widths, box_heights = [], []

    checker = dict(zip(aspect_ratios, range(len(aspect_ratios))))
    if len(checker) != len(aspect_ratios):
        raise Exception('Duplicate aspect_ratios present')  # temporary solution?

    for ar in aspect_ratios:
        if ar == 1.0:
            # Special casing for aspect-ratio 1, as stated on page 6
            # in the paper this says sqrt(step_k * step_k + 1), but the impl differs
            # https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/prior_box_layer.cpp#L165
            box_heights.append(np.sqrt(step_k))
            box_widths.append(np.sqrt(step_k))
            box_heights.append(np.sqrt(step_k * smax))
            box_widths.append(np.sqrt(step_k * smax))
        else:
            box_widths.append(step_k * np.sqrt(ar))
            box_heights.append(step_k / np.sqrt(ar))

    # Half these since we take steps in both directions later. We are kinda using a "radius" here. Refactor?
    box_widths = 0.25 * np.array(box_widths)
    box_heights = 0.25 * np.array(box_heights)
    return box_widths, box_heights


def reshape_for_output(x, prior_boxes, variances):
    r_variances = reshape_variances(variances, num_boxes=len(prior_boxes))
    prior_boxes = np.concatenate((prior_boxes, r_variances), axis=1)
    keras_tensor = K.variable(prior_boxes)
    keras_tensor = K.expand_dims(keras_tensor, 0)
    prior_boxes_tensor = K.tile(keras_tensor, [tf.shape(x)[0], 1, 1])
    return prior_boxes_tensor


def box_centers(layer_width, layer_height, step):
    logger.debug("box_centers: Step={}, layer_height={}, layer_width={}".format(step, layer_height, layer_width))
    linx = np.linspace((0.5 * step) / layer_width, (layer_width - 0.5 * step) / layer_width, layer_width)
    liny = np.linspace((0.5 * step) / layer_height, (layer_height - 0.5 * step) / layer_height, layer_height)
    logger.debug("Box-centers Lin-x: {}".format(linx))
    logger.debug("Box-centers Lin-y: {}".format(liny))
    centers_x, centers_y = np.meshgrid(linx, liny)
    centers_x = centers_x.reshape(-1, 1)
    centers_y = centers_y.reshape(-1, 1)
    return centers_x, centers_y


def make_prior_boxes(layer_height, layer_width, step, box_heights, box_widths, num_priors, clip):
    centers_x, centers_y = box_centers(layer_height=layer_height,
                                       layer_width=layer_width,
                                       step=step)
    # define xmin, ymin, xmax, ymax of prior boxes
    prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
    logger.debug("Prior_Boxes centers: {}".format(prior_boxes))
    logger.debug("Prior_Boxes: width={} heights={}".format(box_widths, box_heights))
    prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors))
    prior_boxes[:, 0::4] -= box_widths
    prior_boxes[:, 1::4] -= box_heights
    prior_boxes[:, 2::4] += box_widths
    prior_boxes[:, 3::4] += box_heights
    logger.debug("Prior_Boxes:\n{}".format(prior_boxes))
    prior_boxes = prior_boxes.reshape(-1, 4)
    if clip:
        prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
    return prior_boxes

def reshape_variances(variances, num_boxes):
    if len(variances) == 1:
        return np.ones((num_boxes, 4)) * variances[0]
    elif len(variances) == 4:
        return np.tile(variances, (num_boxes, 1))
    else:
        raise Exception('Must provide one or four variances.')


def find_input_shape(x):
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    elif hasattr(K, 'int_shape'):
        return K.int_shape(x)


# Change m if you add more PredictionBlocks to the model.
def step_k(k):
    m = 7.0
    return smin + (smax - smin) * (k - 1.0) / (m - 1.0)  # equation 4

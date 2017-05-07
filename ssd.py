from __future__ import division, print_function

from keras.models import Model
from keras.layers import Reshape, Activation, Input, concatenate
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import keras.backend as K

from ssd_layers import PriorBox, Normalize
import vgg16


class SSD:
    """SSD with a better overview"""

    def __init__(self, size=(512, 512), num_classes=80):
        self.mlocs, self.mconf, self.mboxes = [], [], []
        self.size = size
        self.num_classes = num_classes + 1  # extra class used to infer positive or negative loss at training-time.
        self.model = self.create(size=size)

    def create(self, size):
        # VGG16 base
        vgg = vgg16.VGG()
        input_shape = size + (3,) if K.backend() == 'tensorflow' else (3,) + size
        inp = Input(shape=input_shape, name="inp")

        x = vgg.FuncConvBlock(inp, 2, 64)
        x = vgg.FuncConvBlock(x, 2, 128)
        x = vgg.FuncConvBlock(x, 3, 256)
        conv4 = vgg.FuncConvBlock(x, 3, 512, max_pool=False)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(conv4)
        base_model = vgg.FuncConvBlock(x, 3, 512, max_pool=False)

        conv4_3_norm = Normalize(20, name='conv4_3_norm')(conv4)

        # SSD
        fc6 = Conv2D(filters=1024, kernel_size=3, dilation_rate=6, activation='relu', padding='same', name='fc6')(base_model)
        fc7 = Conv2D(filters=1024, kernel_size=1, activation='relu', padding='same', name='fc7')(fc6)
        conv6_2 = self.ConvSSDBlock(fc7, name="conv6", filters=256, stride=(2, 2))
        conv7_2 = self.ConvSSDBlock(conv6_2, name="conv7", filters=128, stride=(2, 2))
        conv8_2 = self.ConvSSDBlock(conv7_2, name="conv8", filters=128, stride=(2, 2))
        conv9_2 = self.ConvSSDBlock(conv8_2, name="conv9", filters=128, stride=(2, 2))
        conv10_2 = self.ConvSSDBlock(conv9_2, name="conv10", filters=128, stride=(1, 1), filter_size=4)

        model = Model(inp, [conv10_2, conv4_3_norm])  # Temporary model to avoid passing all layers to pred_blocks

        # SSD Prediction Blocks
        self.add_prediction_blocks(model)

        return self.finalize_model(inp)

    def add_prediction_blocks(self, model):
        # Clear list of layers for easy management
        self.model = model
        self.mlocs, self.mconf, self.mboxes = [], [], []
        self.PredictionBlock(k=1, attach_layer_name='conv4_3_norm', aspect_ratios=[1.0, 2.0, 1.0 / 2])
        self.PredictionBlock(k=2, attach_layer_name='fc7')
        self.PredictionBlock(k=3, attach_layer_name='conv6_2')
        self.PredictionBlock(k=4, attach_layer_name='conv7_2')
        self.PredictionBlock(k=5, attach_layer_name='conv8_2')
        self.PredictionBlock(k=6, attach_layer_name='conv9_2', aspect_ratios=[1.0, 2.0, 1.0 / 2])
        self.PredictionBlock(k=7, attach_layer_name='conv10_2', aspect_ratios=[1.0, 2.0, 1.0 / 2])

    def finalize_model(self, in_layer):
        merged_mlocs = concatenate(self.mlocs, axis=1, name='multibox_m_locations')
        merged_mconf = concatenate(self.mconf, axis=1, name='multibox_m_confidence')
        # Get number of boxes in mlocs. Four points per box.
        num_boxes = self.get_layer_output_shape(merged_mlocs)[-1] // 4
        logits_mconf = Reshape((num_boxes, self.num_classes), name='multibox_confidence_logits')(merged_mconf)
        final_mlocs = Reshape((num_boxes, 4), name='multibox_locations_final')(merged_mlocs)
        final_mboxes = concatenate(self.mboxes, axis=1, name='multibox_priorbox')
        final_mconf = Activation('softmax', name='multibox_confidence_final')(logits_mconf)

        prediction_output = concatenate([final_mlocs, final_mconf, final_mboxes], axis=2, name='predictions')
        return Model(in_layer, prediction_output)

    '''
        A block we connect to a layer to generate some predictions about which bounding-boxes might fit
        Arguments
            attach_layer_name: layer to attach the PredictionBlock to
            k: Number in the sequence of Prediction-blocks. Used to scale boxes for the layer.
            aspect_ratio: Which aspect-ratios to include for the boxes of this layer.
    '''

    def PredictionBlock(self, attach_layer_name, k, aspect_ratios=None, padding='same'):
        aspect_ratios = self.fix_aspect_ratios(aspect_ratios)
        # aspect_ratio 1 is a special case, and is handled twice, so we need to count it twice.
        num_priors = len(aspect_ratios) + 1 if 1 in aspect_ratios else len(aspect_ratios)
        # Classification-layer
        # The conf and loc parts make up the 3*3(num_prior*(Classes+4)) we see in the paper.
        attach_layer = self.get_layer_output(attach_layer_name)

        x = Conv2D(filters=num_priors * self.num_classes, kernel_size=3, padding=padding,
                   name='{}_mbox_conf'.format(attach_layer_name))(attach_layer)
        flatten = Flatten(name=("{}_mbox_conf_flat".format(attach_layer_name)))
        self.mconf.append(flatten(x))
        x = Conv2D(filters=num_priors * 4, kernel_size=3, padding=padding,
                   name='{}_mbox_loc'.format(attach_layer_name))(attach_layer)
        flatten = Flatten(name=("{}_mbox_loc_flat".format(attach_layer_name)))
        self.mlocs.append(flatten(x))

        ## Make PriorBox
        self.mboxes.append(PriorBox(self.size, k=k, aspect_ratios=aspect_ratios, variances=[0.1, 0.1, 0.2, 0.2],
                                    name=('{}_mbox_priorbox'.format(attach_layer_name)))(attach_layer))

    @staticmethod
    def ConvSSDBlock(layer, name, filters, stride=(2, 2), filter_size=3):
        x = Conv2D(filters=filters, kernel_size=1, strides=(1, 1), activation='relu', padding='same',
                   name="{}_1".format(name))(layer)
        return Conv2D(filters=filters * 2, kernel_size=filter_size, strides=stride, activation='relu', padding='same',
                      name="{}_2".format(name))(x)

    def load_weights_finetune(self, num_classes, weight_loc, freeze_except_priors=True):
        self.model.load_weights(weight_loc)
        self.finetune(num_classes, freeze_except_priors)

    def finetune(self, num_classes, freeze_except_priors):
        self.num_classes = num_classes + 1  # extra class used to infer positive or negative loss at training-time.
        in_layer = self.model.get_layer("inp").input
        self.add_prediction_blocks(self.model)
        model = self.finalize_model(in_layer)
        if freeze_except_priors:
            self.freeze_model(model, 33)  # Freeze after SSD Convolutional layers
        else:
            self.freeze_model(model, 22)  # Freeze after VGG model-layers
        self.model = model

    @staticmethod
    def freeze_model(model, num_freeze_layers):
        for layer in model.layers:
            layer.trainable = True
        for i, layer in enumerate(model.layers):
            if i < num_freeze_layers:
                layer.trainable = False

    @staticmethod
    def get_layer_output_shape(layer):
        if hasattr(layer, '_keras_shape'):
            return layer._keras_shape
        elif hasattr(layer, 'int_shape'):
            return K.int_shape(layer)

    @staticmethod
    def fix_aspect_ratios(aspect_ratios):
        if aspect_ratios is None:
            aspect_ratios = [1, 2, 3, 1.0 / 2, 1.0 / 3]  # default aspect-ratios in the paper.
        # make sure everything is floats so the system doesnt come crashing down.
        return [float(x) for x in aspect_ratios]

    def fit_model(self, ssd_generator, epochs, batch_size, callbacks=None):
        self.model.fit_generator(ssd_generator.generate(True),
                                 steps_per_epoch=int(ssd_generator.train_batches / batch_size),
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=callbacks,
                                 validation_data=ssd_generator.generate(False),
                                 validation_steps=int(ssd_generator.val_batches / batch_size),
                                 workers=1)

    def get_layer_output(self, layer_name):
        return self.model.get_layer(layer_name).output

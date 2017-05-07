from keras.layers.convolutional import Convolution2D, MaxPooling2D

class VGG:
    def __init__(self, border_mode="same"):
        # count is used to make sure conv-layers get same names as the loaded weights, which are loaded by name.
        self.count = 1
        self.border_mode = border_mode

    def FuncConvBlock(self, layer, layers, filters, max_pool=True):
        subcount = 1
        x = layer
        for i in range(layers):
            x = Convolution2D(nb_filter=filters,
                              nb_row=3,
                              nb_col=3,
                              activation='relu',
                              border_mode=self.border_mode,
                              name="conv{}_{}".format(self.count, subcount))(x)
            subcount += 1
        self.count += 1
        if max_pool:
            x = MaxPooling2D(pool_size=(2, 2),
                             strides=(2, 2),
                             border_mode=self.border_mode)(x) #Border mode must be same to match expected output.
        return x

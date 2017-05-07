from keras.layers.convolutional import Conv2D, MaxPooling2D

class VGG:
    def __init__(self, padding="same"):
        # count is used to make sure conv-layers get same names as the loaded weights, which are loaded by name.
        self.count = 1
        self.padding = padding

    def FuncConvBlock(self, layer, layers, filters, max_pool=True):
        subcount = 1
        x = layer
        for i in range(layers):
            x = Conv2D(filters=filters, kernel_size=3, activation='relu', padding=self.padding,
                       name="conv{}_{}".format(self.count, subcount))(x)
            subcount += 1
        self.count += 1
        if max_pool:
            x = MaxPooling2D(pool_size=(2, 2),
                             strides=(2, 2),
                             padding=self.padding)(x) #Border mode must be same to match expected output.
        return x

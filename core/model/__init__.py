import paddle.nn


class DConv2D(paddle.nn.Layer):
    def __init__(self, channels_i, channels_o):
        super().__init__()

        self.residual = paddle.nn.Sequential(
            paddle.nn.Conv2D(channels_i, channels_o, kernel_size=3, padding='same', bias_attr=False),
            paddle.nn.BatchNorm2D(channels_o),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(channels_o, channels_o, kernel_size=3, padding='same', bias_attr=False),
            paddle.nn.BatchNorm2D(channels_o)
        )

        if channels_i == channels_o:
            self.shortcut = paddle.nn.Sequential()
        else:
            self.shortcut = paddle.nn.Sequential(
                paddle.nn.Conv2D(channels_i, channels_o, kernel_size=1),
                paddle.nn.BatchNorm2D(channels_o)
            )

    def forward(self, x):
        return paddle.nn.ReLU()(self.residual(x) + self.shortcut(x))


class Encoder(paddle.nn.Layer):
    def __init__(self, channels):
        super().__init__()

        self.layers = [
            paddle.nn.Sequential(
                paddle.nn.MaxPool2D(kernel_size=2, stride=2),
                DConv2D(channels[i], channels[i + 1] if i + 1 < len(channels) else channels[i])
            )
            for i in range(len(channels))
        ]

    def forward(self, x):
        d = []
        for layer in self.layers:
            x, d = layer(x), d + [x]
        return x, d


class Decoder(paddle.nn.Layer):
    def __init__(self, channels):
        super().__init__()

        self.layers = [
            DConv2D(channels[i] * 2, channels[i + 1] if i + 1 < len(channels) else channels[i])
            for i in range(len(channels))
        ]

    def forward(self, x, d):
        d = list(reversed(d))
        for i in range(len(self.layers)):
            x = paddle.nn.functional.interpolate(x, paddle.shape(d[i])[2:], mode='bilinear')
            x = paddle.concat([x, d[i]], axis=1)
            x = self.layers[i](x)
        return x


class UNet(paddle.nn.Layer):
    def __init__(self, channels_i, channels_o, channels_w):
        super().__init__()

        self.conv2d = DConv2D(channels_i, channels_w[0])

        self.encode = Encoder(channels_w)
        self.decode = Decoder(list(reversed(channels_w)))

        self.classify = paddle.nn.Conv2D(channels_w[0], channels_o, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x, d = self.encode(self.conv2d(x))
        return self.classify(self.decode(x, d))

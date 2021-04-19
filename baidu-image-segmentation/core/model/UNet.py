from paddleseg.models import UNet
from paddleseg.models import layers
import paddle.nn as nn

def UNetOurs(numClass):
    model = UNet(numClass)
    model.encode.double_conv = nn.Sequential(
            layers.ConvBNReLU(1, 64, 3), layers.ConvBNReLU(64, 64, 3))
    #print(model)
    return model
if __name__ == "__main__":
    model = UNet(2)
    model.encode.double_conv = nn.Sequential(
            layers.ConvBNReLU(1, 64, 3), layers.ConvBNReLU(64, 64, 3))
    print(model)
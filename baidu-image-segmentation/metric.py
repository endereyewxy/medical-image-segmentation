import paddle.fluid as fluid
import numpy as np
import paddle

def DiceMetric(predict, label):
    predict = fluid.layers.softmax(predict, axis=1)
    predict = paddle.argmax(predict, axis=1, keepdim=True)
    predict = predict.numpy()
    label = label.numpy()

    predict = predict.flatten()
    label = label.flatten()

    intersection = np.sum(predict * label)
    pre_sum = np.sum(predict)
    lab_sum = np.sum(label)

    return (2. * intersection + 1e-9) / (np.sum(predict) + np.sum(label) + 1e-9)



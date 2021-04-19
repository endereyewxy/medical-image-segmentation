import paddle
import paddle.fluid as fluid


# class DiceLoss(paddle.nn.Layer):
#     """
#     1. 继承paddle.nn.Layer
#     """
#     def __init__(self):
#         """
#         2. 构造函数根据自己的实际算法需求和使用需求进行参数定义即可
#         """
#         super(DiceLoss, self).__init__()

#     def forward(self, predict, label):
#         """
#         3. 实现forward函数，forward在调用时会传递两个参数：input和label
#             - predict：单个或批次训练数据经过模型前向计算输出结果
#             - label：单个或批次训练数据对应的标签数据
#             接口返回值是一个Tensor，根据自定义的逻辑加和或计算均值后的损失
#         """
#         # 使用Paddle中相关API自定义的计算逻辑
#         predict = fluid.layers.softmax(predict,axis=1)
#         predict = fluid.layers.transpose(predict, perm=[0, 2, 3, 1])
#         #predict = fluid.layers.reshape(predict, shape=[-1, 2])

#         label= fluid.layers.transpose(label, perm=[0, 2, 3, 1])
#         #label = fluid.layers.reshape(label, shape=[-1, 1])
#         #entropy = fluid.layers.cross_entropy(predict, label)#交叉熵
#         dice_cost = fluid.layers.dice_loss(input=predict, label=label)#dice_loss

#         return dice_cost

class DiceLoss(paddle.nn.Layer):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predict, label):
        epsilon = 1e-9

        # predict = fluid.layers.transpose(predict, perm=[0, 2, 3, 1])
        # label = fluid.layers.transpose(label, perm=[0, 2, 3, 1])

        predict = fluid.layers.softmax(predict, axis=1)
        predict = predict[:, 1:, :, :]

        reduce_dim = list(range(0, len(predict.shape)))
        inse = fluid.layers.reduce_sum(predict * label, dim=reduce_dim)
        dice_denominator = fluid.layers.reduce_sum(
            predict, dim=reduce_dim) + fluid.layers.reduce_sum(
            label, dim=reduce_dim)
        dice_score = 1 - inse * 2 / (dice_denominator + epsilon)

        return dice_score


def diceloss(predict, label):
    # predict = fluid.layers.argmax(predict, axis=-1)
    # predict = fluid.one_hot(predict, 4)
    predict = fluid.layers.softmax(predict, axis=-1)
    dice = fluid.layers.dice_loss(predict, label)
    return dice


class CELoss(paddle.nn.Layer):
    def __init__(self):
        super(CELoss, self).__init__()
        self.ceLossO = paddle.nn.CrossEntropyLoss(axis=1)

    def forward(self, predict, label):
        loss = self.ceLossO(predict, label)
        return loss


class DiceCELoss(paddle.nn.Layer):
    def __init__(self):
        super(DiceCELoss, self).__init__()
        self.diceLossO = DiceLoss()
        self.ceLossO = paddle.nn.CrossEntropyLoss(axis=1)

    def forward(self, predict, label):
        diceLoss = self.diceLossO(predict, label)
        ceLoss = self.ceLossO(predict, label)
        loss = diceLoss + ceLoss
        return loss

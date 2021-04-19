import numpy as np
import paddle.io
import nibabel as nib
# from core.model import UNet
from core.model.UNet import UNetOurs
from loss import DiceCELoss, DiceLoss, diceloss, CELoss
from metric import DiceMetric
import os
from paddle.io import Dataset, DataLoader
import math
import paddle.fluid as fluid


class DatasetOurs(Dataset):
    def __init__(self, image_path, label_path):

        self.image_paths = []
        self.label_paths = []

        file1_list_onegrade = os.listdir(image_path)
        file2_list_onegrade = os.listdir(label_path)
        for image_name in file1_list_onegrade:
            self.image_paths.append(os.path.join(image_path, image_name))
        for label_name in file2_list_onegrade:
            self.label_paths.append(os.path.join(label_path, label_name))
        # self.images_labels = self._get_total()
        self.positive, self.negative = self._get_total()
        self.randomFuse()

    def _get_total(self):
        positive_imgs = []
        negative_imgs = []
        for img_path, lab_path in zip(self.image_paths, self.label_paths):
            img = nib.load(img_path)
            lab = nib.load(lab_path)
            width, height, queue = img.dataobj.shape
            image = img.get_fdata().astype(np.float32)
            label = lab.get_fdata().astype(np.int64)

            # 截断
            image = np.clip(image, -300, 300)
            for i in range(queue):
                if (np.max(label[:, :, i]) != 0):
                    positive_imgs.append([image[np.newaxis, 120:344, 120:344, i],
                                          label[np.newaxis, 120:344, 120:344, i]])
                else:
                    negative_imgs.append([image[np.newaxis, 120:344, 120:344, i],
                                          label[np.newaxis, 120:344, 120:344, i]])
            print(img_path, "finished!")
        return positive_imgs, negative_imgs
        # imgs.append(positive_imgs)
        # imgs.append(negative_imgs)
        # return imgs

    def __getitem__(self, item):
        # a = random.uniform(0.33,1)
        # b = random.uniform(0,1)
        # if a>b:
        #     image, label = self.images_labels[0][item][0], self.images_labels[0][item][1]
        # else:
        #     image, label = self.images_labels[1][item][0], self.images_labels[1][item][1]
        # # image = paddle.tensor.to_tensor(np.array(image[np.newaxis, :, :], dtype='float32'), place=paddle.CUDAPlace(0))
        # # label = paddle.tensor.to_tensor(np.array(label[np.newaxis, :, :], dtype='int64'), place=paddle.CUDAPlace(0))
        # image = np.concatenate([image, image], axis=0)
        # label = np.concatenate([label, label], axis=0)

        image = self.imgs[item][0]
        label = self.imgs[item][1]
        return image, label

    def __len__(self):
        return len(self.imgs)

    def randomFuse(self):

        positiveNum = len(self.positive)
        negativeNum = len(self.negative)

        negativeNumLimited = math.ceil(positiveNum * 0.6)
        negativeChooseNum = 0

        # 选择负样本
        imgs = []
        index = 0
        while negativeChooseNum < negativeNumLimited:
            if index >= negativeNum:
                index = 0
            else:
                randomNum = np.random.randint(0, 2)
                # print(randomNum, negativeChooseNum, negativeNumLimited)
                if randomNum > 0:
                    imgs.append(self.negative[index])
                    negativeChooseNum += 1
                index += 1
        imgs += self.positive

        self.imgs = imgs
        # return imgs


def train(epoch, model, trainDataLoader, testDataLoader, lossFun, optim, metricFun):
    for epochIndex in range(1, epoch + 1):
        print("当前epoch:", epochIndex)
        model.train()

        avgLoss = 0
        avgDice = 0
        for batchId, (image, label) in enumerate(trainDataLoader()):

            # 可以做相应的数据转换
            predict = model(image)[0]

            # 计算损失
            loss = lossFun(predict, label)
            loss.backward()
            optim.step()
            optim.clear_grad()

            # 评价计算
            diceCoef = metricFun(predict, label)

            if batchId % 20 == 0:
                print("epoch;{}, batchId:{}, loss is:{}, dice is:{}".format(
                    epochIndex, batchId, loss.numpy(), diceCoef)
                )
            avgDice += diceCoef
            avgLoss += loss.numpy()

        avgDice /= batchId + 1
        avgLoss /= batchId + 1
        print("epoch;{}, avg loss is:{}, avg dice is:{}".format(
            epochIndex, avgLoss, avgDice)
        )
        # 模型保存
        # model.save("model")

        # 模型测试
        # test(epochIndex, model, testDataLoader, metricFun)

        # 重新挑选负样本
        if epoch > 1 and epoch % 10 == 0:
            train_dataset.randomFuse()


# class testData():
#     def __init__(self, originPath, labelPath):
#         self.originPaths = []
#         self.labelPaths = []

#         file1_list_onegrade = os.listdir(originPath)
#         file2_list_onegrade = os.listdir(labelPath)
#         for image_name in file1_list_onegrade:
#             self.originPaths.append(os.path.join(image_path, image_name))
#         for label_name in file2_list_onegrade:
#             self.labelPaths.append(os.path.join(label_path, label_name))
#         # self.originPath = originPath
#         # self.labelPatj = labelPath
#         self.imgs = self.dataload(self.originPaths, self.labelPaths)

#     def dataload(self,originPaths, labelPaths):
#         imgs = []
#         for originPath,labelPath in zip(originPaths, labelPaths):

#             img = nib.load(originPath)
#             lab = nib.load(labelPath)
#             width, height, queue = img.dataobj.shape
#             image = img.get_fdata().astype(np.float32)[120:344,120:344, :]
#             label = lab.get_fdata().astype(np.int64)[120:344,120:344, :]
#             image = image[np.newaxis, :, :, :] # 1 224 224 95
#             label = label[np.newaxis, :, :, :]
#             # 转换成 SilceNum Channel Weight Height格式 95 1 224 224
#             # image， label 转换成tensor并放在cuda上
#             image = paddle.tensor.to_tensor(image,place=paddle.CUDAPlace(0))
#             label = paddle.tensor.to_tensor(label,place=paddle.CUDAPlace(0))
#             image = fluid.layers.transpose(image, perm=[3, 0, 1, 2])
#             label = fluid.layers.transpose(label,perm=[3, 0, 1, 2])

#             imgs.append([image, label])
#             print(originPath, "finished！")
#         return imgs


# def test(epochIndex, model, testData,  metric):
#     print("start")
#     avgDice = 0
#     sampelDice = []
#     testBatchSize = 4
#     for index in range(len(testData.imgs)):
#         originData = testData.imgs[index][0] # 95 1 224 224
#         labelData = testData.imgs[index][1]  # 95 1 224 224

#         results = []
#         with paddle.no_grad():
#             for index_i in range(0, len(originData), testBatchSize):

#                 inputData = originData[index_i:index_i+testBatchSize]
#                 # inutData 需要转成tensor 并放到GPU上
#                 inputData = paddle.tensor.to_tensor(inputData,dtype='float32',place=paddle.CUDAPlace(0))
#                 #print("input",inputData.shape)
#                 result = model(inputData)[0]
#                 results.append(result)
#                 #print("finish")
#             # results N batchSize C W H
#             # 拼起来 N*BatchSize C W H
#             # print(len(results))
#             # print(results[0].shape)
#             res=results[0]
#             for i in range(1,len(results)):
#                 res=paddle.concat(x=[res,results[i]], axis=0)
#             # print(len(res))
#             # print(res[0].shape)
#             # 数据类型要一致， 放的位置也要一致
#             sampelDiceCoef = metric(res, labelData)#转成一个数字
#             sampelDice.append(sampelDiceCoef)
#             avgDice += sampelDiceCoef # 等着用来求平均值
#             print(testData.originPaths[index], sampelDiceCoef)
#         print("epoch",epochIndex,"平均Dice为:",avgDice/len(sampelDice))

if __name__ == '__main__':
    # image_path = 'TrainSet01'
    # label_path = 'TrainLabel01'
    image_path = 'Set'
    label_path = 'Label'
    train_dataset = DatasetOurs(image_path, label_path)
    # test_dataset = testData(image_path, label_path)
    loader = DataLoader(train_dataset,
                        batch_size=2,
                        shuffle=True,
                        drop_last=False,
                        num_workers=0)

    # model = UNet(1, 2, [64])
    model = UNetOurs(2)
    # print(model)

    beta1 = paddle.to_tensor([0.9], dtype="float32")
    beta2 = paddle.to_tensor([0.99], dtype="float32")

    optim = paddle.optimizer.Adam(learning_rate=1e-4,
                                  parameters=model.parameters(),
                                  beta1=beta1,
                                  beta2=beta2,
                                  weight_decay=0.01)
    loss = DiceCELoss()  # DiceCELoss()
    metric = DiceMetric

    train(epoch=100, model=model, trainDataLoader=loader,
          testDataLoader=loader, lossFun=loss,
          optim=optim, metricFun=metric)
    # test(epochIndex=100,testData=test_dataset,model=model,metric=metric)



import PIL.Image
import numpy as np
import paddle.io
import paddle.vision.transforms as transforms
import nibabel as nib
from unet import *
import os
import paddle.fluid as fluid
import matplotlib.pyplot as plt
import scipy
from sklearn.model_selection import train_test_split
import imageio
import random


import  paddle
import paddle.fluid as fluid

class Dice_Loss(paddle.nn.Layer):
    """
    1. 继承paddle.nn.Layer
    """
    def __init__(self):
        """
        2. 构造函数根据自己的实际算法需求和使用需求进行参数定义即可
        """
        super(Dice_Loss, self).__init__()
    def forward(self, predict, label):
        """
        3. 实现forward函数，forward在调用时会传递两个参数：input和label
            - predict：单个或批次训练数据经过模型前向计算输出结果
            - label：单个或批次训练数据对应的标签数据
            接口返回值是一个Tensor，根据自定义的逻辑加和或计算均值后的损失
        """
        # 使用Paddle中相关API自定义的计算逻辑
 
        predict = fluid.layers.transpose(predict, perm=[0, 2, 3, 1])
        predict = fluid.layers.reshape(predict, shape=[-1, 2])
        predict = fluid.layers.softmax(predict)
        label = fluid.layers.reshape(label, shape=[-1, 1])
        entropy = fluid.layers.cross_entropy(predict, label)#交叉熵
        dice_cost = fluid.layers.dice_loss(predict, label)#dice_loss
        
        return fluid.layers.reduce_mean(dice_cost + entropy)

class Dataset(paddle.io.Dataset):
    def __init__(self, image_path,label_path,mode='train'):
        self.size = 160, 160
        self.mode = mode

        self.image_paths = []
        self.label_paths = []

        file1_list_onegrade = os.listdir(image_path)
        file2_list_onegrade = os.listdir(label_path)
        for image_name in file1_list_onegrade:
            self.image_paths.append(os.path.join(image_path,image_name))
        for label_name in file2_list_onegrade:
            self.label_paths.append(os.path.join(label_path,label_name))
        self.images, self.labels=self._get_total('')

    def _get_total(self,type='image'):
        imgs=[]
        labs=[]
        positive_label=[]
        negative_label=[]
        positive_image=[]
        negative_image=[]

        for lab_path, img_path in zip(self.label_paths, self.image_paths):
            lab=nib.load(lab_path)
            width, height, queue = lab.dataobj.shape
            label = lab.get_fdata().astype(np.int64)

            img=nib.load(img_path)
            # width, height, queue = img.dataobj.shape
            image=img.get_fdata().astype(np.float32)
            for i in range(queue):
                # imgs.append(image[:,:,i])
                if(np.max(label[:,:,i])!=0):
                    positive_label.append(label[120:344,120:344,i])
                    positive_image.append(image[120:344,120:344,i])
                else:
                    negative_label.append(label[120:344,120:344,i])
                    negative_image.append(image[120:344,120:344,i])
        
        for i in range(len(positive_image)):
            imgs.append(positive_image[i])
            labs.append(positive_label[i])


        for i in range(len(positive_image)):
            imgs.append(np.fliplr(positive_image[i]))
            labs.append(np.fliplr(positive_label[i]))


        for i in range(len(negative_image)):
            imgs.append(negative_image[i])
            labs.append(negative_label[i])

        for i in range(len(positive_image)):
            imgs.append(positive_image[i])
            labs.append(positive_label[i])

        return imgs,labs

    def __getitem__(self, item):
        image=self.images[item]
        label=self.labels[item]
        return np.array(image.reshape(1, 224, 224), dtype='float32'), np.array(label.reshape(1, 224,224), dtype='int64')

    def __len__(self):
        sum=0
        for img in self.images:
            sum=sum+1
        return sum


if __name__ == '__main__':
    image_path='external-libraries/Train_set'
    label_path='external-libraries/Train_Label'
    train_dataset= Dataset(image_path,label_path,mode='train')
    test_dataset= Dataset(image_path,label_path,mode='train') 
    # print(type(total_dataset))
    
    # train_dataset, test_dataset = train_test_split(total_dataset, test_size=0.2)
    # print(type(train_dataset))


    model = paddle.Model(UNet(1, 2, [64]))
    beta1 = paddle.to_tensor([0.9], dtype="float32")
    beta2 = paddle.to_tensor([0.99], dtype="float32")

    optim = paddle.optimizer.Adam(learning_rate=5e-05,
                                parameters=model.parameters(),
                                beta1=beta1,
                                beta2=beta2,
                                weight_decay=0.01)
    # optim = paddle.optimizer.RMSProp(learning_rate=5e-05,
    #                                  rho=0.9,
    #                                  momentum=0.0,
    #                                  epsilon=1e-07,
    #                                  centered=False,
    #                                  parameters=model.parameters())
    # model.prepare(optim, paddle.nn.CrossEntropyLoss(axis=1))
    model.prepare(optim, loss=Dice_Loss())#paddle.nn.CrossEntropyLoss(axis=1))

    model.fit(train_dataset, epochs=10, batch_size=4, verbose=1)

    # model.save('external-libraries/param')

    predict_dataset = test_dataset
    predict_results = model.predict(predict_dataset)

    params_info = model.summary()
    print(params_info)

    print(np.array(predict_results).shape)
    test_data=[]
    for i in range(858):
        test_data.append(np.argmax(predict_results[0][i][0],axis=0))
    for i in range(858):
        imageio.imwrite(os.path.join('test/', '{}.png'.format(i)), test_data[i])



    


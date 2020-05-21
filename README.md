# DCNN_on_SHM
Deep Learning Based Structural Health Monitoring (Damage Detection)
## 1. 基本信息
Python 3.5.3  
Tensorflow 1.1.0  
Numpy 1.12.1  
## 2. 数据类型说明
> Type A: (batch, height, weight) : class label  
> Type B: (batch, height, weight, depth) : feature  
> Type C: (batch, height, weight, depth = classes) : one-hot label, 例如: [0,0,1,0]  
> Type D: (batch, height, weight, depth = 1): class label  
因此, 对于一个变量按照如下格式进行注释:  
variable_name: Type A: int32: (0,class): size  
## 3. 网络结构(STRUC)
### 3.1 FCN_VGG16
训练模式: train
测试模式: test
### 3.2 DeepLabV3_ResNet50
训练模式: train
测试模式: test
### 3.3 U_Net
训练模式: train
测试模式: test
### 3.4 DeepLabV3_ResNeXt50
训练模式: train
测试模式: test
### 3.5 Proposed
训练模式: train/MCD_train  
在MCD模式中, G为DeepLab V3(编码器+ASPP)部分, F1, F2结构相同, 为解码器部分.
测试模式: test/MCD_test
### 3.6 FPHB
训练模式: train
测试模式: test
### 3.7 AdaptSegNet
该模型来自于:Learning to Adapt Structured Output Space for Semantic Segmentation
属于典型的DCNN框架域适应语义分割
## 4. 模式(MOD)
### 4.1 train
### 4.2 test
### 4.3 SS_pre_train
Semi-supervised pre-train
### 4.4 SS_after_pre_train
### 4.5 MCD_train
A step:
### 4.6 MCD_test
## 5. 数据集
### 5.1 CRACK500toSURFACE
train_data1.tfrecords*7:  
source_anno*3000  
source_img*3000  
target_img*3000  
valid_data.tfrecords*1:  
source_anno*3000  
source_img*3000  
target_anno*3000  
target_img*3000  
### 5.2 CRACK500toTUNNEL
### 5.3 
### 5.4 
## 6. 变量
## 8. 参考代码

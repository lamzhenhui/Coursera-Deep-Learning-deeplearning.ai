# import tensorflow as tf
from keras.api.applications import ResNet50

# 加载预训练的 ResNet50 模型（包括权重）
model = ResNet50(weights='imagenet')

# 保存模型为 .h5 文件
model.save('Resnet50.h5')

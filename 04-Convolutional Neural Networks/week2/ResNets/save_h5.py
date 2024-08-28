# # import tensorflow as tf
# from keras.api.applications import ResNet50

# # 加载预训练的 ResNet50 模型（包括权重）
# # model = ResNet50(weights='imagenet')
# model = ResNet50(input_shape=(64, 64, 3), classes=6
#                  )

# # 保存模型为 .h5 文件
# model.save('Resnet50_1.h5')

from keras.api.applications import ResNet50
from keras.api.layers import Dense, Flatten, Input
from keras.api.models import Model

# 使用新的输入形状定义模型，并排除预训练模型的顶层
base_model = ResNet50(weights='imagenet',
                      include_top=False, input_shape=(64, 64, 3))

# 添加自定义的顶层来适应6个类别
x = base_model.output
x = Flatten()(x)  # 展平层
x = Dense(6, activation='softmax')(x)  # 根据您的需求定义全连接层

# 创建一个新的模型
model = Model(inputs=base_model.input, outputs=x)

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# 保存模型为 .h5 文件
model.save('ResNet50_custom.h5')

print(model.summary())

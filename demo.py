# import  os
import numpy as np
# print(os.listdir(os.path.dirname(os.path.abspath(__file__))))
# import pathlib
# data_root="../../data/kaggle_dog/train_valid_test_tiny"
# train_data_root = pathlib.Path(data_root+"/train")
# # path = '../../data/kaggle_dog/train_valid_test_tiny/train'
# print(train_data_root, type(train_data_root))
# # print(train_data_root.glob('*/'))
# label_names = sorted(item.name for item in train_data_root.glob('*/') if item.is_dir())
# # print(label_names)
# train_all_image_paths = [str(path) for path in list(train_data_root.glob('*/*'))]

# label_to_index = dict((name, index) for index, name in enumerate(label_names)) # lable to index relationship
# print(label_to_index)

# 梯度下降案例
# import tensorflow as tf
# opt = tf.keras.optimizers.SGD(learning_rate=0.1)
# var = tf.Variable(1.0)
# print(var)
# loss = lambda: (var ** 2)/2.0         # d(loss)/d(var1) = var1
# step_count = opt.minimize(loss, [var]).numpy()
# print(step_count)
# # Step is `-learning_rate*grad`
# var.numpy()
# print(var.numpy())

# demo 2
# opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
# var = tf.Variable(1.0)
# val0 = var.value()
# loss = lambda: (var ** 2)/2.0         # d(loss)/d(var1) = var1
# # First step is `-learning_rate*grad`
# step_count = opt.minimize(loss, [var]).numpy()
# val1 = var.value()
# print(val0 , val1)
# (val0 - val1).numpy()


def main(print_type=''):
    import tensorflow as tf
    import keras
    import numpy as np
    from keras.api.layers import Activation, Dense
    from keras.api.models import Sequential
    if print_type == '@keras_dot2':
        x = np.arange(10).reshape(1, 5, 2)
        y = np.arange(10, 20).reshape(1, 2, 5)
        print(
            x, x.shape, '\n',
            y, y.shape, '\n',
            keras.layers.Dot(axes=(2, 1))([x, y]))
        print('///')
        # Usage in a Keras model:
        x = keras.layers.Dense(5)(np.arange(10).reshape(1, 2, 5))
        y = keras.layers.Dense(2)(np.arange(10, 20).reshape(1, 5, 2))
        print(
            x, x.shape, '\n',
            y, y.shape, '\n',
            keras.layers.Dot(axes=(1, 2))([x, y])
        )
        print('///')
    if print_type == '@keras_activation':
        layer = keras.layers.Activation('relu')
        input_data = [-3.0, -1.0, 0.0, 2.0]
        input_data = np.array([input_data])
        print(layer(input_data))
        # raise Exception
        # model = Sequential()
        # model.add()
        import tensorflow as tf
        # from tensorflow.keras.models import Sequential
        ########################
        ########################
        ########################
        # 创建一个简单的模型
        input_data = [-3.0, -1.0, 0.0, 2.0]
        input_data = np.array([input_data])
        model = Sequential([
            # Dense(4, input_shape=(4,), use_bias=False),
            Activation('relu')
        ])

        # 设置权重（可选）
        # input_data2 = [
        #     [1.0, 1.0, 1.0, 1.0]
        # ]
        # model.set_weights(input_data2)  # 示例权重
        print("Model weights after setting:", model.get_weights())

        # 输入数据

        # 进行预测
        output = model.predict([input_data])
        print(output)
    # [0.0, 0.0, 0.0, 2.0]
    if print_type == '@keras_dense':
        from keras.src.layers import Dense
        from keras.src.models import Sequential
        from keras.src.layers import Dense

        # 创建一个简单的模型
        model = Sequential()
        model.add(Dense(32, input_dim=50, activation='relu', name='dense_layer'))
        # model.add(Dense(32, activation='relu'))

        # 打印模型的结构概览
        model.summary()

        # 或者直接打印特定层的信息
        # print(model.get_layer(name='dense_layer').get_config())
    if print_type == '@keras_dot':
        x = np.arange(6).reshape(1, 2, 3)
        y = np.arange(6).reshape(1, 2, 3)
        # axes(分别带表两个向量分别用第几轴进行点击,会影响点积后的维度)
        ret = keras.layers.Dot(axes=2)([x, y])
        """
[[[0 1]
  [2 3]
  [4 5]]] [[[0 1 2]
  [3 4 5]]]
(1, 3, 2) (1, 2, 3)
tf.Tensor(
[[[10 28]
  [13 40]]], shape=(1, 2, 2), dtype=int64)
(1, 2, 2)
  """
        print(x, y)
        print(x.shape, y.shape)
        print(ret)
        print(ret.shape)
    if print_type == '@keras_concat2':
        import numpy as np
        from keras.layers import concatenate
        from keras.models import Model
        from keras.layers import Input

        # 创建两个形状相同的张量
        a = Input(shape=(3, 4))
        b = Input(shape=(3, 4))

        # 沿着最后一维拼接
        c = concatenate([a, b], axis=-1)
        print(c.shape)

        # 输出的 c 将会有 (3, 8) 的形状
    if print_type == '@keras_concat':
        # mean that  axis :1  轴,  决定了从哪个轴去做concat的维度
        x = np.arange(20).reshape(2, 2, 5)

        y = np.arange(20).reshape(2, 2, 5)
        print(
            x, x.shape, '\n',
            y, y.shape, '\n',
            keras.layers.Concatenate(axis=1)([x, y]))
        """
# keras.layers.Concatenate(axis=0)([x, y]))

# keras.layers.Concatenate(axis=1)([x, y]))
[[[ 0  1  2  3  4]
  [ 5  6  7  8  9]]

 [[10 11 12 13 14]
  [15 16 17 18 19]]] (2, 2, 5)
 [[[ 0  1  2  3  4]
  [ 5  6  7  8  9]]

 [[10 11 12 13 14]
  [15 16 17 18 19]]] (2, 2, 5)
 tf.Tensor(
[[[ 0  1  2  3  4]
  [ 5  6  7  8  9]]

 [[10 11 12 13 14]
  [15 16 17 18 19]]

 [[ 0  1  2  3  4]
  [ 5  6  7  8  9]]

 [[10 11 12 13 14]
   [15 16 17 18 19]]], shape=(4, 2, 5), dtype=int64)
 ##############
        [[[ 0  1  2  3  4]
  [ 5  6  7  8  9]]

 [[10 11 12 13 14]
  [15 16 17 18 19]]] (2, 2, 5)
 [[[ 0  1  2  3  4]
  [ 5  6  7  8  9]]

 [[10 11 12 13 14]
  [15 16 17 18 19]]] (2, 2, 5)
 tf.Tensor(
[[[ 0  1  2  3  4]
  [ 5  6  7  8  9]
  [ 0  1  2  3  4]
  [ 5  6  7  8  9]]

 [[10 11 12 13 14]
  [15 16 17 18 19]
  [10 11 12 13 14]
  [15 16 17 18 19]]], shape=(2, 4, 5), dtype=int64)

######   keras.layers.Concatenate(axis=-1)([x, y]))

  [[[ 0  1  2  3  4]
  [ 5  6  7  8  9]]

 [[10 11 12 13 14]
  [15 16 17 18 19]]] (2, 2, 5)
 [[[ 0  1  2  3]
  [ 4  5  6  7]]

 [[ 8  9 10 11]
  [12 13 14 15]]] (2, 2, 4)
 tf.Tensor(
[[[ 0  1  2  3  4  0  1  2  3]
  [ 5  6  7  8  9  4  5  6  7]]

 [[10 11 12 13 14  8  9 10 11]
  [15 16 17 18 19 12 13 14 15]]], shape=(2, 2, 9), dtype=int64)

  """
    if print_type == '@keras_backend':
        # import tensorflow as tf
        import keras.api.backend as K
        print(K, 'K')
        from tensorflow.python.keras import backend as K
        print(K, 'K')


# else:

    # import keras.backend as K
    if print_type == '@load_model':
        from keras.api.models import load_model
        from keras.api.layers import Dense, Flatten, Input
        # import tensorflow as tf
        # tf.compat.v1.disable_v2_behavior()
        # from tf.compat.v1.keras.models import load_model
        # model = load_model('./models/tr_model.h5',
        # model = load_model(
        #     '/Users/meta/lam/deep/Coursera-Deep-Learning-deeplearning.ai/05-Sequence Models/week3/Trigger word detection/models/tr_model.h5')
        # model = load_model(
        #     '/Users/meta/lam/deep/Coursera-Deep-Learning-deeplearning.ai/04-Convolutional Neural Networks/week2/ResNets/ResNet50_custom.h5')
        model = load_model('/Users/meta/lam/deep/Coursera-Deep-Learning-deeplearning.ai/05-Sequence Models/week3/Trigger word detection/models/tr_model.h5',
                           custom_objects={'Dense': Dense})
    if print_type == 'layers':
        from tensorflow.keras.layers import Input

        print('>>>')
        # 创建输入层
        try:

            input_layer = Input(shape=(224, 224, 3))

            # 获取输入层的形状
            input_shape = input_layer.shape

            # 打印输入层的形状
            print("Input Layer Shape:", input_shape)

            # 打印输入层的详细信息
            print("Input Layer Details:")
            print(input_layer)
            """
            Tensor("input_1:0", shape=(None, 224, 224, 3), dtype=float32)
            输入层的名称和索引
            """
        except Exception as e:
            print(str(e))
    elif print_type == 'frobenius':
        import numpy as np

        # 创建一个矩阵
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6]])
        #    [7, 8, 9]])

        # 计算 Frobenius 范数
        frobenius_norm = np.linalg.norm(matrix, 'fro')

        print("矩阵:")
        print(matrix)
        print("\nFrobenius 范数:")
        print(frobenius_norm)

    elif print_type == '矩阵转置':
        import numpy as np
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6]])
        print(matrix)
        print(matrix.T)
        print(np.dot(matrix.T, matrix))
        print(np.dot(matrix, matrix.T))
        m1 = np.array([[1, 2, 3],
                       [4, 5, 6]])
        m2 = np.array([
            [7, 8], [9, 10], [11, 12]
        ])
        ret = np.dot(m1, m2)
        print(ret)
        print('>>>')
        # import numpy as np

        # 创建一个矩阵
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])

        # 计算矩阵的转置与自身相乘的结果
        result1 = np.dot(matrix.T, matrix)
        result2 = np.dot(matrix, matrix.T)

        # 求迹
        trace1 = np.trace(result1)
        trace2 = np.trace(result2)

        # 输出结果
        print("矩阵:")
        print(matrix)
        print("\n矩阵转置与自身相乘的结果:")
        print(result1)
        print("\n迹 (trace):")
        print(trace1)

        print("\n矩阵与自身转置相乘的结果:")
        print(result2)
        print("\n迹 (trace):")
        print(trace2)

    elif print_type == 'initializers':
        print('initializers')
        import tensorflow as tf
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        print(initializer)

    elif print_type == 'other':
        print('from keras.utils.data_utils import get_file')
        import keras
        import tensorflow as tf
        # import keras.src.utils.dataset_utils import
        # import keras.src.utils.
        from tensorflow.python.keras.utils.data_utils import get_file
        # keras.utils.get_file()

        # from keras.utils.data_utils import get_file
        print()

        keras.utils
        tf.compat.v1
    elif print_type == 'plot_model':
        print('plot_model')
        from tensorflow.python.keras.utils.vis_utils import model_to_dot
        from tensorflow.python.keras.utils.vis_utils import plot_model
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Subtract

        # 定义输入层
        input_1 = Input(shape=(224, 224, 3))
        input_2 = Input(shape=(224, 224, 3))

        # 定义减法层
        output = Subtract()([input_1, input_2])

        import keras
        # 创建模型
        model = keras.Model(inputs=[input_1, input_2], outputs=output)
        # import pydot
        # pydot.Dot.create(pydot.Dot())
        # import tensorflow
        # from tensorflow.python.keras.utils.vis_utils import plot_model
        keras.utils.plot_model(
            model, to_file='model.png', show_shapes=True, show_dtype=True,
            show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96
        )

        # plot_model(model, to_file='model.png')
        # inputs = Input(shape=(224, 224, 3))
        # outputs = Input(shape=(224, 224, 3))
        # model = keras.Model(inputs=inputs, outputs=outputs)

        # dot_img_file = '/tmp/model_1.png'
        # keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

    elif print_type == 'layers_model':
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Subtract

        # 定义输入层
        input_1 = Input(shape=(224, 224, 3))
        input_2 = Input(shape=(224, 224, 3))

        # 定义减法层
        output = Subtract()([input_1, input_2])

        # 创建模型
        model = Model(inputs=[input_1, input_2], outputs=output)

        # 打印模型结构
        model.summary()
    else:
        print('hello world2')


if __name__ == '__main__':
    import numpy as np
    print_type = 'layers_model'
    print_type = 'plot_model'
    print_type = 'initializers'
    print_type = '矩阵转置'
    print_type = 'frobenius'
    print_type = '@load_model'
    print_type = '@keras_backend'
    print_type = '@fix_pdf'
    print_type = '@keras_concat'
    # print_type = '@keras_dense'
    # print_type = '@keras_activation'
    # print_type = '@keras_dot'
    print_type = '@keras_dot2'
    # print_type = '@keras_concat2'
    # print_type = 'other'
    # print_type = 'layers'
    main(print_type)

    # wget https://github.com/acarafat/coursera-deep-learning-specialization/blob/2830f55679808621193e017baace3733c79142cc/C5%20-%20Sequence%20Models/Week%203/Trigger%20word%20detection/models/tr_model.h5.zip
    # curl -O https://github.com/acarafat/coursera-deep-learning-specialization/blob/2830f55679808621193e017baace3733c79142cc/C5%20-%20Sequence%20Models/Week%203/Trigger%20word%20detection/models/tr_model.h5.zip
# pip install -e .

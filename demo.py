# import  os

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
    print_type = 'layers_model'
    print_type = 'plot_model'
    print_type = 'initializers'
    print_type = '矩阵转置'
    print_type = 'frobenius'
    # print_type = 'other'
    # print_type = 'layers'
    main(print_type)

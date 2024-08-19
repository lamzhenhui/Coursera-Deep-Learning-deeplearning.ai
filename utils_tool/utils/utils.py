
def keras_version_check_v2():
    """
    检查keras版本是否大于2.0
    """
    from keras import __version__ as keras_version
    from packaging import version

    # 获取Keras版本
    # print("Keras Version:", keras_version)

    # 检查Keras版本是否大于2

    if version.parse(keras_version) > version.parse('2'):
        # print("Keras版本大于2")
        return True
        # 使用Keras 2.x的API
    else:
        # print("Keras版本不大于2")
        return False


def version_check(pack_name='', vesion_lg_parse=''):
    from scipy import __version__ as pac_version
    from packaging import version

    # 获取Keras版本
    # print("Keras Version:", keras_version)

    # 检查Keras版本是否大于2

    if version.parse(pac_version) > version.parse(vesion_lg_parse):
        # print("Keras版本大于2")
        return True
        # 使用Keras 2.x的API
    else:
        # print("Keras版本不大于2")
        return False


if __name__ == '__main__':
    # keras_version_check_v2()
    print(version_check(pack_name='scipy', vesion_lg_parse='1.14'))
    print(1)

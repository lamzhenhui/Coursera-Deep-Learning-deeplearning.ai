from setuptools import setup, find_packages

setup(
    name='utils_tool',
    version='0.1',
    packages=find_packages(),  # 自动查找并包含所有包
    install_requires=[],  # 你可以在这里添加依赖包
    entry_points={
        'console_scripts': [
            # 'my_package_main = my_package.main:main',  # 可选，定义命令行脚本
        ],
    },
)

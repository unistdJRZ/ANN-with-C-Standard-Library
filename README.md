# ANN-with-C-Standard-Library
用C++标准库实现的一个鸡掰前馈神经网络，依赖个别OpenCV函数辅助，用于神经网络的学习目的，本机运行在VS15上
随代码给出了三个784-24-24-10分别由100，500，1000个样本训练得到的神经网络的权值文件，可直接载入程序中运行
本项目中用到的MNIST数据集请自行在MNIST官网：http://yann.lecun.com/exdb/mnist/ 进行下载
本项目中依赖OpenCV的：
                      1.随机数生成RNG类（Random Number Generator）
                      2.XML文件解析（FileNode类）
函数的具体文件请查阅OpenCV官方文档.
项目兼容OpenCV 3.3及其以上版本，请在运行前配置OpenCV

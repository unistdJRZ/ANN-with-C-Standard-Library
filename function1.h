#ifndef FUNC_H
#define FUNC_H
#include "layer1.h"
#include<ctime>
#include<cmath>
#include<iostream>
#include<opencv2\core.hpp>
#include<opencv2\opencv.hpp>
using namespace cv;
using namespace std;
const double inf = 1e9;
double sigmoid(double x);
double diff_sigmoid(double x);
double ReLU(double x);
double diff_ReLU(double x);
void init(vector<layer*>& net);//对网络进行初始化
void setzero(mat& m);//将一个矩阵设为0
int rev(int n);//小端int转换
void read_img(string file, vector<mat>& pic, int used);//读取mnist样本文件
void read_label(string file, vector<unsigned char>& type, int used);//读取mnist标签文件
void convert_to_mat(vector<unsigned char>& label, vector<mat>& ans);//将读到的label转成输出矩阵
mat mul(mat& a, mat& b);//矩阵乘
mat add(mat& a, mat& b,double flag);//矩阵和
void update_diff(layer* now, layer* last);//更新now->diff
void update_weight(layer* now, layer* next,double rate);//更新now->delta_weight,now->delta_bias
void BP(vector<layer*>& net,mat& loss,double rate);//反向传播
mat FP(vector<layer*>& net);//正向传播
mat countloss(mat& ans, mat& sample);//计算损失(平方前)
void train(vector<layer*>& net, int n, double rate, double wrong);
void trainonce(vector<layer*>& net, vector<mat>& minibatch, vector<mat>& label, double rate);//进行一次训练
void update(vector<layer*>& net);//更新权值和偏置
void save(vector<layer*>& net, string filename);//保存网络,保存到filename.xml
void load(vector<layer*>& net,string filename);//从filename载入网络
bool judge(mat& ans, int label);//判断分类正误
//debug
void outmat(mat& out);//输出矩阵
double test(vector<layer*>& net,vector<mat>& img,vector<unsigned char>& label);//进行一次正传输出累积误差
void net_read(vector<layer*>& net, mat& img);//读取图片到网络
void pow_ele(mat& m);//计算矩阵每个值的平方
#endif
// MNIST_NN.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include<iostream>
#include<fstream>
#include<cstring>
#define USED_SAMPLE 1000
using namespace std;
const int numlayer = 4;
int main()
{
	ios::sync_with_stdio(false);
	cin.tie(0);
	string fileimage = "train-images.idx3-ubyte";
	string filelabel = "train-labels.idx1-ubyte";
	string savefile = "784-24-24-10_1000SAMPLES";
	vector<layer*> Net(numlayer);
	layer NN[numlayer];
	vector<mat> data;
	vector<mat> label_mat;
	vector<unsigned char> label;
	mat ans;
	//创建一个784-24-24-10的神经网络
	NN[0] = layer(784, 0);
	NN[1] = layer(24, 784);
	NN[2] = layer(24, 24);
	NN[3] = layer(10, 24);
	for (int i = 0; i < numlayer; i++) Net[i] = &NN[i];
	init(Net);
	read_img(fileimage, data,USED_SAMPLE);
	read_label(filelabel,label,USED_SAMPLE);
	convert_to_mat(label, label_mat);
	load(Net, savefile);
	int n = 0;
	double errorrate = 100;
	while (n < 1000 && errorrate > 5) {
		errorrate = test(Net, data, label);
		trainonce(Net, data, label_mat, 0.001);
		cout << "Error rate:" << errorrate << "%" << endl;
		if (n++ % 10 == 0) save(Net, savefile);
	}
	cout <<"trained: "<< test(Net, data, label) << endl;
	system("pause");
    return 0;
}


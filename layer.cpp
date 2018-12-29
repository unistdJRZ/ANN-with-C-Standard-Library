#include"layer1.h"
#include"stdafx.h"
using namespace std;
layer::layer(){};
layer::~layer(){};
layer::layer(int now,int last){
    weight=mat(now,cell(last));
    delta_weight=mat(now,cell(last));
    val=mat(now,cell(1));
    z=mat(now,cell(1));
    diff=mat(now,cell(1));
    bias=mat(now,cell(1));
    delta_bias=mat(now,cell(1));
}
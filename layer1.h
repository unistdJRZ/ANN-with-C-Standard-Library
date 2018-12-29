#ifndef NN_H_
#define NN_H_
#include<vector>
#include<algorithm>
using namespace std;
typedef vector<double> cell;
typedef vector<cell> mat;
/*一层神经网络*/
class layer {
public:
	mat weight;//权重(mxn)
	mat val;//值(mx1)
	mat bias;//偏置(mx1)
	mat z;//z(x)
	mat diff;//diff Ek diff aj
	mat delta_weight;//调整权重
	mat delta_bias;//调整偏置
//public:
	layer();
	layer(int now, int last);//这层网络为n个节点
	~layer();
};
#endif
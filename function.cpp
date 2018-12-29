#include"stdafx.h"
#include"function1.h"
double sigmoid(double x){
    return 1/(1+exp(-x));
}

double diff_sigmoid(double x){//sigmoid求导
    return sigmoid(x)*(1-sigmoid(x));
}

double ReLU(double x){
    return max(x,0.0);
}

double diff_ReLU(double x){//ReLU求导
    return x>0?1:0;
}

void update_diff(layer* now,layer* last){//aL->diff更新依赖a(L+1)diff和a(L+1)的w
	for (int i = 0; i < now->diff.size(); i++) now->diff[i][0] = 0.0;//预置为0
	for (int i = 0; i < now->diff.size(); i++) {//更新now每一个节点的diff
		for (int j = 0; j < last->val.size(); j++) {//now每一个节点的diff需对last求和
			now->diff[i][0] += last->diff[j][0] * diff_sigmoid(last->z[j][0])*last->weight[j][i];
        }
    }
}

void update_weight(layer* now,layer* next,double rate){//wL更新依赖a(L-1)的值
	for (int i = 0; i < now->weight.size(); i++) {
		for (int j = 0; j < next->val.size(); j++) {//所有边的更新依据仅仅依据now层每个节点的diff来更新
			now->delta_weight[i][j] += rate*now->diff[i][0] * diff_sigmoid(now->z[i][0])*next->val[j][0];//每个节点有num(L-1)条边(每条边更新)
        }
		now->delta_bias[i][0] += rate*now->diff[i][0] * diff_sigmoid(now->z[i][0]);//每个节点只有一个bias(一个节点只更新一次)
    }
}

mat FP(vector<layer*>& net){//对每个样本进行正向传播
    for(int i=1;i<net.size();i++){//FP到n-1层
        layer* now=net[i];
        layer* last=net[i-1];
        now->z=mul(now->weight,last->val);
		for (int k = 0; k < now->val.size(); k++) now->val[k][0] = sigmoid(now->z[k][0]+now->bias[k][0]);
    }
    return net[net.size()-1]->val;
}

void BP(vector<layer*>& net,mat& loss,double rate){//对每个样本进行一次反向传播
	int size=net.size();
    layer* lastlayer=net[size-1];//输出层
    layer* firstlayer=net[0];//输入层
	for (int i = 0; i < lastlayer->val.size(); i++) lastlayer->diff[i][0] = 2 * loss[i][0];//输出层的diff由损失确定
    update_weight(lastlayer,net[size-2],rate);//对输出层进行BP(没有上一层)
	for (int i = size - 2; i > 0; i--) {//对隐层进行BP
		layer* now = net[i];
		layer* last = net[i + 1];
		layer* next = net[i - 1];
		update_diff(now, last);//初始化now->diff(这层和上一层)
		update_weight(now, next, rate);//更新now->weight,now->bias(这层和下一层)
    }//输入层没有连边无需BP
}

void update(vector<layer*>& net){//更新weight和bias
    for(int i=1;i<net.size();i++){
		net[i]->bias = add(net[i]->bias, net[i]->delta_bias,-1);
		setzero(net[i]->delta_bias);
		net[i]->weight = add(net[i]->weight, net[i]->delta_weight,-1);
		setzero(net[i]->delta_weight);
    }
}

mat add(mat& a,mat& b,double flag){//矩阵和
    mat ans(a.size(),cell(a[0].size()));
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[0].size(); j++) {
			ans[i][j] = a[i][j] + b[i][j]*flag;
		}
	}
    return ans;
}

void trainonce(vector<layer*>& net,vector<mat>& minibatch,vector<mat>& label,double rate){
    int nsamp=minibatch.size();
	mat loss(10, cell(1));
    //将图片的数值复制到输入层
    for(int ns=0;ns<nsamp;ns++){
        mat now=minibatch[ns];
		net_read(net, now);
        mat ans=FP(net);//正向传播
		loss = add(ans,label[ns],-1);//计算损失
		BP(net, loss, rate);//反向传播
		//update(net);
    }
	//outmat(net[2]->delta_bias);
    update(net);//更新权值和偏置
}

mat countloss(mat& ans,mat& sample){//计算损失值(平方前)
    int size=ans.size();
    mat loss(10,cell(1));
    for(int i=0;i<size;i++){
		loss[i][0] = ans[i][0] - sample[i][0];
    }
    return loss;
}

mat mul(mat& a,mat &b){//矩阵乘
    mat ans(a.size(),cell(b[0].size()));
    for(int i=0;i<a.size();i++){
        for(int k=0;k<b.size();k++){
            for(int j=0;j<b[0].size();j++){
                ans[i][j]+=a[i][k]*b[k][j];
            }
        }
    }
    return ans;
}

void setzero(mat& m){//初始化矩阵为0
    int h=m.size();
    int w=m[0].size();
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            m[i][j]=0;
        }
    }
}

void outmat(mat& out) {//debug输出矩阵
	for (int i = 0; i < out.size(); i++) {
		for (int j = 0; j < out[0].size(); j++) {
			cout << out[i][j] << " ";
		}
		cout << endl;
	}
	cout << "--------------------------" << endl;
}

double test(vector<layer*>& net,vector<mat>& img,vector<unsigned char>& label) {//进行一次测试计算正确率
	int sum = 0;
	for (int i = 0; i < label.size(); i++) {
		mat imgnow = img[i];
		int labelnow = (int)label[i];
		net_read(net, imgnow);
		mat ans = FP(net);
		if (!judge(ans, labelnow)) sum++;
	}
	return ((double)sum / label.size()) * 100;
}

bool judge(mat& ans, int label) {//判断分类正误
	double maxx = -inf;
	int locans = -1;
	for (int i = 0; i < ans.size(); i++) {
		if (ans[i][0] > maxx) {
			maxx = ans[i][0];
			locans = i;
		}
	}
	return locans == label ? true : false;
}

void net_read(vector<layer*>& net,mat& img) {
	layer* input = net[0];
	int num = 0;
	for (int i = 0; i < img.size(); i++) {
		for (int j = 0; j < img[0].size(); j++) {
			input->val[num++][0] = img[i][j];
		}
	}
}

void pow_ele(mat& m) {
	for (int i = 0; i < m.size(); i++) {
		for (int j = 0; j < m[0].size(); j++) {
			m[i][j] *= m[i][j];
		}
	}
}

void save(vector<layer*>& net, string filename) {
	FileStorage file(filename + ".xml", FileStorage::WRITE);
	file << "activefunc" << "Sigmoid";
	for (int i = 1; i < net.size(); i++) {//从隐层开始保存(输入层没有权重)
		layer* now = net[i];
		string layername = "layer" + to_string(i);
		//记录now大小
		file << layername << "{";
		file << "layerinfo" << "{";
		file << "numnow" << (int)now->weight.size();
		file << "numlast" << (int)now->weight[0].size();
		file << "}";
		//记录now权重
		file << "weight" << "[";
		for (int j = 0; j < now->weight.size(); j++) {
			for (int k = 0; k < now->weight[0].size(); k++) {
				file << now->weight[j][k];
			}
		}
		file << "]";
		//记录now偏置
		file << "bias" << "[";
		for (int j = 0; j < now->bias.size(); j++) {
			file << now->bias[j][0];
		}
		file << "]";
		file << "}";
	}
}

void load(vector<layer*>& net, string filename) {
	FileStorage file(filename+".xml", FileStorage::READ);
	for (int i = 1; i < net.size(); i++) {
		layer* now = net[i];
		string layername = "layer" + to_string(i);
		FileNode layernow = file[layername];
		FileNode layerinfo = layernow["layerinfo"];
		//检查能否载入
		int numnow = (int)layerinfo["numnow"];
		int numlast = (int)layerinfo["numlast"];
		if (numnow != now->weight.size() || numlast != now->weight[0].size()) {
			cout << "unmatched loadfile! load terminate..." << endl;
			return;
		}
		//载入权重
		FileNode layerweight = layernow["weight"];
		FileNodeIterator val = layerweight.begin();
		for (int j = 0; j < numnow; j++) {
			for (int k = 0; k < numlast; k++) {
				now->weight[j][k] = (double)(*val++);
			}
		}
		FileNode layerbias = layernow["bias"];
		val = layerbias.begin();
		for (int j = 0; j < numnow; j++) {
			now->bias[j][0] = (double)(*val++);
		}
	}
}

void init(vector<layer*>& net) {//Nguyen-Widrow 算法初始化权值
	RNG rng;//Multiply With Carry 算法随机数生成器
	for (int i = 0; i < net[0]->weight.size(); i++) {//初始化第一层网络为-1到1的权值
		for (int j = 0; j < net[0]->weight[0].size(); j++) net[0]->weight[i][j] = rng.uniform(-1.0, 1.0);
		net[0]->bias[i][0] = rng.uniform(-1.0, 1.0);
	}
	for (int i = 1; i < net.size(); i++) {//遍历每层网络
		layer* now = net[i];//本层
		layer* last = net[i - 1];//上一层
		double val = 0;//随机数val
		double G = now->val.size() > 2 ? 0.7*pow((double)last->val.size(), 1.0 / now->val.size() - 1) : 1.0;//得到常数G
		for (int j = 0; j < now->val.size(); j++) {//遍历本层
			double s = 0;
			for (int k = 0; k < last->val.size(); k++) {//遍历上一层
				val = rng.uniform(0.0, 1.0) * 2 - 1;//生成0到1的随机数
				now->weight[j][k] = val;//得到初始权值的分子
				s += fabs(val);//计算初始权值的分母
			}
			if (i < net.size()) {
				s = 1.0 / (s - fabs(val));//得到初始权值分母
				for (int k = 0; k < last->val.size(); k++) now->weight[i][j]*=s;//初始化权值
				now->bias[j][0] = G*(-1 + j*2.0 / now->val.size());//初始化偏置
			}
		}
		setzero(now->delta_bias);//初始化delta_bias为0
		setzero(now->delta_weight);//初始化delta_weight为0
	}
}

int rev(int n) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = n & 255;
	ch2 = (n >> 8) & 255;
	ch3 = (n >> 16) & 255;
	ch4 = (n >> 24) & 255;
	return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
void read_label(string file, vector<unsigned char>& type,int used) {//读取mnist标签文件
	ifstream label(file, ios::binary);
	if (label.is_open()) {
		int useless, sum;
		label.read((char*)&useless, sizeof(useless));
		label.read((char*)&sum, sizeof(sum));
		sum = rev(sum);
		cout << "find label " << sum << endl;
		for (int i = 0; i < used; i++) {
			unsigned char t;
			label.read((char*)&t, sizeof(t));
			type.push_back(t);
		}
	}
}
void convert_to_mat(vector<unsigned char>& label, vector<mat>& ans) {//将读到的label转成输出矩阵
	for (int i = 0; i < label.size(); i++) {
		mat tmp(10, cell(1));
		setzero(tmp);
		tmp[label[i]][0] = 1;
		ans.push_back(tmp);
	}
}
void read_img(string file, vector<mat>& pic,int used) {//读取mnist样本文件
	ifstream img(file, ios::binary);
	if (img.is_open()) {
		int useless, sum, rows, cols;
		img.read((char*)&useless, sizeof(useless));
		img.read((char*)&sum, sizeof(sum));
		img.read((char*)&rows, sizeof(rows));
		img.read((char*)&cols, sizeof(cols));
		useless = rev(useless);
		sum = rev(sum);
		rows = rev(rows);
		cols = rev(cols);
		cout << "find picture " << sum << " size " << rows << "x" << cols << " used " << used << " sample" << endl;
		for (int i = 1; i <= used; i++) {
			mat frame(rows, cell(cols));//从ubyte中读入mat中方便载入网络
			for (int j = 0; j < rows; j++) {
				for (int k = 0; k < cols; k++) {
					unsigned char pixel;
					img.read((char*)&pixel, sizeof(pixel));
					frame[j][k] = (double)pixel;
				}
			}
			pic.push_back(frame);
		}
	}
}
#include "caffe/caffe_tools.hpp"
/**	根据层的名字获取其在网络中的Index
*	Net的Blob是指，每个层的输出数据，即Feature Maps
*/
unsigned int get_blob_index(boost::shared_ptr< Net<float> > & net, char *query_blob_name)
{
	std::string str_query(query_blob_name);
	vector< string > const & blob_names = net->blob_names();
	for (unsigned int i = 0; i != blob_names.size(); ++i)
	{
		if (str_query == blob_names[i])
			return i;
	}
	LOG(FATAL) << "Unknown blob name: " << str_query;
}

/*	根据层的名字获取对应层的输出数据 */
std::vector<float> get_blob_data(boost::shared_ptr< Net<float> > & net, char *query_blob_name)
{
	// 读取网络指定Feature层数据
	unsigned int blob_id = get_blob_index(net, query_blob_name);
	boost::shared_ptr<Blob<float> > blob = net->blobs()[blob_id];
	unsigned int num_data = blob->count();
	const float *blob_ptr = (const float *)blob->cpu_data();
	std::vector<float> blob_value(blob_ptr, blob_ptr + num_data);
	return blob_value;
}

/*===========================================================================*\
* Funtion: get_blob_data
*===========================================================================
* DESCRIPTION:
*  根据层的名字获取对应层的输出数据, 通过指定num 和 通道数返回对应通道上的featmap
* Input:
*  	net: 目标网络
*	query_blob_name: 目标网络的层名
*	num: blob的第一个维度
*	channel： blob的第2维度，指定对应的featmap单元
* Return:
*	featmap: 单通道浮点的特征矩阵，其大小为blobs的第3，4个维度
\*===========================================================================*/
cv::Mat get_blob_data(boost::shared_ptr< Net<float> > & net, char *query_blob_name, int num, int channel)
{
	// 读取网络指定Feature层数据
	unsigned int blob_id = get_blob_index(net, query_blob_name);
	boost::shared_ptr<Blob<float> > blob = net->blobs()[blob_id];
	unsigned int num_data = blob->count();
	float *blob_ptr = (float *)blob->cpu_data();
	int n = blob->num();
	int k = blob->channels();
	int h = blob->height();
	int w = blob->width();
	int offset = (num*k + channel)*h*w;
	cv::Mat featmap(h, w, CV_32FC1, blob_ptr + offset);
	return featmap;
}

/*===========================================================================*\
* Funtion: get_blob_data
*===========================================================================
* DESCRIPTION:
*  根据层的名字获取对应层的输出数据, 通过指定num 和 通道数返回对应通道上的featmap
* Input:
*  	net: 目标网络
*	query_blob_name: 目标网络的层名
* Return:
*	featmaps: list of 单通道浮点的特征矩阵，其大小为blobs的第3，4个维度
\*===========================================================================*/
void get_blob_data(boost::shared_ptr< Net<float> > & net, char *query_blob_name, std::vector<cv::Mat> &featmaps)
{
	// 读取网络指定Feature层数据
	unsigned int blob_id = get_blob_index(net, query_blob_name);
	boost::shared_ptr<Blob<float> > blob = net->blobs()[blob_id];
	unsigned int num_data = blob->count();
	float *blob_ptr = (float *)blob->cpu_data();
	int n = blob->num();
	int k = blob->channels();
	int h = blob->height();
	int w = blob->width();
	for (int i = 0; i < n*k; i++)
	{
		int offset = i*h*w;
		cv::Mat featmap(h, w, CV_32FC1, blob_ptr + offset);
		featmaps.push_back(featmap);
	}
}




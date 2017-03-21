#include "caffe/caffe_tools.hpp"
/**	���ݲ�����ֻ�ȡ���������е�Index
*	Net��Blob��ָ��ÿ�����������ݣ���Feature Maps
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

/*	���ݲ�����ֻ�ȡ��Ӧ���������� */
std::vector<float> get_blob_data(boost::shared_ptr< Net<float> > & net, char *query_blob_name)
{
	// ��ȡ����ָ��Feature������
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
*  ���ݲ�����ֻ�ȡ��Ӧ����������, ͨ��ָ��num �� ͨ�������ض�Ӧͨ���ϵ�featmap
* Input:
*  	net: Ŀ������
*	query_blob_name: Ŀ������Ĳ���
*	num: blob�ĵ�һ��ά��
*	channel�� blob�ĵ�2ά�ȣ�ָ����Ӧ��featmap��Ԫ
* Return:
*	featmap: ��ͨ������������������СΪblobs�ĵ�3��4��ά��
\*===========================================================================*/
cv::Mat get_blob_data(boost::shared_ptr< Net<float> > & net, char *query_blob_name, int num, int channel)
{
	// ��ȡ����ָ��Feature������
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
*  ���ݲ�����ֻ�ȡ��Ӧ����������, ͨ��ָ��num �� ͨ�������ض�Ӧͨ���ϵ�featmap
* Input:
*  	net: Ŀ������
*	query_blob_name: Ŀ������Ĳ���
* Return:
*	featmaps: list of ��ͨ������������������СΪblobs�ĵ�3��4��ά��
\*===========================================================================*/
void get_blob_data(boost::shared_ptr< Net<float> > & net, char *query_blob_name, std::vector<cv::Mat> &featmaps)
{
	// ��ȡ����ָ��Feature������
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




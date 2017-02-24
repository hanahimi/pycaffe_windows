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

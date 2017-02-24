#ifndef __MY_CAFFE_TOOLS
#define __MY_CAFFE_TOOLS

#include <caffe/caffe.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <utility>
#include <vector>
#include <string>

using namespace caffe;

unsigned int get_blob_index(boost::shared_ptr< Net<float> > & net, char *query_blob_name);
std::vector<float> get_blob_data(boost::shared_ptr< Net<float> > & net, char *query_blob_name);



#endif
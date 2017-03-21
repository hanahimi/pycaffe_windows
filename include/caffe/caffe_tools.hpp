#ifndef __MY_CAFFE_TOOLS
#define __MY_CAFFE_TOOLS

#include <caffe/caffe.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <utility>
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"

using namespace caffe;

unsigned int get_blob_index(boost::shared_ptr< Net<float> > & net, char *query_blob_name);
std::vector<float> get_blob_data(boost::shared_ptr< Net<float> > & net, char *query_blob_name);
cv::Mat get_blob_data(boost::shared_ptr< Net<float> > & net, char *query_blob_name, int num, int channel);
void get_blob_data(boost::shared_ptr< Net<float> > & net, char *query_blob_name, std::vector<cv::Mat> &featmaps);

#endif
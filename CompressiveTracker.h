/************************************************************************
* File:	CompressiveTracker.h
* Brief: C++ demo for paper: Kaihua Zhang, Lei Zhang, Ming-Hsuan Yang,"Real-Time Compressive Tracking," ECCV 2012.
* Version: 1.0
* Author: Yang Xian
* Email: yang_xian521@163.com
* Date:	2012/08/03
* History:
* Revised by Kaihua Zhang on 14/8/2012
* Email: zhkhua@gmail.com
* Homepage: http://www4.comp.polyu.edu.hk/~cskhzhang/
************************************************************************/
#pragma once

#include <opencv2/imgproc/imgproc.hpp>



#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include "CompressiveTracker.h"
#include "caffe/caffe.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/filler.hpp"
#include <string>
#include <vector>
#include <boost/make_shared.hpp>

#define NetF float 
using namespace caffe;
using std::vector;
using namespace cv;
//---------------------------------------------------
class CompressiveTracker
{

public:
	CompressiveTracker(void);
	~CompressiveTracker(void);
	static vector<vector<double> > w2c;
	static vector<vector<double> > w2c_t; // transpose
	static	boost::shared_ptr< Net<float> > ctNet;
private:
	int featureMinNumRect;
	int featureMaxNumRect;
	int featureNum;
	vector<vector<Rect> > features;
	vector<vector<float> > featuresWeight;
	int rOuterPositive;
	vector<Rect> samplePositiveBox;
	vector<Rect> sampleNegativeBox;
	int rSearchWindow;
	Mat imageIntegral;
	Mat samplePositiveFeatureValue;
	Mat sampleNegativeFeatureValue;
	vector<float> muPositive;
	vector<float> sigmaPositive;
	vector<float> muNegative;
	vector<float> sigmaNegative;
	float learnRate;
	vector<Rect> detectBox;
	Mat detectFeatureValue;
	RNG rng;
	int* feaMap;
	//***********************************************//


	Mat z_pca;
	Mat z_npca;
	vector<string> non_compressed_features;
	vector<string> compressed_features;
	Mat data_mean, cov_matrix, data_matrix;
	Mat pca_variances, pca_basis;
	Mat projection_matrix;
	Mat old_cov_matrix;
	Mat im_patch;
	cv::Point pos;
	cv::Size sz_with_padding;
	int colorMapNumber;
	vector<Mat> multiFeature;
	//***************************************************//
//*******************************//
	Mat cnnim_patch;
	int mapNum;
	int frameN;
	int mapDim;
	string mapName;
	//****************************//



private:
	void HaarFeature(Rect& _objectBox, int _numFeature);
	void sampleRect(Mat& _image, Rect& _objectBox, float _rInner, float _rOuter, int _maxSampleNum, vector<Rect>& _sampleBox);
	void sampleRect(Mat& _image, Rect& _objectBox, float _srw, vector<Rect>& _sampleBox);
	void getMultiFeatureValue(vector<Mat>& cMap, vector<Rect>& _sampleBox, Mat& _sampleFeatureValue);

	void testgetMultiFeatureValue(Mat& zz, vector<Mat>& cMap, vector<Rect>& _sampleBox, Mat& _sampleFeatureValue);

	void getFeatureValue( Mat& _imageIntegral, vector<Rect>& _sampleBox, Mat& _sampleFeatureValue);
	void classifierUpdate(Mat& _sampleFeatureValue, vector<float>& _mu, vector<float>& _sigma, float _learnRate);
	void radioClassifier(vector<float>& _muPos, vector<float>& _sigmaPos, vector<float>& _muNeg, vector<float>& _sigmaNeg,
						Mat& _sampleFeatureValue, float& _radioMax, int& _radioMaxIndex);

	//**********************************************************//
	void get_subwindow(Mat &im, cv::Point pos, cv::Size sz, vector<string> &non_pca_features, vector<string> &pca_features, vector<vector<double> > &w2c, Mat &out_npca, Mat &out_pca);
	vector<Mat> get_feature_map(Mat &im_patch, vector<string> &features);
	vector<Mat> im2c(Mat &im, vector<vector<double> > &w2c, int color);
	template<class T>
	vector<T> get_max(vector<vector<T> > &inp, int dim);
	vector<double> get_vector(int dim, int ind);
	Mat reshape(vector<double> &inp, int rows, int cols);
	vector<double> select_indeces(vector<double> &inp, vector<int> &indeces);
	vector<Mat> feature_projection(Mat &x_npca, Mat& x_pca);
	//**************************************************************//
	void corpImg(Mat &im, cv::Point pos, cv::Size sz);
public:
	void processFrame(Mat& _frame, Rect& _objectBox);
	void init(Mat& _frame, Rect& _objectBox);
};

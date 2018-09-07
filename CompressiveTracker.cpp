#include "CompressiveTracker.h"
#include <math.h>
#include <iostream>
#include <fstream>
using namespace cv;
using namespace std;


int numCompressDim = 10;

//------------------------------------------------
CompressiveTracker::CompressiveTracker(void)
{
	featureMinNumRect = 2;
	featureMaxNumRect = 4;	// number of rectangle from 2 to 4
	featureNum = 50;	// number of all weaker classifiers, i.e,feature pool
	rOuterPositive = 4;	// radical scope of positive samples
	rSearchWindow = 25; // size of search window
	//已经初始化
	muPositive = vector<float>(featureNum, 0.0f);
	muNegative = vector<float>(featureNum, 0.0f);
	sigmaPositive = vector<float>(featureNum, 1.0f);
	sigmaNegative = vector<float>(featureNum, 1.0f);
	learnRate = 0.85f;	// Learning rate parameter

mapNum = 128;
mapName = "conv2_1";
mapDim = 112;
	non_compressed_features = vector<string>({ "gray" });
	compressed_features = vector<string>({ "cn"});
	feaMap = new int[50];
	multiFeature.resize(mapNum);
	frameN = 1;
}




CompressiveTracker::~CompressiveTracker(void)
{
}


void CompressiveTracker::HaarFeature(Rect& _objectBox, int _numFeature)
/*Description: compute Haar features
  Arguments:
  -_objectBox: [x y width height] object rectangle
  -_numFeature: total number of features.The default is 50.
*/
{
	features = vector<vector<Rect> >(_numFeature, vector<Rect>());
	featuresWeight = vector<vector<float> >(_numFeature, vector<float>());
	
	int numRect;
	Rect rectTemp;
	float weightTemp;
      
	for (int i=0; i<_numFeature; i++)
	{
		numRect = cvFloor(rng.uniform((double)featureMinNumRect, (double)featureMaxNumRect));
	    
		//int c = 1;
		for (int j=0; j<numRect; j++)
		{
			
			rectTemp.x = cvFloor(rng.uniform(0.0, (double)(_objectBox.width - 3)));
			rectTemp.y = cvFloor(rng.uniform(0.0, (double)(_objectBox.height - 3)));
			rectTemp.width = cvCeil(rng.uniform(0.0, (double)(_objectBox.width - rectTemp.x - 2)));
			rectTemp.height = cvCeil(rng.uniform(0.0, (double)(_objectBox.height - rectTemp.y - 2)));
			features[i].push_back(rectTemp);

			weightTemp = (float)pow(-1.0, cvFloor(rng.uniform(0.0, 2.0))) / sqrt(float(numRect));
            //weightTemp = (float)pow(-1.0, c);
			
			featuresWeight[i].push_back(weightTemp);
           
		}
	}
}


void CompressiveTracker::sampleRect(Mat& _image, Rect& _objectBox, float _rInner, float _rOuter, int _maxSampleNum, vector<Rect>& _sampleBox)
/* Description: compute the coordinate of positive and negative sample image templates
   Arguments:
   -_image:        processing frame
   -_objectBox:    recent object position 
   -_rInner:       inner sampling radius
   -_rOuter:       Outer sampling radius
   -_maxSampleNum: maximal number of sampled images
   -_sampleBox:    Storing the rectangle coordinates of the sampled images.
*/
{


	int rowsz = _image.rows - _objectBox.height - 1;
	int colsz = _image.cols - _objectBox.width - 1;
	float inradsq = _rInner*_rInner;
	float outradsq = _rOuter*_rOuter;


  	
	int dist;

	int minrow = max(0,(int)_objectBox.y-(int)_rInner);
	int maxrow = min((int)rowsz-1,(int)_objectBox.y+(int)_rInner);
	int mincol = max(0,(int)_objectBox.x-(int)_rInner);
	int maxcol = min((int)colsz-1,(int)_objectBox.x+(int)_rInner);
    
	
	
	int i = 0;

	float prob = ((float)(_maxSampleNum))/(maxrow-minrow+1)/(maxcol-mincol+1);

	int r;
	int c;
    
    _sampleBox.clear();//important
    Rect rec(0,0,0,0);

	for( r=minrow; r<=(int)maxrow; r++ )
		for( c=mincol; c<=(int)maxcol; c++ ){
			dist = (_objectBox.y-r)*(_objectBox.y-r) + (_objectBox.x-c)*(_objectBox.x-c);

			if( rng.uniform(0.,1.)<prob && dist < inradsq && dist >= outradsq ){

                rec.x = c;
				rec.y = r;
				rec.width = _objectBox.width;
				rec.height= _objectBox.height;
				
                _sampleBox.push_back(rec);				
				
				i++;
			}
		}
	
		_sampleBox.resize(i);
		
}






void CompressiveTracker::sampleRect(Mat& _image, Rect& _objectBox, float _srw, vector<Rect>& _sampleBox)
/* Description: Compute the coordinate of samples when detecting the object.*/
{




	int rowsz = _image.rows - _objectBox.height - 1;
	int colsz = _image.cols - _objectBox.width - 1;
	float inradsq = _srw*_srw;	
	

	int dist;

	int minrow = max(0,(int)_objectBox.y-(int)_srw);
	int maxrow = min((int)rowsz-1,(int)_objectBox.y+(int)_srw);
	int mincol = max(0,(int)_objectBox.x-(int)_srw);
	int maxcol = min((int)colsz-1,(int)_objectBox.x+(int)_srw);

	int i = 0;

	int r;
	int c;

	Rect rec(0,0,0,0);
    _sampleBox.clear();//important


	for( r=minrow; r<=(int)maxrow; r++ )
		for( c=mincol; c<=(int)maxcol; c++ ){
			dist = (_objectBox.y-r)*(_objectBox.y-r) + (_objectBox.x-c)*(_objectBox.x-c);

			if( dist < inradsq ){

				rec.x = c;
				rec.y = r;
				rec.width = _objectBox.width;
				rec.height= _objectBox.height;

				_sampleBox.push_back(rec);				

				i++;
			}
		}
	
		_sampleBox.resize(i);



}

// Update the mean and variance of the gaussian classifier
void CompressiveTracker::classifierUpdate(Mat& _sampleFeatureValue, vector<float>& _mu, vector<float>& _sigma, float _learnRate)
{
	Scalar muTemp;
	Scalar sigmaTemp;

	//由每一维的分布求出高斯分布的均值和方差
	for (int i=0; i<featureNum; i++)
	{
		meanStdDev(_sampleFeatureValue.row(i), muTemp, sigmaTemp);
	   
		_sigma[i] = (float)sqrt( _learnRate*_sigma[i]*_sigma[i]	+ (1.0f-_learnRate)*sigmaTemp.val[0]*sigmaTemp.val[0] 
		+ _learnRate*(1.0f-_learnRate)*(_mu[i]-muTemp.val[0])*(_mu[i]-muTemp.val[0]));	// equation 6 in paper

		_mu[i] = _mu[i]*_learnRate + (1.0f-_learnRate)*muTemp.val[0];	// equation 6 in paper
	}
}

// Compute the ratio classifier 
void CompressiveTracker::radioClassifier(vector<float>& _muPos, vector<float>& _sigmaPos, vector<float>& _muNeg, vector<float>& _sigmaNeg,
										 Mat& _sampleFeatureValue, float& _radioMax, int& _radioMaxIndex)
{
	float sumRadio;
	_radioMax = -FLT_MAX;
	_radioMaxIndex = 0;
	float pPos;
	float pNeg;
	int sampleBoxNum = _sampleFeatureValue.cols;

	for (int j=0; j<sampleBoxNum; j++)
	{
		sumRadio = 0.0f;
		for (int i=0; i<featureNum; i++)
		{
			pPos = exp( (_sampleFeatureValue.at<float>(i,j)-_muPos[i])*(_sampleFeatureValue.at<float>(i,j)-_muPos[i]) / -(2.0f*_sigmaPos[i]*_sigmaPos[i]+1e-30) ) / (_sigmaPos[i]+1e-30);
			pNeg = exp( (_sampleFeatureValue.at<float>(i,j)-_muNeg[i])*(_sampleFeatureValue.at<float>(i,j)-_muNeg[i]) / -(2.0f*_sigmaNeg[i]*_sigmaNeg[i]+1e-30) ) / (_sigmaNeg[i]+1e-30);
			sumRadio += log(pPos+1e-30) - log(pNeg+1e-30);	// equation 4
		}
		if (_radioMax < sumRadio)
		{
			_radioMax = sumRadio;
			_radioMaxIndex = j;
		}
	}



}

void CompressiveTracker::corpImg(Mat &im, cv::Point pos, cv::Size sz)
{
	cv::Rect range((int)floor(pos.x - sz.width / 2), (int)floor(pos.y - sz.height / 2), 0, 0);
	range.width = sz.width + range.x;
	range.height = sz.height + range.y;
	//check for out-of-bounds coordinates, and set them to the values at
	//the borders




	int top = 0, bottom = 0, left = 0, right = 0;

	if (range.width > (im.cols - 1))
	{
		right = (range.width - im.cols + 1);
		range.width = im.cols - 1;
	}
	if (range.height > (im.rows - 1))
	{
		bottom = (range.height - im.rows + 1);
		range.height = im.rows - 1;
	}
	if (range.x < 0)
	{
		left = -range.x;
		range.x = 0;
	}
	if (range.y < 0)
	{
		top = -range.y;
		range.y = 0;
	}
	range.width -= (range.x);
	range.height -= (range.y);




	copyMakeBorder(im(range).clone(), cnnim_patch, top, bottom, left, right, BORDER_REPLICATE);



}



void CompressiveTracker::init(Mat& _frame, Rect& _objectBox)
{

	// compute feature template
	HaarFeature(_objectBox, featureNum);


	//***************************************************************//

/**********************************
	pos.x = _objectBox.x + round(_objectBox.width/2);
	pos.y = _objectBox.y + round(_objectBox.height/2);
	sz_with_padding.width = (2 * _objectBox.width);
	sz_with_padding.height = (2 * _objectBox.height);


	get_subwindow(_frame, pos, sz_with_padding, non_compressed_features, compressed_features, w2c, z_npca, z_pca);


	vector<Mat> x = feature_projection(z_npca, z_pca);


/**********************************
	//***************************************************************/




	pos.x = _objectBox.x + round(_objectBox.width/2);
	pos.y = _objectBox.y + round(_objectBox.height/2);
	sz_with_padding.width = (2 * _objectBox.width);
	sz_with_padding.height = (2 * _objectBox.height);




	corpImg(_frame, pos, sz_with_padding);




Mat zte;
cv::resize(cnnim_patch, zte, cv::Size(224, 224));

    std::vector<cv::Mat> dv = { zte };

    std::vector<int> label = { 0 };
    caffe::MemoryDataLayer<NetF> *m_layer_ = (caffe::MemoryDataLayer<NetF> *)ctNet->layers()[0].get();
m_layer_->AddMatVector(dv, label);



    std::vector<caffe::Blob<NetF>*> input_vec;
   ctNet->Forward(input_vec);
    boost::shared_ptr<caffe::Blob<NetF>> featureData = ctNet->blob_by_name(mapName);
    const NetF* pstart = featureData->cpu_data();





	std::cout << featureData->channels() << std::endl;
	std::cout << featureData->width() << std::endl;
	std::cout << featureData->height() << std::endl;






vector<Mat> cnnfeaMaps;
for(int j = 0; j < mapNum; j++)
{
	Mat tmpMap(mapDim, mapDim, CV_64FC1);
    for (int m = 0; m < mapDim; m++)
    {
    	for(int n = 0; n < mapDim; n++)
    	{
    		tmpMap.at<double>(m, n) = *pstart;
        	pstart++;

    	}
	}

cv::resize(tmpMap, tmpMap, cv::Size(cnnim_patch.cols, cnnim_patch.rows));


	cnnfeaMaps.push_back(tmpMap);
}




//************************************************************************//











	// compute sample templates

	Rect tmpBox;
	tmpBox.x = cnnfeaMaps.at(0).cols / 4;
	tmpBox.y = cnnfeaMaps.at(0).rows / 4;
	tmpBox.width = cnnfeaMaps.at(0).cols/2 ;
	tmpBox.height = cnnfeaMaps.at(0).rows/2;

	sampleRect(cnnfeaMaps.at(0), tmpBox, rOuterPositive, 0, 1000000, samplePositiveBox);
	sampleRect(cnnfeaMaps.at(0), tmpBox, rSearchWindow*1.5, rOuterPositive + 4.0, 100, sampleNegativeBox);





	for (int i = 0; i < 50; i++)
	{
		float colorMax1 = -FLT_MAX;
		for (int j = 0; j < mapNum; j++)
		{
			integral(cnnfeaMaps.at(j), imageIntegral, CV_64FC1);




			getFeatureValue(imageIntegral, samplePositiveBox, samplePositiveFeatureValue);
			getFeatureValue(imageIntegral, sampleNegativeBox, sampleNegativeFeatureValue);



			Scalar PmuTemp;
			Scalar PsigmaTemp;
			meanStdDev(samplePositiveFeatureValue.row(i), PmuTemp, PsigmaTemp);

			Scalar NmuTemp;
			Scalar NsigmaTemp;
			meanStdDev(sampleNegativeFeatureValue.row(i), NmuTemp, NsigmaTemp);
			float ttmp = sqrt((PmuTemp.val[0] - NmuTemp.val[0]) *(PmuTemp.val[0] - NmuTemp.val[0])) / (PsigmaTemp.val[0] + NsigmaTemp.val[0]); 
	//		float ttmp = ((PmuTemp.val[0] - NmuTemp.val[0]) *(PmuTemp.val[0] - NmuTemp.val[0])) / (PsigmaTemp.val[0] + NsigmaTemp.val[0]);


			if (ttmp > colorMax1)
			{
				colorMax1 = ttmp;
				feaMap[i] = j;
			}
		}



	}




	for (int i = 0; i < mapNum; i++)
	{
		cv::integral(cnnfeaMaps.at(i), multiFeature[i], CV_64FC1);
	}



	getMultiFeatureValue(multiFeature, samplePositiveBox, samplePositiveFeatureValue);
	getMultiFeatureValue(multiFeature, sampleNegativeBox, sampleNegativeFeatureValue);


	classifierUpdate(samplePositiveFeatureValue, muPositive, sigmaPositive, learnRate);
	classifierUpdate(sampleNegativeFeatureValue, muNegative, sigmaNegative, learnRate);


	multiFeature.clear();

}
void CompressiveTracker::processFrame(Mat& _frame, Rect& _objectBox)
{





	/************************************************************************************
	get_subwindow(_frame, pos, sz_with_padding, non_compressed_features, compressed_features, w2c, z_npca, z_pca);
	vector<Mat> x = feature_projection(z_npca, z_pca);
	//********************************************************************************************/

	//************************************************************************************//


corpImg(_frame, pos, sz_with_padding);
Mat zte;
cv::resize(cnnim_patch, zte, cv::Size(224, 224));

    std::vector<cv::Mat> dv = { zte };

    std::vector<int> label = { 0 };
    caffe::MemoryDataLayer<NetF> *m_layer_ = (caffe::MemoryDataLayer<NetF> *)ctNet->layers()[0].get();
m_layer_->AddMatVector(dv, label);



    std::vector<caffe::Blob<NetF>*> input_vec;
   ctNet->Forward(input_vec);
    boost::shared_ptr<caffe::Blob<NetF>> featureData = ctNet->blob_by_name(mapName);
    const NetF* pstart = featureData->cpu_data();





vector<Mat> cnnfeaMaps;

for(int j = 0; j < mapNum; j++)
{
	Mat tmpMap(mapDim, mapDim, CV_64FC1);
    for (int m = 0; m < mapDim; m++)
    {
    	for(int n = 0; n < mapDim; n++)
    	{
    		tmpMap.at<double>(m, n) = *pstart;
        	pstart++;

    	}
	}

cv::resize(tmpMap, tmpMap,  cv::Size(cnnim_patch.cols, cnnim_patch.rows));
	cnnfeaMaps.push_back(tmpMap);
}

	//********************************************************************************************//


	Rect tmpBox1;
	tmpBox1.x = cnnfeaMaps.at(0).cols / 4;
	tmpBox1.y = cnnfeaMaps.at(0).rows / 4;
	tmpBox1.width = cnnfeaMaps.at(0).cols / 2;
	tmpBox1.height = cnnfeaMaps.at(0).rows / 2;

	sampleRect(cnnfeaMaps.at(0), tmpBox1, rSearchWindow, detectBox);



	for (int k = 0; k < mapNum; k++)
	{
		cv::integral(cnnfeaMaps.at(k), multiFeature[k], CV_64FC1);
	}
	getMultiFeatureValue(multiFeature, detectBox, detectFeatureValue);
	multiFeature.clear();



	int radioMaxIndex;
	float radioMax;
	radioClassifier(muPositive, sigmaPositive, muNegative, sigmaNegative, detectFeatureValue, radioMax, radioMaxIndex);

	Rect tmpBox2;
	tmpBox2 = detectBox[radioMaxIndex];




	// update
	sampleRect(cnnfeaMaps.at(0), tmpBox2, rOuterPositive, 0.0, 1000000, samplePositiveBox);
	sampleRect(cnnfeaMaps.at(0), tmpBox2, rSearchWindow*1.5, rOuterPositive + 4.0, 100, sampleNegativeBox);


	getMultiFeatureValue(multiFeature, samplePositiveBox, samplePositiveFeatureValue);
	getMultiFeatureValue(multiFeature, sampleNegativeBox, sampleNegativeFeatureValue);



	classifierUpdate(samplePositiveFeatureValue, muPositive, sigmaPositive, learnRate);
	classifierUpdate(sampleNegativeFeatureValue, muNegative, sigmaNegative, learnRate);

	Point maxLoc;
	maxLoc.x = (int)(tmpBox2.x + tmpBox2.width / 2);
	maxLoc.y = (int)(tmpBox2.y + tmpBox2.height / 2);
	pos = pos - cv::Point((int)floor(sz_with_padding.width / 2), (int)floor(sz_with_padding.height / 2)) + cv::Point(maxLoc.x + 1, maxLoc.y + 1);
	if (pos.x < 0)
	{
		pos.x = 0;
	}
	if (pos.y < 0)
	{
		pos.y = 0;
	}
	if (pos.x >= _frame.cols)
	{
		pos.x = _frame.cols - 1;
	}
	if (pos.y >= _frame.rows)
	{
		pos.y = _frame.rows;
	}


	_objectBox.x = (int)(pos.x - tmpBox1.width/2);
	_objectBox.y = (int)(pos.y - tmpBox1.height / 2);
	_objectBox.width = tmpBox1.width;
	_objectBox.height = tmpBox1.height;


}


void CompressiveTracker::get_subwindow(Mat &im, cv::Point pos, cv::Size sz, vector<string> &non_pca_features, vector<string> &pca_features, vector<vector<double> > &w2c, Mat &out_npca, Mat &out_pca)
{
	// Extracts the non-PCA and PCA features from image im at position pos and
	// window size sz. The features are given in non_pca_features and
	// pca_features. out_npca is the window of non-PCA features and out_pca is
	// the PCA-features reshaped. 
	// w2c is the Color Names matrix if used.

	cv::Rect range((int)floor(pos.x - sz.width / 2), (int)floor(pos.y - sz.height / 2),
		0, 0);
	range.width = sz.width + range.x;
	range.height = sz.height + range.y;
	//check for out-of-bounds coordinates, and set them to the values at
	//the borders
	int top = 0, bottom = 0, left = 0, right = 0;

	if (range.width > (im.cols - 1))
	{
		right = (range.width - im.cols + 1);
		range.width = im.cols - 1;
	}
	if (range.height > (im.rows - 1))
	{
		bottom = (range.height - im.rows + 1);
		range.height = im.rows - 1;
	}
	if (range.x < 0)
	{
		left = -range.x;
		range.x = 0;
	}
	if (range.y < 0)
	{
		top = -range.y;
		range.y = 0;
	}
	range.width -= (range.x);
	range.height -= (range.y);

	//extract image
	// Mat im_patch(sz.height,sz.width,im.type());
	copyMakeBorder(im(range).clone(), im_patch, top, bottom, left, right, BORDER_REPLICATE);

	// compute non-pca feature map
	if (non_pca_features.size())
	{
		out_npca = get_feature_map(im_patch, non_pca_features)[0];
	}

	// compute pca feature map
	if (pca_features.size())
	{
		vector<Mat> temp_pca = get_feature_map(im_patch, pca_features);
		int total_len = sz.width*sz.height;
		out_pca = Mat::zeros(temp_pca.size(), total_len, CV_64FC1);
		int ind = 0;
		double* data = ((double*)out_pca.data);
		int temp_pca_size = temp_pca.size();
		for (int i = 0; i < temp_pca_size; i++)
		{
			Mat tmp = temp_pca[i].t();
			memcpy(data + i * total_len, tmp.data, total_len * sizeof(double));
		}
		out_pca = out_pca.t();
	}
}





vector<Mat> CompressiveTracker::get_feature_map(Mat &im_patch, vector<string> &features)
{
	// the names of the features that can be used
	vector<string> valid_features({ "gray", "cn" });
	//
	// the dimension of the valid features
	vector<int> feature_levels({ 1, 10 });

	int num_valid_features = valid_features.size();
	//cout << "here" << endl;
	vector<bool> used_features(num_valid_features, false);
	// get the used features
	for (int i = 0; i < num_valid_features; i++)
	{
		if (find(features.begin(), features.end(), valid_features[i]) != features.end())
		{
			used_features[i] = 1;
		}
	}


	// total number of used feature levels
	int num_feature_levels = 0;
	int feature_levels_size = feature_levels.size();
	for (int i = 0; i < feature_levels_size; i++)
	{
		num_feature_levels += feature_levels[i] * used_features[i];
	}
	int level = 0;
	// allocate space (for speed)
	vector<Mat> out(num_feature_levels, Mat::zeros(im_patch.rows, im_patch.cols, CV_64FC1));
	// If grayscale image
	if (im_patch.channels() == 1)
	{
		// Features that are available for grayscale sequances
		// Grayscale values (image intensity)
		im_patch.convertTo(out[0], CV_64F);
		out[0] = (out[0] / 255) - 0.5;
	}
	else
	{

		// Features that are available for color sequances

		// Grayscale values (image intensity)
		if (used_features[0])
		{
			cv::cvtColor(im_patch, out[0], CV_BGR2GRAY);
			out[0].convertTo(out[0], CV_64F);
			out[0] = (out[0] / 255) - 0.5;
		}

		// Color Names
		if (used_features[1])
		{
			// extract color descriptor
			Mat double_patch;
			im_patch.convertTo(double_patch, CV_64FC1);
			out = im2c(double_patch, w2c, -2);
		}
	}

	return out;
}




vector<Mat> CompressiveTracker::im2c(Mat &im, vector<vector<double> > &w2c, int color)
{
	vector<Mat> out;
	// input im should be DOUBLE !
	// color=0 is color names out
	// color=-1 is colored image with color names out
	// color=1-11 is prob of colorname=color out;
	// color=-1 return probabilities
	// order of color names: black ,   blue   , brown       , grey       , green   , orange   , pink     , purple  , red     , white    , yellow
	double color_values[][3] = { { 0, 0, 0 }, { 0, 0, 1 }, { .5, .4, .25 }, { .5, .5, .5 }, { 0, 1, 0 }, { 1, .8, 0 }, { 1, .5, 1 }
	, { 1, 0, 1 }, { 1, 0, 0 }, { 1, 1, 1 }, { 1, 1, 0 } };

	vector<Mat> im_split;
	cv::split(im, im_split);
	Mat RR = im_split[2];
	Mat GG = im_split[1];
	Mat BB = im_split[0];

	double*  RRdata = ((double*)RR.data), *GGdata = ((double*)GG.data), *BBdata = ((double*)BB.data);
	int w = RR.cols;
	int h = RR.rows;
	vector<int> index_im(w * h, 0);
	int l = index_im.size();

	for (int i = 0; i < l; i++)
	{
		//int j = (i / w) + (i//w) * h;
		// I don't need +1 in the formula because the indeces are zero based here
		index_im[i] = (int)(floor(RRdata[i] / 8) + 32 * floor(GGdata[i] / 8) + 32 * 32 * floor(BBdata[i] / 8));
	}

	if (color == 0)
	{
		vector<double> w2cM = get_max(w2c, 2);
		vector<double> selected = select_indeces(w2cM, index_im);
		out.push_back(reshape(selected, im.rows, im.cols));
	}

	if (color > 0 && color < 12)
	{
		vector<double> w2cM = get_vector(2, color - 1);
		vector<double> selected = select_indeces(w2cM, index_im);
		out.push_back(reshape(selected, im.rows, im.cols));
	}

	if (color == -1)
	{
		out.push_back(im);
		vector<double> w2cM = get_max(w2c, 2);
		vector<double> temp = select_indeces(w2cM, index_im);
		Mat out2 = reshape(temp, im.rows, im.cols);
	}

	if (color == -2)
	{
		for (int i = 0; i < 10; i++)
		{
			vector<double> vec = get_vector(2, i);
			vector<double> selected = select_indeces(vec, index_im);
			Mat temp = reshape(selected, im.rows, im.cols);
			out.push_back(temp);
		}
	}

	return out;
}

template<class T>
vector<T> CompressiveTracker::get_max(vector<vector<T> > &inp, int dim)
{
	// dim = 1 max row, 2 max column
	vector<T> ret;
	if (dim == 1)
	{
		int inp0_size = inp[0].size();
		for (int j = 0; j < inp0_size; j++)
		{
			ret.push_back(inp[0][j]);
			int inp_size = inp.size();
			for (int i = 1; i < inp_size; i++)
			{
				if (inp[i][j] > ret[j])
				{
					ret[j] = inp[i][j];
				}
			}
		}
	}
	else if (dim == 2)
	{
		int inp_size = inp.size();
		for (int i = 0; i < inp_size; i++)
		{
			ret.push_back(*max_element(inp[i].begin(), inp[i].end()));
		}
	}
	return ret;
}
vector<double> CompressiveTracker::get_vector(int dim, int ind)
{
	// dim = 1  row, 2  column
	if (dim == 2)
	{
		return w2c_t[ind];
	}
	else// if (dim == 1)
	{
		return w2c[ind];
	}
}
Mat CompressiveTracker::reshape(vector<double> &inp, int rows, int cols)
{
	Mat result(rows, cols, CV_64FC1);
	double* data = ((double*)result.data);



	memcpy(data, ((double*)(&inp[0])), rows*cols*sizeof(double));
	return result;
}
vector<double> CompressiveTracker::select_indeces(vector<double> &inp, vector<int> &indeces)
{
	int sz = std::min(inp.size(), indeces.size());
	vector<double> res(sz, 0);
	for (int i = 0; i < sz; i++)
	{
		res[i] = inp[indeces[i]];
	}
	return res;
}


vector<Mat> CompressiveTracker::feature_projection(Mat &x_npca, Mat& x_pca)
{
	// Calculates the compressed feature map by mapping the PCA features with
	// the projection matrix and concatinates this with the non-PCA features.
	// The feature map is then windowed.

	vector<Mat> z;

	if (x_pca.cols == 0)
	{
		// if no PCA-features exist, only use non-PCA
		z.push_back(x_npca);
	}
	else
	{
		// add x_npca if it exists first
		if (x_npca.cols != 0)
		{ //if not empty
			z.push_back(x_npca);
		}

		// project the PCA-features using the projection matrix and reshape
		// to a window



//		Mat tmp = (x_pca * projection_matrix);

		Mat tmp;
		x_pca.copyTo(tmp);

	//	int sizes[] = { cos_window.rows, cos_window.cols, projection_matrix.cols };
		for (int i = 0; i < tmp.cols; i++)
		{
			Mat tmpCol = tmp.col(i).clone();
			Mat tmpCol2(sz_with_padding.width, sz_with_padding.height, CV_64FC1);
			memcpy(tmpCol2.data, tmpCol.data, tmp.rows * sizeof(double));
			// concatinate the feature windows
			z.push_back(tmpCol2.t());
		}
	}

	// do the windowing of the output
// 	int sz = z.size();
// 	for (int i = 0; i < sz; i++)
// 	{
// 		Mat tmp = z[i].mul(cos_window);
// 		z[i] = tmp;
// 	}
	return z;
}







void CompressiveTracker::testgetMultiFeatureValue(Mat& zz, vector<Mat>& cMap, vector<Rect>& _sampleBox, Mat& _sampleFeatureValue)
{

	int sampleBoxSize = _sampleBox.size();
	_sampleFeatureValue.create(featureNum, sampleBoxSize, CV_32F);
	float tempValue;
	int xMin;
	int xMax;
	int yMin;
	int yMax;



	for (int i = 0; i < featureNum; i++)
	{
		for (int j = 0; j < sampleBoxSize; j++)
		{
			tempValue = 0.0f;
			for (size_t k = 0; k < features[i].size(); k++)
			{

				feaMap[i] = 1;

				xMin = _sampleBox[j].x + features[i][k].x;
				xMax = _sampleBox[j].x + features[i][k].x + features[i][k].width;
				yMin = _sampleBox[j].y + features[i][k].y;
				yMax = _sampleBox[j].y + features[i][k].y + features[i][k].height;
				tempValue += featuresWeight[i][k] * (zz.at<double>(yMin, xMin) + zz.at<double>(yMax, xMax) - zz.at<double>(yMin, xMax) - zz.at<double>(yMax, xMin));
			}
			_sampleFeatureValue.at<float>(i, j) = tempValue;
		}
	}

}

// Compute the features of samples
void CompressiveTracker::getFeatureValue(Mat& _imageIntegral, vector<Rect>& _sampleBox, Mat& _sampleFeatureValue)
{
	int sampleBoxSize = _sampleBox.size();
	_sampleFeatureValue.create(featureNum, sampleBoxSize, CV_32F);
	float tempValue;
	int xMin;
	int xMax;
	int yMin;
	int yMax;

	for (int i = 0; i < featureNum; i++)
	{
		for (int j = 0; j < sampleBoxSize; j++)
		{
			tempValue = 0.0f;
			for (size_t k = 0; k < features[i].size(); k++)
			{
				xMin = _sampleBox[j].x + features[i][k].x;
				xMax = _sampleBox[j].x + features[i][k].x + features[i][k].width;
				yMin = _sampleBox[j].y + features[i][k].y;
				yMax = _sampleBox[j].y + features[i][k].y + features[i][k].height;
				tempValue += featuresWeight[i][k] * (_imageIntegral.at<double>(yMin, xMin) + _imageIntegral.at<double>(yMax, xMax) - _imageIntegral.at<double>(yMin, xMax) - _imageIntegral.at<double>(yMax, xMin));
			}
			_sampleFeatureValue.at<float>(i, j) = tempValue;



		}
	}
}

void CompressiveTracker::getMultiFeatureValue(vector<Mat>& cMap, vector<Rect>& _sampleBox, Mat& _sampleFeatureValue)
{

	int sampleBoxSize = _sampleBox.size();
	_sampleFeatureValue.create(featureNum, sampleBoxSize, CV_32F);
	float tempValue;
	int xMin;
	int xMax;
	int yMin;
	int yMax;



	for (int i = 0; i < featureNum; i++)
	{

	//	cout << feaMap[i] << endl;

		for (int j = 0; j < sampleBoxSize; j++)
		{

			tempValue = 0.0f;
			for (size_t k = 0; k < features[i].size(); k++)
			{

		//		feaMap[i] = 1;



				xMin = _sampleBox[j].x + features[i][k].x;
				xMax = _sampleBox[j].x + features[i][k].x + features[i][k].width;
				yMin = _sampleBox[j].y + features[i][k].y;
				yMax = _sampleBox[j].y + features[i][k].y + features[i][k].height;




				tempValue += featuresWeight[i][k] * (cMap[feaMap[i]].at<double>(yMin, xMin) + cMap[feaMap[i]].at<double>(yMax, xMax) - cMap[feaMap[i]].at<double>(yMin, xMax) - cMap[feaMap[i]].at<double>(yMax, xMin));
			}

			_sampleFeatureValue.at<float>(i, j) = tempValue;
		}
	}



}











// 	data_matrix = Mat::zeros(z_pca.rows, z_pca.cols, CV_64FC1);
// 	reduce(z_pca, data_mean, 0, CV_REDUCE_AVG);
// 	// substract the mean from the appearance to get the data matrix
// 	double*data = ((double*)data_matrix.data);
// 	for (int i = 0; i < z_pca.rows; i++)
// 	{
// 		memcpy(data + i * z_pca.cols, ((Mat)(z_pca.row(i) - data_mean)).data, z_pca.cols * sizeof(double));
// 	}
// 	// calculate the covariance matrix
// 	cov_matrix = (1.0 / (sz_with_padding.width * sz_with_padding.height - 1))* (data_matrix.t() * data_matrix);
// 	Mat vt;
// 	cv::SVD::compute(cov_matrix, pca_variances, pca_basis, vt);
// 	projection_matrix = pca_basis(cv::Rect(0, 0, numCompressDim, pca_basis.rows)).clone();
// 	Mat projection_variances = Mat::zeros(numCompressDim, numCompressDim, CV_64FC1);
// 	for (int i = 0; i < numCompressDim; i++)
// 	{
// 		((double*)projection_variances.data)[i + i*numCompressDim] = ((double*)pca_variances.data)[i];
// 	}
// 
// 	old_cov_matrix = projection_matrix * projection_variances * projection_matrix.t();







// 	reduce(z_pca, data_mean, 0, CV_REDUCE_AVG);
// 	double*data = ((double*)data_matrix.data);
// 	for (int i = 0; i < z_pca.rows; i++)
// 	{
// 		memcpy(data + i * z_pca.cols, ((Mat)(z_pca.row(i) - data_mean)).data, z_pca.cols * sizeof(double));
// 	}
// 	cov_matrix = (1.0 / (sz_with_padding.width * sz_with_padding.height - 1))
// 		* (data_matrix.t() * data_matrix);
// 	Mat vt;
// 	cv::SVD::compute((1 - 0.075) * old_cov_matrix + 0.075 * cov_matrix,
// 		pca_variances, pca_basis, vt);
// 
// 	projection_matrix = pca_basis(cv::Rect(0, 0, numCompressDim, pca_basis.rows)).clone();
// 	Mat projection_variances = Mat::zeros(numCompressDim, numCompressDim, CV_64FC1);
// 	for (int i = 0; i < numCompressDim; i++)
// 	{
// 		((double*)projection_variances.data)[i + i * numCompressDim] = ((double*)pca_variances.data)[i];
// 	}
// 	old_cov_matrix =(1 - 0.075) * old_cov_matrix + 0.075 * (projection_matrix * projection_variances * projection_matrix.t());
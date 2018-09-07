/************************************************************************
* File:	RunTracker.cpp
* Brief: C++ demo for Kaihua Zhang's paper:"Real-Time Compressive Tracking"
* Version: 1.0
* Author: Yang Xian
* Email: yang_xian521@163.com
* Date:	2012/08/03
* History:
************************************************************************/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include "CompressiveTracker.h"



using namespace caffe;
using namespace cv;
using namespace std;

Rect box; // tracking object
bool drawing_box = false;
bool gotBB = false;	// got tracking box or not
bool fromfile = false;
string video;

#define NetF float 
// tracking box mouse callback
void mouseHandler(int event, int x, int y, int flags, void *param)
{
	switch (event)
	{
	case CV_EVENT_MOUSEMOVE:
		if (drawing_box)
		{
			box.width = x - box.x;
			box.height = y - box.y;
		}
		break;
	case CV_EVENT_LBUTTONDOWN:
		drawing_box = true;
		box = Rect(x, y, 0, 0);
		break;
	case CV_EVENT_LBUTTONUP:
		drawing_box = false;
		if (box.width < 0)
		{
			box.x += box.width;
			box.width *= -1;
		}
		if( box.height < 0 )
		{
			box.y += box.height;
			box.height *= -1;
		}
		gotBB = true;
		break;
	default:
		break;
	}
}


void tracking(string srcVideo, string srcRes, int* bd);

vector<vector<double> > CompressiveTracker::w2c_t;
vector<vector<double> > CompressiveTracker::w2c;
boost::shared_ptr< Net<float> > CompressiveTracker::ctNet;



void zload_w2c()
{
	// load the normalized Color Name matrix
	ifstream ifstr("w2crs.txt");
	for (int i = 0; i < 10; i++)
	{
		CompressiveTracker::w2c_t.push_back(vector<double>(32768, 0));
	}
	vector<double> tmp(10, 0);
	for (int i = 0; i < 32768; i++)
	{
		CompressiveTracker::w2c.push_back(tmp);
	}
	double tmp_val;
	for (int i = 0; i < 32768; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			ifstr >> tmp_val;
			CompressiveTracker::w2c[i][j] = CompressiveTracker::w2c_t[j][i] = tmp_val;
		}
	}
	ifstr.close();		
}


void loadNet()
{
	Caffe::set_mode(Caffe::GPU);
	char *proto = "/data1/ztq/experiment/CT+CNNmap/model/VGG/VGG_ILSVRC_16_layers_deploy_all.prototxt"; /* 加载CaffeNet的配置 */
	Phase phase = TEST; /* or TRAIN */
	CompressiveTracker::ctNet = boost::shared_ptr<Net<float>>(new caffe::Net<float>(proto, phase));
//boost::shared_ptr< Net<float> > net(new caffe::Net<float>(proto, phase));

	char *model = "/data1/ztq/experiment/CT+CNNmap/model/VGG/VGG_ILSVRC_16_layers.caffemodel";
	CompressiveTracker::ctNet->CopyTrainedLayersFrom(model);
}


int main()
{



	string str2 = "/data1/ztq/experiment/";
	string str3 = "/data1/ztq/experiment/CT+CNNmap/test_conv2_1/resultBD/";
	string srcVideo, srcBD, srcRes;
	string tmp;
	loadNet();

string sequenceRoot = "/data1/ztq/experiment/dirs.txt";
ifstream sequenceName(sequenceRoot);
string sequencePer;

while(sequenceName >> sequencePer)
{



			string fileNname = sequencePer + ".avi";

			srcVideo = str2 + "ALLvideo/" + fileNname;
			srcBD = str2 + "sequence/" + fileNname;
			srcBD.erase(srcBD.length() - 4, 4);
			srcBD = srcBD + "/groundtruth_rect.txt";
			srcRes = str3 + fileNname;
			srcRes.erase(srcRes.length() - 4, 4);
			srcRes = srcRes + ".txt";





			ifstream infile(srcBD);
			int pos;
			int bd[4] = { 0 };
			string firstNumber;
			infile >> firstNumber;
			pos = firstNumber.find(',');
			if (pos < 0)
			{
				bd[0] = atoi(firstNumber.c_str());
				infile >> bd[1];
				infile >> bd[2];
				infile >> bd[3];
			}

			else
			{
				string word;
				vector<string> vec;
				while (1)
				{
					int pos = firstNumber.find(',');
					if (pos == 0)
					{
						firstNumber = firstNumber.substr(1);
						continue;
					}
					if (pos < 0)
					{
						vec.push_back(firstNumber);
						int m = 0;
						m++;
						break;
					}
					word = firstNumber.substr(0, pos);
					firstNumber = firstNumber.substr(pos + 1);
					vec.push_back(word);

				}
				string oo;
				oo = vec.at(0);
				bd[0] = atoi(oo.c_str());
				oo = vec.at(1);
				bd[1] = atoi(oo.c_str());
				oo = vec.at(2);
				bd[2] = atoi(oo.c_str());
				oo = vec.at(3);
				bd[3] = atoi(oo.c_str());

			}





			string tmpRoot1 = "/data1/ztq/experiment/sequence/";
			string tmpRoot2 = fileNname;
			tmpRoot2.erase(tmpRoot2.length() - 4, 4);
			string tmpRoot3 = tmpRoot1 + tmpRoot2 + "/img/0001.jpg";
			Mat tmpIm = imread(tmpRoot3);



			Scalar muTemp;
			Scalar sigmaTemp;
			meanStdDev(tmpIm, muTemp, sigmaTemp);



			int count = 0;
			if (muTemp.val[0] == muTemp.val[1])
				count++;
			if (muTemp.val[0] == muTemp.val[2])
				count++;
			if (sigmaTemp.val[0] == sigmaTemp.val[1])
				count++;
			if (sigmaTemp.val[0] == sigmaTemp.val[2])
				count++;


			if (count != 4 && (4 * bd[2] * bd[3]) < 32768)
  			{
				tracking(srcVideo, srcRes, bd);
			}
			
}
	return 0;
}


void tracking(string srcVideo, string srcRes, int* bd)
{
	ofstream outfile(srcRes);

	VideoCapture capture;
	capture.open(srcVideo);

	// Register mouse callback to draw the tracking box

	// CT framework

	Mat frame;
	Mat last_gray;

	capture.set(CV_CAP_PROP_FRAME_WIDTH, 340);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

	// Initialization


	capture >> frame;




// 	while (!gotBB)
// 	{
// 		rectangle(frame, box, Scalar(0, 0, 255));
// 		imshow("CT", frame);
// 		waitKey(1);
// 	}

	


	// Remove callback
	setMouseCallback("CT", NULL, NULL);
	// CT initialization
	CompressiveTracker ct;

	box.x = bd[0];
	box.y = bd[1];
	box.width = bd[2];
	box.height = bd[3];


	ct.init(frame, box);

	outfile << box.x << " " << box.y << " " << box.width << " " << box.height << endl;

	Mat current_gray;

	srcRes.erase(srcRes.length() - 4, 4);
	srcRes = srcRes + ".avi";

//	VideoWriter writer(srcRes, CV_FOURCC('M', 'J', 'P', 'G'), 40, frame.size(), 1);

	while (capture.read(frame))
	{



		//box这个量既使用又更新，所以需要用引用来使用
		ct.processFrame(frame, box);

		rectangle(frame, box, Scalar(0, 0, 255));

		outfile << box.x << " " << box.y << " " << box.width << " " << box.height << endl;

	//	imshow("CT", frame);

//		writer << frame;

		if (cvWaitKey(33) == 'q') { break; }
	}

	outfile.close();
}



/***************
std::string Blob_names = "conv1_3";
	net->ForwardPrefilled();
	const boost::shared_ptr<Blob<float> > feature_blob = net->blob_by_name(Blob_names);//net的blob指的是特征图
    float num_imgs = feature_blob->num();
            cout << num_imgs << endl; 
            float feat_dim = ->count() / feature_blob->num();//计算特征维度
            cout << feat_dim << endl;
            const float* data_ptr = (const float *)feature_blob->cpu_data();//特征块数据
            cout << *data_ptr << endl;
************/

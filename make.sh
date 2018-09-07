#!/bin/sh
g++ CompressiveTracker.cpp RunTracker.cpp -o CT  -std=c++11 \
-lboost_system -I/usr/local/include/boost   \
-I /usr/local/include/opencv  \
-I /data1/ztq/experiment/caffe1601/caffe/include  \
-I /usr/local/cuda/include   \
-I /data1/ztq/experiment/caffe1601/caffe/src/caffe   \
-I /data1/ztq/experiment/caffe1601/caffe/build/src  \
-L /data1/ztq/experiment/caffe1601/caffe/.build_release/lib/ -lcaffe \
-L/usr/local/lib \
-L /usr/local/lib/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lglog -lgflags 



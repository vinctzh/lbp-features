#include "LTP.hpp"
#include <iostream>
#include <limits>

#include "lbputils.hpp"
using namespace std;

LTP::LTP():LTP(8,1,6){
}

LTP::LTP(int samples,int radius, int thresh)
	:samples(samples),radius(radius),thresh(thresh)
{
	neigbourhood = NULL;
	lbputils::generateNeigbourhood(&neigbourhood, samples, radius);
	uniformMap = NULL;
	lbputils::generateUniformPatterns(&uniformMap, samples);
}

LTP::~LTP(){
	delete neigbourhood;
	neigbourhood = NULL;
	delete uniformMap;
	uniformMap = NULL;
}

int LTP::getLTPBlock(const Mat& src, int row, int col, bool uniform, int mode){
	int ltp = 0;
	for (int n = 0; n < samples; n++){
	//for (int n = samples-1; n >=0; n--){

		// sample points
		double x = neigbourhood[2 * n];
		double y = neigbourhood[2 * n + 1];
		
		double interValue = lbputils::getBiLinearInterpolatedPixel(src, x + row, y + col);
		double centerValue;

		switch (src.type())
		{
		case CV_8SC1:
			centerValue = (double)src.at<char>(row, col);
			break;
		case CV_8UC1:
			centerValue = (double)src.at<unsigned char>(row, col);
			break;
		case CV_16SC1:
			centerValue = (double)src.at<short>(row, col);
			break;
		case CV_16UC1:
			centerValue = (double)src.at<unsigned short>(row, col);
			break;
		case CV_32SC1:
			centerValue = (double)src.at<int>(row, col);
			break;
		case CV_32FC1:
			centerValue = (double)src.at<float>(row, col);
			break;
		case CV_64FC1:
			centerValue = (double)src.at<double>(row, col);
			break;
		}
		double differ = interValue - centerValue;
		if ((differ > thresh) && (abs(differ-thresh) > std::numeric_limits<float>::epsilon())){
			if (mode == LTP_UPPER_MODE){
				ltp += (1 << n);
			}
		}
		else if ((differ < -thresh) && (abs(differ + thresh) > std::numeric_limits<float>::epsilon())) {
			if (mode == LTP_LOWER_MODE){
				ltp += (1 << n);
			}
		}
	}
	if (uniform)
		return uniformMap[ltp];
	else 
		return ltp;
}

Mat LTP::getLTP(const Mat& src, bool uniform, int mode){
	int rows = src.rows;
	int cols = src.cols;
	int newRows = rows + 2 * radius;
	int newCols = cols + 2 * radius;
	Mat result = Mat::zeros(rows, cols, CV_32SC1);
	Mat paddedSrc = Mat::ones(newRows, newCols, src.type());

	src.copyTo(paddedSrc(Rect(radius, radius, cols, rows)));

	for (int i = radius; i < newRows - radius; i++){
		for (int j = radius; j < newCols - radius; j++){
			result.at<int>(i - radius, j - radius) = getLTPBlock(paddedSrc, i, j, uniform, mode);
		}
	}
	return result;
}
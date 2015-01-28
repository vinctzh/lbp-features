#include "LBP.hpp"
#include <iostream>
#include <limits>
using namespace std;

#define PI CV_PI
LBP::LBP():LBP(8,1){
}
LBP::LBP(int samples, int radius):samples(samples),radius(radius){
	generateNeigbourhood();
	generateUniformPatterns();
}

LBP::~LBP(){
	delete neigbourhood;
	neigbourhood = NULL;
	delete uniformMap;
	uniformMap = NULL;
}
void LBP::generateNeigbourhood() {
	neigbourhood = new double[2 * samples];
	const double increaseStep = 2.0 * PI / samples;
	for (int i = 0; i <samples; i++){
		neigbourhood[i * 2] = (double)radius * cos((double)i*increaseStep);
		neigbourhood[i * 2 + 1] = (double)radius*-sin((double)i*increaseStep);
	}
}
bool isUniform(unsigned int pattern, int samples){
	int bitSize = samples;
	int *binPattern = new int[bitSize];
	binPattern[bitSize - 1] = pattern % 2;
	for (int i = bitSize - 1; i >= 0; i--){
		binPattern[i] = pattern % 2;
		pattern >>= 1;
	}
	int count = 0;
	for (int i = 0; i < bitSize-1; i++){
		if (binPattern[i] != binPattern[i + 1])
			count++;
	}
	if (binPattern[0] != binPattern[bitSize - 1])
		count++;
	delete binPattern;
	if (count > 2) 
		return false;
	else
		return true;
}

void LBP::generateUniformPatterns(){
	int length = pow(2.0, samples);
	int nonUniformPattern = (samples*(samples - 1)) + 2;
	int uniformCount = 0;
	uniformMap = new int[length];
	for (int i = 0; i < length; i++){
		if (isUniform((unsigned int)i,samples))
			uniformMap[i] = uniformCount++;
		else
			uniformMap[i] = nonUniformPattern;
	}	
}

void LBP::printNeigbourhood(){
	/*
	(-1, -1)	(-1.0)	(-1,1)
	(0,-1)		(0,0)	(0,1)
	(1,-1)		(1,0)	(1,1)
	*/
	for (int i = 0; i < samples; i++){
		cout << "(" << neigbourhood[i * 2] << "," << neigbourhood[i * 2 + 1] <<")" << endl;
	}
}

Mat LBP::getLBP(const Mat& src, bool uniform){
	int rows = src.rows;
	int cols = src.cols;
	int newRows = rows + 2*radius;
	int newCols = cols + 2*radius;
	Mat result = Mat::zeros(rows, cols, CV_32SC1);
	Mat paddedSrc = Mat::ones(newRows, newCols,src.type());
	
	src.copyTo(paddedSrc(Rect(radius, radius, cols, rows)));
	for (int i = radius; i < newRows - radius; i++){
		for (int j = radius; j < newCols - radius; j++){
			result.at<int>(i - radius, j - radius) = getLBPBlock(paddedSrc, i, j,uniform);
		}
	}
	return result;
}

int LBP::getLBPBlock(const Mat& src, int row, int col, bool uniform){
	int lbp = 0;
	for (int n = 0; n <samples; n++){
		// sample points
		double x = neigbourhood[2*n];
		double y = neigbourhood[2*n+1];
		// relative indices		
		double interValue = getBiLinearInterpolatedPixel(src, x + row, y + col);
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
		lbp += ((interValue > centerValue) && (abs(interValue - centerValue) > std::numeric_limits<float>::epsilon())) << n;
	}
	if (uniform)
		return uniformMap[lbp];
	else 
		return lbp;
}

double getPixelValue(const Mat& src, int row, int col){
	double pixelValue;
	switch (src.type())
	{
	case CV_8SC1:
		pixelValue = (double)src.at<char>(row, col);
		break;
	case CV_8UC1:
		pixelValue = (double)src.at<unsigned char>(row, col);
		break;
	case CV_16SC1:
		pixelValue = (double)src.at<short>(row, col);
		break;
	case CV_16UC1:
		pixelValue = (double)src.at<unsigned short>(row, col);
		break;
	case CV_32SC1:
		pixelValue = (double)src.at<int>(row, col);
		break;
	case CV_32FC1:
		pixelValue = (double)src.at<float>(row, col);
		break;
	case CV_64FC1:
		pixelValue = (double)src.at<double>(row, col);
		break;
	}
	return pixelValue;
}

double getSingleLinearInterpolatedPixel(const Mat& src, double row, double col){
	int row_round = (int)round(row);
	int col_round = (int)round(col);
	return getPixelValue(src, row_round, col_round);
}

double LBP::getBiLinearInterpolatedPixel(const Mat& src, double row, double col){
	int row_floor = (int)floor(row);
	int row_ceil = (int)ceil(row);
	int col_floor = (int)floor(col);
	int col_ceil = (int)ceil(col);

	if (row_floor == row_ceil || col_floor == col_ceil){
		return getSingleLinearInterpolatedPixel(src, row, col);
	}
	double dcol = col_ceil - col_floor;
	double w1 = (col_ceil - col) / dcol;
	double w2 = (col - col_floor) / dcol;
	double drow = row_ceil - row_floor;
	double w3 = (row_ceil - row) / drow;
	double w4 = (row - row_floor) / drow;

	double interpo;
	switch (src.type())
	{
	case CV_8SC1:
		interpo = w3*(w1*src.at<char>(row_floor, col_floor) + w2*src.at<char>(row_floor, col_ceil)) + w4*(w1*src.at<char>(row_ceil, col_floor) + w2*src.at<char>(row_floor, col_ceil));
		break;
	case CV_8UC1:
		interpo = w3*(w1*src.at<unsigned char>(row_floor, col_floor) + w2*src.at<unsigned char>(row_floor, col_ceil)) + w4*(w1*src.at<unsigned char>(row_ceil, col_floor) + w2*src.at<unsigned char>(row_floor, col_ceil));
		break;
	case CV_16SC1:
		interpo = w3*(w1*src.at<short>(row_floor, col_floor) + w2*src.at<short>(row_floor, col_ceil)) + w4*(w1*src.at<short>(row_ceil, col_floor) + w2*src.at<short>(row_floor, col_ceil));
		break;
	case CV_16UC1:
		interpo = w3*(w1*src.at<unsigned short>(row_floor, col_floor) + w2*src.at<unsigned short>(row_floor, col_ceil)) + w4*(w1*src.at<unsigned short>(row_ceil, col_floor) + w2*src.at<unsigned short>(row_floor, col_ceil));
		break;
	case CV_32SC1:
		interpo = w3*(w1*src.at<int>(row_floor, col_floor) + w2*src.at<int>(row_floor, col_ceil)) + w4*(w1*src.at<int>(row_ceil, col_floor) + w2*src.at<int>(row_floor, col_ceil));
		break;
	case CV_32FC1:
		interpo = w3*(w1*src.at<float>(row_floor, col_floor) + w2*src.at<float>(row_floor, col_ceil)) + w4*(w1*src.at<float>(row_ceil, col_floor) + w2*src.at<float>(row_floor, col_ceil));
		break;
	case CV_64FC1:
		interpo = w3*(w1*src.at<double>(row_floor, col_floor) + w2*src.at<double>(row_floor, col_ceil)) + w4*(w1*src.at<double>(row_ceil, col_floor) + w2*src.at<double>(row_floor, col_ceil));
		break;	
	default:
		interpo = 0;
		break;
	}
	return interpo;
}
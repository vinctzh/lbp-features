#include "LBP.hpp"
#include "LTP.hpp"
#include <iostream>
using namespace std;

int main(){
	LBP *lbp = new LBP(8,2);
	Mat img = imread("yaleB03_P00A+000E+00.pgm", 0);
	imshow("src", img);
	Mat lbpImg = lbp->getLBP(img, true);
	Mat showImg;
	lbpImg.convertTo(showImg, CV_8UC1);
	normalize(showImg, showImg, 0, 255, CV_MINMAX);
	imshow("lbp", showImg);
	LTP *ltp = new LTP();
	Mat ltpImg =ltp->getLTP(img, true, LTP_UPPER_MODE);
	ltpImg.convertTo(showImg, CV_8UC1);
	normalize(showImg, showImg, 0, 255, CV_MINMAX);
	imshow("upperltp", showImg);

	ltpImg = ltp->getLTP(img, true, LTP_LOWER_MODE);
	ltpImg.convertTo(showImg, CV_8UC1);
	normalize(showImg, showImg, 0, 255, CV_MINMAX);
	imshow("lowerltp", showImg);

	delete ltp;
	waitKey();
	delete lbp;
	
}
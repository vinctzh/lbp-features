#include <math.h>
#include <opencv2\opencv.hpp>
using namespace cv;
class LBP
{
public:
	LBP();
	LBP(int samples, int radius);
	~LBP();
	Mat getLBP(const Mat& src,bool uniform);

	void printNeigbourhood();
	

private:
	int samples;
	int radius;
	double *neigbourhood;
	int *uniformMap;
	void generateNeigbourhood();
	void generateUniformPatterns();
	int getLBPBlock(const Mat& src, int row, int col, bool uniform);
	double getBiLinearInterpolatedPixel(const Mat& src, double row, double col);
};
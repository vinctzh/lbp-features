#include <math.h>
#include <opencv2\opencv.hpp>
using namespace cv;

#define LTP_UPPER_MODE 0
#define LTP_LOWER_MODE 1

class LTP
{
public:
	LTP();
	LTP(int samples, int radius, int thresh);
	~LTP();
	Mat getLTP(const Mat&src, bool uniform, int mode);
	
private:
	int samples;
	int radius;
	int thresh;
	double *neigbourhood;
	int *uniformMap;
	void setThresh(int thresh);
	int getLTPBlock(const Mat& src, int row, int col, bool uniform, int mode);
};

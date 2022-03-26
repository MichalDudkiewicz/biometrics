#include "BiometricsProject.h"
#include "EasyBMP/EasyBMP.h"
#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat image;
    namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    cv::waitKey(0);
	std::cout << "Hello biometrics" << std::endl;
	BMP Image;
	Image.SetSize(400, 400);
	Image.WriteToFile("test.bmp");
	return 0;
}

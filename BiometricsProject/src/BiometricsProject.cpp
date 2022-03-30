#include "BiometricsProject.h"
#include "EasyBMP/EasyBMP.h"
#include "opencv2/core/mat.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

int main()
{
	std::string image_path = cv::samples::findFile("101_2.tif");
	cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

	cv::Mat out;
	cv::namedWindow("Display Image1", cv::WINDOW_AUTOSIZE );

	cv::imshow("Display Image2", img);
    cv::waitKey(0);

    cv::medianBlur(img, out, 5);
    cv::imshow("Display Image3", out);
    cv::waitKey(0);

	cv::adaptiveThreshold(out, img, 255, cv::AdaptiveThresholdTypes::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 17, 1);
	cv::imshow("Display Image4", img);
	cv::waitKey(0);

    cv::medianBlur(img, out, 5);
    cv::imshow("Display Image5", out);
    cv::waitKey(0);

    cv::bitwise_not(out, img);
    cv::imshow("Display Image6", img);
    cv::waitKey(0);

    cv::ximgproc::thinning(img, out, cv::ximgproc::ThinningTypes::THINNING_ZHANGSUEN);
    cv::imshow("Display Image7", out);
    cv::waitKey(0);

    cv::bitwise_not(out, img);
    cv::imshow("Display Image8", img);
    cv::waitKey(0);

	return 0;
}

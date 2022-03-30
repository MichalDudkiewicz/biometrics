#include "BiometricsProject.h"
#include "EasyBMP/EasyBMP.h"
#include "opencv2/core/mat.hpp"
#include <opencv2/opencv.hpp>

int main()
{
	std::string image_path = cv::samples::findFile("101_2.tif");
	cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

	cv::Mat out;
	cv::namedWindow("Display Image1", cv::WINDOW_AUTOSIZE );

	cv::imshow("Display Image2", img);
    cv::waitKey(0);

//	cv::cvtColor(img, out);

	//tried cropping
	//cv::Rect crop = cv::boundingRect(out);
	//cv::Mat cropped_img = out(cv::Range(crop.y, crop.y + crop.height), cv::Range(crop.x, crop.x + crop.width));
	//cv::imshow("Display Image", cropped_img);
	//cv::waitKey(0);

    cv::medianBlur(img, out, 3);
    cv::imshow("Display Image3", out);
    cv::waitKey(0);

	cv::adaptiveThreshold(out, img, 255, cv::AdaptiveThresholdTypes::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 17, 1);
	cv::imshow("Display Image4", img);
	cv::waitKey(0);

    cv::medianBlur(img, out, 5);
    cv::imshow("Display Image5", out);
    cv::waitKey(0);

//    cv::ximgproc::thinning

//	cv::morphologyEx(img, out, cv::MORPH_OPEN, cv::getStructuringElement(1 ,cv::Size(3,3), cv::Point(1,1)));
//	cv::imshow("Display Image5", out);
//	cv::waitKey(0);
//
//	cv::dilate(out, img, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(2, 2)));
//	cv::imshow("Display Image6", img);
//	cv::waitKey(0);
//
//	cv::dilate(img, out, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2), cv::Point(1, 1)));
//	cv::imshow("Display Image7", out);
//	cv::waitKey(0);

	//cv::Laplacian(out, img, CV_16S, 3, 1, 0, cv::BORDER_DEFAULT);
	//cv::convertScaleAbs(img, out);
	//cv::imshow("Display Image", out);
	//cv::waitKey(0);

	return 0;
}

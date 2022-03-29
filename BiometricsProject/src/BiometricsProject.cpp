#include "BiometricsProject.h"
#include "EasyBMP/EasyBMP.h"
#include <opencv2/opencv.hpp>

int main()
{
	std::string image_path = cv::samples::findFile("110_2.tif");
	cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
	cv::Mat out;

	cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );

	cv::imshow("Display Image", img);
    cv::waitKey(0);

	cv::cvtColor(img, out, cv::COLOR_BGR2GRAY);

	//tried cropping
	//cv::Rect crop = cv::boundingRect(out);
	//cv::Mat cropped_img = out(cv::Range(crop.y, crop.y + crop.height), cv::Range(crop.x, crop.x + crop.width));
	//cv::imshow("Display Image", cropped_img);
	//cv::waitKey(0);
	
	cv::threshold(img, out, 80, 255, cv::THRESH_BINARY);
	cv::imshow("Display Image", out);
	cv::waitKey(0);

	cv::morphologyEx(out, img, cv::MORPH_OPEN, cv::getStructuringElement(1 ,cv::Size(3,3), cv::Point(1,1)));
	cv::imshow("Display Image", img);
	cv::waitKey(0);

	cv::dilate(img, out, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(2, 2)));
	cv::imshow("Display Image", out);
	cv::waitKey(0);

	cv::medianBlur(out, img, 3);
	cv::imshow("Display Image", img);
	cv::waitKey(0);

	cv::dilate(img, out, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2), cv::Point(1, 1)));
	cv::imshow("Display Image", out);
	cv::waitKey(0);

	//cv::Laplacian(out, img, CV_16S, 3, 1, 0, cv::BORDER_DEFAULT);
	//cv::convertScaleAbs(img, out);
	//cv::imshow("Display Image", out);
	//cv::waitKey(0);

	return 0;
}

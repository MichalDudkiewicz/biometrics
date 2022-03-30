#include "BiometricsProject.h"
#include "EasyBMP/EasyBMP.h"
#include "opencv2/core/mat.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

enum MinutiaeType
{
    End, Fork
};

struct Pixel
{
public:
    Pixel(int x1, int y1) : x(x1), y(y1)
    {

    }

    int x, y;
};

void printMinutaes(const std::vector<std::pair<Pixel, MinutiaeType>> &minutaes, cv::Mat& mat)
{
    for (const auto& minutae : minutaes)
    {
        for (int i = minutae.first.x - 1; i <= minutae.first.x + 1; i++)
        {
            for (int j = minutae.first.y - 1; j <= minutae.first.y + 1; j++)
            {
                uchar& color = mat.at<uchar>(i,j);
                color = 0;
            }
        }

    }
}

std::vector<std::pair<Pixel, MinutiaeType>> retrieveMinutiaes(const cv::Mat& mat)
{
    std::vector<std::pair<Pixel, MinutiaeType>> minutiaes;
    for(int i = 1; i < mat.rows - 1; i++)
    {
        for(int j = 1; j < mat.cols - 1; j++)
        {
            const auto& centerPixel = mat.at<uchar>(i, j);
            if ((int)centerPixel == 0)
            {
                int crossingNumber = 0;
                int colStepper = 1;
                int rowStepper = 0;
                int rowIndex = i - 1;
                int columnIndex = j - 1;
                auto* previousPixel = &mat.at<uchar>(rowIndex, columnIndex);
                do {
                    if (rowIndex == i+1 && columnIndex == j-1)
                    {
                        colStepper = 0;
                        rowStepper = -1;
                    }
                    else if (rowIndex == i+1 && columnIndex == j+1)
                    {
                        colStepper = -1;
                        rowStepper = 0;
                    }
                    else if (rowIndex == i-1 && columnIndex == j+1)
                    {
                        colStepper = 0;
                        rowStepper = 1;
                    }
                    rowIndex += rowStepper;
                    columnIndex += colStepper;
                    auto* pixelValue = &mat.at<uchar>(rowIndex, columnIndex);
                    if (*pixelValue != *previousPixel)
                    {
                        crossingNumber++;
                    }
                    previousPixel = pixelValue;
                } while(!(columnIndex == j - 1 && rowIndex == i - 1));

                const int halfCrossingNumber = crossingNumber / 2;

                if (halfCrossingNumber == 1)
                {
                    Pixel pixel(i, j);
                    minutiaes.emplace_back(std::make_pair(pixel, MinutiaeType::End));

                }
                else if (halfCrossingNumber >= 3)
                {
                    Pixel pixel(i, j);
                    minutiaes.emplace_back(std::make_pair(pixel, MinutiaeType::Fork));
                }
            }
        }
    }
    return minutiaes;
}

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


    const auto minutiaes = retrieveMinutiaes(img);

    printMinutaes(minutiaes, img);

    cv::imshow("Display Image9", img);
    cv::waitKey(0);

	return 0;
}

#include "BiometricsProject.h"
#include "EasyBMP/EasyBMP.h"
#include "opencv2/core/mat.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <tuple>

enum MinutiaeType
{
    End, Fork
};

enum MinutiaeDirection
{
    Horizontal, Vertical, DiagonalDecreasing, DiagonalIncreasing
};

struct Pixel
{
public:
    Pixel(int x1, int y1) : x(x1), y(y1)
    {

    }

    int x, y;
};

void validateMinutaes(std::vector<std::tuple<Pixel, MinutiaeType, MinutiaeDirection>> &minutaes, const cv::Mat& mat)
{
    std::vector<std::tuple<Pixel, MinutiaeType, MinutiaeDirection>> minutaesCopy;
    for (const auto & minutae : minutaes)
    {
        int counter = 0;
        for (int i = std::get<0>(minutae).y + 1; i < mat.cols; i++)
        {
            const auto& color = mat.at<uchar>(std::get<0>(minutae).x, i);
            if (color == 0)
            {
                counter++;
                break;
            }
        }
        for (int i = std::get<0>(minutae).y - 1; i >= 0; i--)
        {
            const auto& color = mat.at<uchar>(std::get<0>(minutae).x, i);
            if (color == 0)
            {
                counter++;
                break;
            }
        }
        for (int i = std::get<0>(minutae).x + 1; i < mat.rows; i++)
        {
            const auto& color = mat.at<uchar>(i, std::get<0>(minutae).y);
            if (color == 0)
            {
                counter++;
                break;
            }
        }
        for (int i = std::get<0>(minutae).x - 1; i >= 0; i--)
        {
            const auto& color = mat.at<uchar>(i, std::get<0>(minutae).y);
            if (color == 0)
            {
                counter++;
                break;
            }
        }
        if (counter > 2)
        {
            minutaesCopy.push_back(minutae);
        }
    }
    minutaes = minutaesCopy;
}

void printMinutaes(const std::vector<std::tuple<Pixel, MinutiaeType, MinutiaeDirection>> &minutaes, cv::Mat& mat)
{
    for (const auto& minutae : minutaes)
    {
        for (int i = std::get<0>(minutae).x - 1; i <= std::get<0>(minutae).x + 1; i++)
        {
            for (int j = std::get<0>(minutae).y - 1; j <= std::get<0>(minutae).y + 1; j++)
            {
                uchar& color = mat.at<uchar>(i,j);
                color = 0;
            }
        }

    }
}

std::vector<std::tuple<Pixel, MinutiaeType, MinutiaeDirection>> retrieveMinutiaes(const cv::Mat& mat)
{
    std::vector<std::tuple<Pixel, MinutiaeType, MinutiaeDirection>> minutiaes;
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

                if (halfCrossingNumber != 2 && halfCrossingNumber > 0)
                {
                    MinutiaeDirection minutiaeDirection;
                    if (mat.at<uchar>(i - 1, j - 1) != mat.at<uchar>(i + 1, j + 1))
                    {
                        minutiaeDirection = MinutiaeDirection::DiagonalDecreasing;
                    }
                    else if (mat.at<uchar>(i + 1, j - 1) != mat.at<uchar>(i - 1, j + 1))
                    {
                        minutiaeDirection = MinutiaeDirection::DiagonalIncreasing;
                    }
                    else if (mat.at<uchar>(i, j - 1) != mat.at<uchar>(i, j + 1))
                    {
                        minutiaeDirection = MinutiaeDirection::Horizontal;
                    }
                    else if (mat.at<uchar>(i + 1, j) != mat.at<uchar>(i - 1, j))
                    {
                        minutiaeDirection = MinutiaeDirection::Vertical;
                    }
                    else
                    {
                        throw std::runtime_error("coś jest nie tak");
                    }
                    Pixel pixel(i, j);

                    if (halfCrossingNumber == 1)
                    {
                        minutiaes.emplace_back(std::make_tuple(pixel, MinutiaeType::End, minutiaeDirection));
                    }
                    else if (halfCrossingNumber >= 3)
                    {
                        minutiaes.emplace_back(std::make_tuple(pixel, MinutiaeType::Fork, minutiaeDirection));
                    }
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


    auto minutiaes = retrieveMinutiaes(img);

    cv::Mat imgCopy = img.clone();
    printMinutaes(minutiaes, img);

    cv::imshow("Display Image9", img);
    cv::waitKey(0);

    validateMinutaes(minutiaes, imgCopy);

    printMinutaes(minutiaes, imgCopy);

    cv::imshow("Display Image10", imgCopy);
    cv::waitKey(0);

	return 0;
}

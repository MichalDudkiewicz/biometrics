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

double getOrientation(const std::vector<cv::Point>& pts)
{
    //Construct a buffer used by the pca analysis
    int sz = static_cast<int>(pts.size());
    cv::Mat data_pts = cv::Mat(sz, 2, CV_64F);
    for (int i = 0; i < data_pts.rows; i++)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }
    //Perform PCA analysis
    cv::PCA pca_analysis(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);
    //Store the center of the object
    cv::Point cntr = cv::Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
        static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
    //Store the eigenvalues and eigenvectors
    std::vector<cv::Point2d> eigen_vecs(2);
    std::vector<double> eigen_val(2);
    for (int i = 0; i < 2; i++)
    {
        eigen_vecs[i] = cv::Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
            pca_analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
    }
    double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
    angle = (angle * 180.f) / CV_PI;
    std::cout << angle << std::endl;
    return angle;
}

int main()
{
	std::string image_path = cv::samples::findFile("101_2.tif");
	cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

	cv::Mat out;
    cv::Mat buffer;
	//cv::namedWindow("Display Image1", cv::WINDOW_AUTOSIZE );
    // 
    //rows = y, cols = x

	cv::imshow("Display Image2", img);
    cv::waitKey(0);

    buffer = img;

    cv::equalizeHist(buffer, out);
    cv::imshow("Display Image hist", out);
    cv::waitKey(0);

    buffer = out;

    //buffer.convertTo(out, CV_32F);
    //buffer = out;

    cv::Mat Result;

    cv::Mat kernel = cv::getGaborKernel(cv::Size(11, 11), 5, 0.0f, 9, 0.04, CV_PI / 4);
    cv::filter2D(buffer, out, CV_32F, kernel);
    Result = out;

    // Convert image to binary
    cv::Mat bw;
    cv::threshold(img, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    // Find all the contours in the thresholded image
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(bw, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    for (size_t i = 0; i < contours.size(); i++)
    {
        // Calculate the area of each contour
        double area = contourArea(contours[i]);
        // Ignore contours that are too small or too large
        if (area < 1e2 || 1e5 < area) continue;
        // Find the orientation of each shape
        //getOrientation(contours[i]);
        kernel = cv::getGaborKernel(cv::Size(11, 11), 5, getOrientation(contours[i]), 10, 0.04, 0);
        cv::filter2D(buffer, out, CV_32F, kernel);
        Result += out;
    }

    Result.convertTo(out, CV_8U, 1.0 / 255.0);
    buffer = out;

    cv::imshow("Display Image gabor", out);
    cv::waitKey(0);

    //cv::medianBlur(buffer, out, 5);
    //cv::imshow("Display Image3", out);
    //cv::waitKey(0);

    //buffer = out;

	cv::adaptiveThreshold(buffer, out, 255, cv::AdaptiveThresholdTypes::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 17, 1);
	cv::imshow("Display Image4", out);
	cv::waitKey(0);

    buffer = out;

    //cv::medianBlur(buffer, out, 3);
    //cv::imshow("Display Image5", out);
    //cv::waitKey(0);

    //buffer = out;

    cv::ximgproc::thinning(buffer, out, cv::ximgproc::ThinningTypes::THINNING_ZHANGSUEN);
    cv::imshow("Display Image7", out);
    cv::waitKey(0);

    buffer = out;

    //cv::bitwise_not(buffer, out);
    //cv::imshow("Display Image8", out);
    //cv::waitKey(0);

    //buffer = out;

    auto minutiaes = retrieveMinutiaes(buffer);

    cv::Mat imgCopy = buffer.clone();
    printMinutaes(minutiaes, buffer);

    cv::imshow("Display Image9", buffer);
    cv::waitKey(0);

    validateMinutaes(minutiaes, imgCopy);

    printMinutaes(minutiaes, imgCopy);

    cv::imshow("Display Image10", imgCopy);
    cv::waitKey(0);

	return 0;
}

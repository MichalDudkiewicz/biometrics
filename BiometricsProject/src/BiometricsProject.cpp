#include "BiometricsProject.h"
#include "EasyBMP/EasyBMP.h"
#include "opencv2/core/mat.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <tuple>
#include <numeric>

namespace
{
    constexpr int kSobelSize = 3;
}

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

double Variance(std::vector<double> samples)
{
    const size_t sz = samples.size();
    if (sz == 1) {
        return 0.0;
    }

    // Calculate the mean
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / sz;

    // Now calculate the variance
    auto variance_func = [&mean, &sz](double accumulator, const double& val) {
        return accumulator + ((val - mean)*(val - mean) / (sz - 1));
    };

    return std::accumulate(samples.begin(), samples.end(), 0.0, variance_func);
}

double StandardDeviation(std::vector<double> samples)
{
    return sqrt(Variance(samples));
}

cv::Mat gabor(cv::Mat& myImg, const std::vector<std::vector<std::optional<double>>>& orientationMatrix){
    // prepare the output matrix for filters
    cv::Mat img(myImg.rows, myImg.cols, CV_32F);
    img = cv::Scalar(0);

    //predefine parameters for Gabor kernel
    int k = 30;
    cv::Size kSize(k, k);
    const int sobelInK = k / kSobelSize;

    double lambda = 10;
    double sigma = 4;
    double gamma = 1;

    std::optional<double> previousTheta;

    for (int r = 0; r < orientationMatrix.size() - sobelInK; r+=sobelInK)
    {
        for (int c = 0; c < orientationMatrix[r].size() - sobelInK; c+=sobelInK)
        {
            double sum = 0.0;
            std::vector<double> dirs;
            for (int i = r; i < orientationMatrix.size() && i < r + sobelInK; i++)
            {
                for (int j = c; j < orientationMatrix[r].size() && j < c + sobelInK; j++)
                {
                    const auto dir = orientationMatrix[i][j];
                    if (dir.has_value())
                    {
                        sum += dir.value();
                        dirs.push_back(dir.value());
                    }
                }
            }

            if (dirs.size() > 0)
            {
                const auto stdDev = StandardDeviation(dirs);
                if (stdDev >1.5)
                {
                    const int s = 3;
                    for (int i = r; i < orientationMatrix.size() && i < r + sobelInK; i++)
                    {
                        for (int j = c; j < orientationMatrix[r].size() && j < c + sobelInK; j++)
                        {
                            const auto dir = orientationMatrix[i][j];
                            if (dir.has_value())
                            {
                                double theta = dir.value();
                                cv::Mat kernel = cv::getGaborKernel(cv::Size(s, s), sigma, theta, lambda, gamma);
                                cv::Mat gabor(s, s, CV_32F);
                                cv::filter2D(myImg(cv::Range(i * kSobelSize,i * kSobelSize + s), cv::Range(j * kSobelSize, j * kSobelSize + s)), gabor, CV_32F, kernel);
                                cv::Mat aux = img.rowRange(i * kSobelSize,i * kSobelSize + s).colRange(j * kSobelSize, j * kSobelSize + s);
                                gabor.copyTo(aux);
                            }
                        }
                    }
                }
                else
                {
                    const double theta = sum / dirs.size();
                    cv::Mat kernel = cv::getGaborKernel(kSize, sigma, theta, lambda, gamma);
                    cv::Mat gabor(k, k, CV_32F);
                    cv::filter2D(myImg(cv::Range(r * kSobelSize,r * kSobelSize + k), cv::Range(c * kSobelSize, c * kSobelSize + k)), gabor, CV_32F, kernel);
                    cv::Mat aux = img.rowRange(r * kSobelSize,r * kSobelSize + k).colRange(c * kSobelSize, c * kSobelSize + k);
                    gabor.copyTo(aux);

                    if (previousTheta.has_value() && abs(previousTheta.value() - theta) > 0.5)
                    {

                        double sum2 = 0.0;
                        std::vector<double> dirs2;
                        for (int i = r - sobelInK/2; i < orientationMatrix.size() && i < r + sobelInK/2; i++)
                        {
                            for (int j = c - sobelInK/2; j < orientationMatrix[r].size() && j < c + sobelInK/2; j++)
                            {
                                const auto dir = orientationMatrix[i][j];
                                if (dir.has_value())
                                {
                                    sum2 += dir.value();
                                    dirs2.push_back(dir.value());
                                }
                            }
                        }
                        if (!dirs2.empty())
                        {
                            const double theta2 = sum2 / dirs2.size();
                            cv::Mat kernel2 = cv::getGaborKernel(kSize, sigma, theta2, lambda, gamma);
                            cv::Mat gabor2(k, k, CV_32F);
                            cv::filter2D(myImg(cv::Range(r * kSobelSize - k/2,r * kSobelSize + k/2), cv::Range(c * kSobelSize - k/2, c * kSobelSize + k/2)), gabor2, CV_32F, kernel2);
                            cv::Mat aux2 = img.rowRange(r * kSobelSize - k/2,r * kSobelSize + k/2).colRange(c * kSobelSize - k/2, c * kSobelSize + k/2);
                            gabor2.copyTo(aux2);
                        }

                    }
                    previousTheta = theta;
                }
            }
        }
    }

    return img;
}


int main()
{
	std::string image_path = cv::samples::findFile("101_2.tif");
	cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

	cv::Mat out;
    cv::Mat buffer;

	cv::imshow("Original", img);
    cv::waitKey(0);

    buffer = img;

    const auto clahe = cv::createCLAHE();
    clahe->apply(buffer, out);
    cv::imshow("Contrast stretching", out);
    cv::waitKey(0);
    buffer = out;

    cv::GaussianBlur(buffer, out, cv::Size(3,3), 0);
//    cv::medianBlur(buffer, out, 5);
    cv::imshow("Blur", out);
    cv::waitKey(0);
    buffer = out;

    // orientation image
    const int k = kSobelSize;
    std::vector<std::vector<std::optional<double>>> orientationMatrix;
    for (int i = 0; i < buffer.rows / k - 1; i++)
    {
        int y = i * k + k/2;
        orientationMatrix.emplace_back();
        for (int j = 0; j < buffer.cols / k - 1; j++)
        {
            int x = j * k + k/2;
            cv::Mat cropped = buffer(cv::Range(y - k/2,y + k/2 + 1), cv::Range(x - k/2,x + + k/2 + 1));
            cv::Mat grad_x;
            cv::Scharr(cropped, grad_x, CV_32F, 1, 0, k);
            cv::Mat grad_y;
            cv::Scharr(cropped, grad_y, CV_32F, 0, 1, k);

            cv::Scalar gx = cv::sum(grad_x);
            cv::Scalar gy = cv::sum(grad_y);

            const double Gx = gx.val[0];
            const double Gy = gy.val[0];
            std::optional<double> theta;
            if (Gx != 0 || Gy != 0)
            {
                theta = atan2(Gy, Gx);
                if (theta.value() < 0)
                {
                    theta.value() += M_PI;
                }
            }

            orientationMatrix[i].push_back(theta);
        }
    }


    out = gabor(buffer, orientationMatrix);
    cv::imshow("Gabor", out);
    cv::waitKey(0);
    out.convertTo(buffer, CV_8UC1);
    cv::imshow("Conv", buffer);
    cv::waitKey(0);

//    cv::GaussianBlur(buffer, out, cv::Size(3,3), 0);
//    cv::imshow("Blur", out);
//    cv::waitKey(0);
//    buffer = out;

//    const auto clahe = cv::createCLAHE(40.0, cv::Size(90, 90));
//    clahe->apply(buffer, out);
//    cv::imshow("Contrast stretching", out);
//    cv::waitKey(0);
//    buffer = out;

//    const auto ke = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
//    cv::morphologyEx(buffer, out, cv::MORPH_OPEN, ke);
//    cv::imshow("Blur", out);
//    cv::waitKey(0);
//    buffer = out;

//    cv::bitwise_not(buffer, buffer);
//    cv::imshow("Thinned1", buffer);
    cv::waitKey(0);
    cv::ximgproc::thinning(buffer, out, cv::ximgproc::ThinningTypes::THINNING_ZHANGSUEN);
    cv::bitwise_not(out, out);
    cv::imshow("Thinned2", out);
    cv::waitKey(0);

    auto minutiaes = retrieveMinutiaes(out);

    cv::Mat imgCopy = out.clone();
    printMinutaes(minutiaes, out);

    cv::imshow("Minutaes", out);
    cv::waitKey(0);

    validateMinutaes(minutiaes, imgCopy);

    printMinutaes(minutiaes, imgCopy);

    cv::imshow("Minutaes after validation", imgCopy);
    cv::waitKey(0);

	return 0;
}

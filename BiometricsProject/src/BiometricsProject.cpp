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
    constexpr int K = 100;
    constexpr int L = 4;
    constexpr int M = 400;
    constexpr int N = 600;
    int A[K][L][M][N];
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

bool operator==(const Pixel& pix1, const Pixel& pix2)
{
    return pix1.x == pix2.x && pix1.y==pix2.y;
}

bool operator!=(const Pixel& pix1, const Pixel& pix2)
{
    return !(pix1 == pix2);
}

void validateMinutaes(std::vector<std::tuple<Pixel, MinutiaeType, MinutiaeDirection>> &minutaes, const cv::Mat& mat)
{
    std::vector<std::tuple<Pixel, MinutiaeType, MinutiaeDirection>> minutaesCopy;
    for (const auto & minutae : minutaes)
    {
        bool tooClose = false;
        const auto position = std::get<0>(minutae);
        for (const auto& other : minutaes)
        {
            const auto otherPosition = std::get<0>(other);
            const int distance = std::sqrt(pow(position.x - otherPosition.x, 2) + pow(position.y - otherPosition.y, 2));
            constexpr int threshold = 4;
            if (otherPosition != position && distance < threshold)
            {
                tooClose = true;
                break;
            }
        }
        if (tooClose)
        {
            continue;
        }

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
        if (counter > 3)
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

cv::Mat getMinutaes(const std::vector<std::tuple<Pixel, MinutiaeType, MinutiaeDirection>> &minutaes, const cv::Mat& mat)
{
    cv::Mat minutaesMat = mat;
    minutaesMat = cv::Scalar(255);
    for (const auto& minutae : minutaes)
    {
        const auto dir = std::get<2>(minutae);
        const int i = std::get<0>(minutae).x;
        const int j = std::get<0>(minutae).y;
        uchar& color = minutaesMat.at<uchar>(i,j);
        uchar& color1 = minutaesMat.at<uchar>(i-1,j);
        uchar& color2 = minutaesMat.at<uchar>(i+1,j);
        uchar& color3 = minutaesMat.at<uchar>(i,j-1);
        uchar& color4 = minutaesMat.at<uchar>(i,j+1);
        uchar& color5 = minutaesMat.at<uchar>(i-1,j-1);
        uchar& color6 = minutaesMat.at<uchar>(i+1,j+1);
        uchar& color7 = minutaesMat.at<uchar>(i+1,j-1);
        uchar& color8 = minutaesMat.at<uchar>(i-1,j+1);
        color = 0;
        switch (dir) {
            case MinutiaeDirection::Horizontal:
                color3 = 0;
                color4 = 0;
                break;
            case MinutiaeDirection::Vertical:
                color1 = 0;
                color2 = 0;
                break;
            case MinutiaeDirection::DiagonalDecreasing:
                color5 = 0;
                color6 = 0;
                break;
            case MinutiaeDirection::DiagonalIncreasing:
                color7 = 0;
                color8 = 0;
                break;
            default:
                throw std::runtime_error("unsupported minutae dir");
        }
    }
    return minutaesMat;
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

/*
 * @see https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/
 * @see https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
 * @see https://www.sciencedirect.com/topics/engineering/sobel-edge-detection
 * @see https://en.wikipedia.org/wiki/Sobel_operator
 */
cv::Mat gabor(cv::Mat& myImg, const std::vector<std::vector<std::optional<double>>>& orientationMatrix){
    // prepare the output matrix for filters
    cv::Mat img(myImg.rows, myImg.cols, CV_32F);
    img = cv::Scalar(0);

    //predefine parameters for Gabor kernel
    int k = 30;
    cv::Size kSize(k, k);
    const int sobelInK = k / kSobelSize;

    double lambda = 8;
    double sigma = 4;
    double gamma = 10;

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
                if (stdDev >0.8)
                {
                    const int s = 15;
                    const int newSobelInK = s/kSobelSize;

                    for (int m = 0; m < k/s; m++) {
                        for (int n = 0; n < k/s; n++) {

                            double sum2 = 0.0;
                            std::vector<double> dirs2;

                            int newR = r + newSobelInK * n;
                            int newC = c + m * newSobelInK;

                            for (int i = newR; i < orientationMatrix.size() && i < newR + newSobelInK; i++) {
                                for (int j = newC; j < orientationMatrix[newC].size() && j < newC + newSobelInK; j++) {
                                    const auto dir = orientationMatrix[i][j];
                                    if (dir.has_value()) {
                                        sum2 += dir.value();
                                        dirs2.push_back(dir.value());
                                    }
                                }
                            }

                            if (dirs2.size() > 0)
                            {
                                const double theta = sum2 / dirs2.size();
                                cv::Mat kernel = cv::getGaborKernel(cv::Size(s, s), sigma, theta, 2 * lambda, gamma);
                                cv::Mat gabor(s, s, CV_32F);
                                cv::filter2D(myImg(cv::Range(newR * kSobelSize,newR * kSobelSize + s), cv::Range(newC * kSobelSize, newC * kSobelSize + s)), gabor, CV_32F, kernel);
                                cv::Mat aux = img.rowRange(newR * kSobelSize,newR * kSobelSize + s).colRange(newC * kSobelSize, newC * kSobelSize + s);
                                gabor.copyTo(aux);

                                previousTheta = std::nullopt;
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

void waitForSpace()
{
    while(true)
    {
        const auto k = cv::waitKey();
        if (k==32) // space on linux (?)
        {
            break;
        }
        else if (k==-1)
        {
            continue;
        }
        else
        {
            std::cout << "key pressed: " << k << '\n';
        }
    }
}

/*
 * @see https://answers.opencv.org/question/165566/thumbprint-gabor-filtering-orientation-map-and-normalization/
 * @see https://answers.opencv.org/question/6364/fingerprint-matching-in-mobile-devices-android-platform/
 * @see https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.93.845&rep=rep1&type=pdf
 * @see http://biometrics.cse.msu.edu/Publications/Fingerprint/MSU-CPS-97-35fenhance.pdf
 */
std::vector<std::tuple<Pixel, MinutiaeType, MinutiaeDirection>> minutaesFromFingerprint(const std::string& fingerprintImageLocalPath, bool showPics = false)
{
    std::string image_path = cv::samples::findFile(fingerprintImageLocalPath);
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    cv::Mat out;
    cv::Mat buffer;

    if (showPics)
    {
        cv::imshow("Original " + fingerprintImageLocalPath, img);
        waitForSpace();
    }

    buffer = img;

    // sometimes also DFT is used when needed
    // https://docs.opencv.org/4.x/d8/d01/tutorial_discrete_fourier_transform.html

    // about CLAHE: https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    // https://stackoverflow.com/questions/38504864/opencv-clahe-parameters-explanation
    const auto clahe = cv::createCLAHE();
    clahe->apply(buffer, out);

    if (showPics)
    {
        cv::imshow("Contrast stretching " + fingerprintImageLocalPath, out);
        waitForSpace();
    }
    buffer = out;

    // https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
    cv::GaussianBlur(buffer, out, cv::Size(3,3), 0);
//    cv::medianBlur(buffer, out, 5);
    if (showPics)
    {
        cv::imshow("Blur " + fingerprintImageLocalPath, out);
        waitForSpace();
    }
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


    cv::Mat mask;
    const auto kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::Mat bufferMask;
    cv::morphologyEx(buffer, mask, cv::MORPH_OPEN, kernel2, cv::Point(-1, -1),5);
    cv::morphologyEx(mask, bufferMask, cv::MORPH_DILATE, kernel2, cv::Point(-1, -1),10);
    cv::threshold(bufferMask, mask, 200, 255, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8UC1);
//    cv::bitwise_not(mask, mask);
    buffer = buffer + mask;

    if (showPics)
    {
        cv::imshow("Masked before gabor " + fingerprintImageLocalPath, buffer);
        waitForSpace();
    }

    out = gabor(buffer, orientationMatrix);
    if (showPics)
    {
        cv::imshow("Gabor " + fingerprintImageLocalPath, out);
        waitForSpace();
    }
    buffer = out;

//    cv::Mat mask;
//    const auto kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30,30));
//    cv::Mat bufferMask;
//    cv::morphologyEx(out, mask, cv::MORPH_CLOSE, kernel2, cv::Point(-1, -1), 2);
//
//    cv::morphologyEx(mask, bufferMask, cv::MORPH_ERODE, kernel2);
//    cv::threshold(mask, bufferMask, 200, 255, cv::THRESH_BINARY);
//    mask = bufferMask;
//    mask.convertTo(mask, CV_8UC1);
//    cv::bitwise_not(mask, mask);


    cv::GaussianBlur(buffer, out, cv::Size(5,5), 0);
    if (showPics)
    {
        cv::imshow("Blur After " + fingerprintImageLocalPath, out);
        waitForSpace();
    }
    buffer = out;

    buffer.convertTo(buffer, CV_8UC1);
    if (showPics)
    {
        cv::imshow("Conv " + fingerprintImageLocalPath, buffer);
        waitForSpace();
    }

    cv::adaptiveThreshold(buffer, out, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 9, 2);
    buffer = out;
    if (showPics)
    {
        cv::imshow("Threshold2 " + fingerprintImageLocalPath, buffer);
        waitForSpace();
    }

    cv::ximgproc::thinning(buffer, out, cv::ximgproc::ThinningTypes::THINNING_ZHANGSUEN);
    cv::bitwise_not(out, out);
    if (showPics)
    {
        cv::imshow("Thinned2 " + fingerprintImageLocalPath, out);
        waitForSpace();
    }

    buffer = out + mask;
    if (showPics)
    {
        cv::imshow("Masked2 " + fingerprintImageLocalPath, buffer);
        waitForSpace();
    }
    out = buffer;

    auto minutiaes = retrieveMinutiaes(out);

    cv::Mat imgCopy;
    out.copyTo(imgCopy);
    printMinutaes(minutiaes, out);

    if (showPics)
    {
        cv::imshow("Minutaes " + fingerprintImageLocalPath, out);
        waitForSpace();
    }

    validateMinutaes(minutiaes, imgCopy);

    printMinutaes(minutiaes, imgCopy);

    if (showPics)
    {
        cv::imshow("Minutaes after validation " + fingerprintImageLocalPath, imgCopy);
        waitForSpace();
    }

    auto minutaeTemplate = getMinutaes(minutiaes, imgCopy);
    if (showPics)
    {
        cv::imshow("Minutaes template " + fingerprintImageLocalPath, minutaeTemplate);
        waitForSpace();
    }

    return minutiaes;
}

//int hough()
//{
//    return score
//}

int toDegrees(MinutiaeDirection dir)
{
    switch (dir) {
        case MinutiaeDirection::Vertical:
            return 90;
        case MinutiaeDirection::Horizontal:
            return 0;
        case MinutiaeDirection::DiagonalIncreasing:
            return 45;
        case MinutiaeDirection::DiagonalDecreasing:
            return 135;
        default:
            throw std::runtime_error("unsupported dir");
    }
}

int main()
{
    const bool showPics = false;
    const auto patternMinutaes = minutaesFromFingerprint("101_1.tif", showPics);
    const auto minutaesToCheck = minutaesFromFingerprint("102_1.tif", showPics);

    // @see https://github.com/opencv/opencv/blob/05b15943d6a42c99e5f921b7dbaa8323f3c042c6/samples/gpu/generalized_hough.cpp
    // @see http://amroamroamro.github.io/mexopencv/opencv/generalized_hough_demo.html
    // @see https://www.researchgate.net/profile/Fayyaz-Ul-Amir-Afsar-Minhas/publication/320076836_FINGERPRINT_BASED_PERSON_IDENTIFICATION_AND_VERIFICATION/links/59ccae0b45851556e9878a4a/FINGERPRINT-BASED-PERSON-IDENTIFICATION-AND-VERIFICATION.pdf?origin=publication_detail

    std::array<int, 4> possibleThetas{0, 45, 90, 135};
    std::vector<float> possibleScales;
    possibleScales.reserve(K);
    float scale = 0.5;
    for (int k=0; k<K; k++)
    {
        possibleScales.push_back(scale);
        scale += 0.01;
    }

    int maxVotes = 0;
    for (const auto& p : patternMinutaes)
    {
        const auto pPos = std::get<0>(p);
        const auto alpha = toDegrees(std::get<2>(p));
        for (const auto& q : minutaesToCheck)
        {
            const auto beta = toDegrees(std::get<2>(q));
            const auto qPos = std::get<0>(q);
            for(int l = 0; l < possibleThetas.size(); l++)
            {
                const auto theta = possibleThetas[l];
                if (alpha + theta == beta)
                {
                    for (int k = 0; k < possibleScales.size(); k++)
                    {
                        const float s = possibleScales[k];
                        const int deltaX = qPos.x - s * (cos(theta) * pPos.x + sin(theta) * pPos.y);
                        const int deltaY = qPos.y - s * (-sin(theta) * pPos.x + cos(theta) * pPos.y);
                        const int m = deltaX;
                        const int n = deltaY;
                        if (n >= 0 && n < N && m >= 0 && m < M)
                        {
                            A[k][l][m][n] += 1;
                            if (A[k][l][m][n] > maxVotes)
                            {
                                maxVotes = A[k][l][m][n];
                            }
                        }
                    }
                }
            }
        }
    }

    if ((float)maxVotes / (float)minutaesToCheck.size() >= 0.04f) {
        std::cout << "Matched!\n";
    }
    else
    {
        std::cout << "Not matched!\n";
    }
	return 0;
}

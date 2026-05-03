#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iomanip>
#include <iostream>

#include "backend/Module1_Thresholding/Thresholding.h"

namespace fs = std::filesystem;

static cv::Mat makeBimodalImage()
{
    cv::Mat img(200, 200, CV_8UC1, cv::Scalar(0));
    img.colRange(0, 100).setTo(60);
    img.colRange(100, 200).setTo(180);
    return img;
}

static cv::Mat makeNoisyImage(const cv::Mat& base)
{
    cv::Mat noise(base.size(), CV_16SC1);
    cv::randn(noise, 0, 20);

    cv::Mat base16;
    base.convertTo(base16, CV_16SC1);

    cv::Mat noisy16 = base16 + noise;
    cv::Mat noisy8;
    noisy16.convertTo(noisy8, CV_8UC1);
    return noisy8;
}

static cv::Mat makeOverlapImage()
{
    cv::Mat img(200, 200, CV_8UC1);
    cv::RNG rng(12345);

    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            if (j < 100)
                img.at<uchar>(i, j) = static_cast<uchar>(rng.uniform(70, 130));
            else
                img.at<uchar>(i, j) = static_cast<uchar>(rng.uniform(110, 170));
        }
    }

    return img;
}

static void saveImage(const fs::path& path, const cv::Mat& img)
{
    cv::imwrite(path.string(), img);
}

static void report(const std::string& name, const cv::Mat& img)
{
    auto optimal = Thresholding::optimalThreshold(img, false);
    auto otsu = Thresholding::otsuThreshold(img, false);

    std::cout << name << '\n';
    std::cout << "  optimal: threshold=" << std::fixed << std::setprecision(2)
              << optimal.threshold << " iterations=" << optimal.iterations
              << " time_ms=" << optimal.timingMs << '\n';
    std::cout << "  otsu   : threshold=" << std::fixed << std::setprecision(2)
              << otsu.threshold
              << " time_ms=" << otsu.timingMs << '\n';
    std::cout << "  same?  : " << ((std::abs(optimal.threshold - otsu.threshold) < 1.0) ? "close" : "different") << '\n';
}

int main()
{
    fs::path outDir = fs::current_path() / "threshold_probe_output";
    fs::create_directories(outDir);

    cv::Mat img1 = makeBimodalImage();
    cv::Mat img2 = makeNoisyImage(img1);
    cv::Mat img3 = makeOverlapImage();

    saveImage(outDir / "bimodal.png", img1);
    saveImage(outDir / "noisy.png", img2);
    saveImage(outDir / "overlap.png", img3);

    report("bimodal", img1);
    report("noisy", img2);
    report("overlap", img3);

    std::cout << "saved images to: " << outDir.string() << '\n';
    return 0;
}
#pragma once
#include <opencv2/core.hpp>

class Clustering {
public:
    struct Result {
        cv::Mat clustered;
        double timingMs;
        int numClusters; 
    };

    // 1. Basic Spatial Local Thresholding (Mean-based)
    static Result localThresholding(const cv::Mat& src, int windowSize);

    // 2. Region Growing (from scratch using a queue-based flood fill)
    static Result regionGrowing(const cv::Mat& src, double tolerance);

    // 3. K-Means Clustering (from scratch)
    static Result kMeans(const cv::Mat& src, int k, int maxIter = 50);
};
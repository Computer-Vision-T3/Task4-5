#include "Clustering.h"
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <queue>
#include <random>
#include <numeric>

Clustering::Result Clustering::localThresholding(const cv::Mat& src, int windowSize) {
    auto t0 = std::chrono::high_resolution_clock::now();
    
    cv::Mat gray;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else gray = src.clone();

    cv::Mat out = cv::Mat::zeros(gray.size(), CV_8UC1);
    if (windowSize % 2 == 0) windowSize++; // Ensure odd window
    int half = windowSize / 2;

    for (int r = 0; r < gray.rows; ++r) {
        for (int c = 0; c < gray.cols; ++c) {
            int rStart = std::max(0, r - half);
            int rEnd = std::min(gray.rows - 1, r + half);
            int cStart = std::max(0, c - half);
            int cEnd = std::min(gray.cols - 1, c + half);

            double sum = 0;
            int count = 0;
            for (int i = rStart; i <= rEnd; ++i) {
                for (int j = cStart; j <= cEnd; ++j) {
                    sum += gray.at<uchar>(i, j);
                    count++;
                }
            }
            double mean = sum / count;
            out.at<uchar>(r, c) = (gray.at<uchar>(r, c) > mean) ? 255 : 0;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return { out, std::chrono::duration<double, std::milli>(t1 - t0).count(), 2 };
}

Clustering::Result Clustering::regionGrowing(const cv::Mat& src, double tolerance) {
    auto t0 = std::chrono::high_resolution_clock::now();

    cv::Mat workImg;
    if (src.channels() == 1) cv::cvtColor(src, workImg, cv::COLOR_GRAY2BGR);
    else workImg = src.clone();

    cv::Mat out = cv::Mat::zeros(workImg.size(), CV_8UC3);
    cv::Mat visited = cv::Mat::zeros(workImg.size(), CV_8UC1);

    int numClusters = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(50, 255);

    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    for (int r = 0; r < workImg.rows; ++r) {
        for (int c = 0; c < workImg.cols; ++c) {
            if (visited.at<uchar>(r, c)) continue;

            // Start a new region
            numClusters++;
            cv::Vec3b regionColor(dis(gen), dis(gen), dis(gen)); // Random color per region
            cv::Vec3b seedVal = workImg.at<cv::Vec3b>(r, c);

            std::queue<cv::Point> q;
            q.push(cv::Point(c, r));
            visited.at<uchar>(r, c) = 1;

            while (!q.empty()) {
                cv::Point p = q.front();
                q.pop();
                out.at<cv::Vec3b>(p.y, p.x) = regionColor;

                for (int i = 0; i < 4; ++i) {
                    int nx = p.x + dx[i];
                    int ny = p.y + dy[i];

                    if (nx >= 0 && nx < workImg.cols && ny >= 0 && ny < workImg.rows) {
                        if (!visited.at<uchar>(ny, nx)) {
                            cv::Vec3b neighborVal = workImg.at<cv::Vec3b>(ny, nx);
                            double dist = std::sqrt(std::pow(neighborVal[0] - seedVal[0], 2) +
                                                    std::pow(neighborVal[1] - seedVal[1], 2) +
                                                    std::pow(neighborVal[2] - seedVal[2], 2));
                            if (dist <= tolerance) {
                                visited.at<uchar>(ny, nx) = 1;
                                q.push(cv::Point(nx, ny));
                            }
                        }
                    }
                }
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return { out, std::chrono::duration<double, std::milli>(t1 - t0).count(), numClusters };
}

Clustering::Result Clustering::kMeans(const cv::Mat& src, int k, int maxIter) {
    auto t0 = std::chrono::high_resolution_clock::now();

    cv::Mat workImg;
    if (src.channels() == 1) cv::cvtColor(src, workImg, cv::COLOR_GRAY2BGR);
    else workImg = src.clone();

    // Flatten image into data points
    int nPoints = workImg.rows * workImg.cols;
    std::vector<cv::Vec3f> points(nPoints);
    for (int r = 0; r < workImg.rows; ++r) {
        for (int c = 0; c < workImg.cols; ++c) {
            cv::Vec3b p = workImg.at<cv::Vec3b>(r, c);
            points[r * workImg.cols + c] = cv::Vec3f(p[0], p[1], p[2]);
        }
    }

    // Initialize centers randomly from existing points
    std::vector<cv::Vec3f> centers(k);
    std::mt19937 rng(42); 
    std::uniform_int_distribution<int> dist(0, nPoints - 1);
    for (int i = 0; i < k; ++i) centers[i] = points[dist(rng)];

    std::vector<int> labels(nPoints, 0);
    bool changed = true;

    for (int iter = 0; iter < maxIter && changed; ++iter) {
        changed = false;
        std::vector<cv::Vec3f> newCenters(k, cv::Vec3f(0, 0, 0));
        std::vector<int> counts(k, 0);

        // Assign points to nearest center
        for (int i = 0; i < nPoints; ++i) {
            float minDist = std::numeric_limits<float>::max();
            int bestCluster = 0;
            for (int j = 0; j < k; ++j) {
                cv::Vec3f diff = points[i] - centers[j];
                float d2 = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2];
                if (d2 < minDist) {
                    minDist = d2;
                    bestCluster = j;
                }
            }
            if (labels[i] != bestCluster) {
                labels[i] = bestCluster;
                changed = true;
            }
            newCenters[bestCluster] += points[i];
            counts[bestCluster]++;
        }

        // Update centers
        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                centers[j] = cv::Vec3f(newCenters[j][0] / counts[j], 
                                       newCenters[j][1] / counts[j], 
                                       newCenters[j][2] / counts[j]);
            }
        }
    }

    // Reconstruct Image
    cv::Mat out(workImg.size(), CV_8UC3);
    for (int i = 0; i < nPoints; ++i) {
        int r = i / workImg.cols;
        int c = i % workImg.cols;
        cv::Vec3f color = centers[labels[i]];
        out.at<cv::Vec3b>(r, c) = cv::Vec3b((uchar)color[0], (uchar)color[1], (uchar)color[2]);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return { out, std::chrono::duration<double, std::milli>(t1 - t0).count(), k };
}
#include "Thresholding.h"

#include <opencv2/imgproc.hpp>   // cv::cvtColor, cv::split, cv::merge  (NOT threshold)
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

// ═══════════════════════════════════════════════════════════════════════════════
//  PRIVATE HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

std::vector<double> Thresholding::buildHistogram(const cv::Mat& gray)
{
    CV_Assert(gray.type() == CV_8UC1);
    std::vector<double> hist(256, 0.0);
    const int total = gray.rows * gray.cols;
    for (int r = 0; r < gray.rows; ++r)
    {
        const uchar* row = gray.ptr<uchar>(r);
        for (int c = 0; c < gray.cols; ++c)
            hist[row[c]] += 1.0;
    }
    for (auto& v : hist) v /= total;
    return hist;
}

std::vector<double> Thresholding::smoothHistogram(const std::vector<double>& hist,
                                                   double sigma)
{
    // Build a small Gaussian kernel
    int half = static_cast<int>(std::ceil(3.0 * sigma));
    int ksize = 2 * half + 1;
    std::vector<double> kernel(ksize);
    double sum = 0.0;
    for (int i = 0; i < ksize; ++i)
    {
        double x = i - half;
        kernel[i] = std::exp(-(x * x) / (2.0 * sigma * sigma));
        sum += kernel[i];
    }
    for (auto& k : kernel) k /= sum;

    // Convolve histogram with kernel (mirror padding)
    std::vector<double> out(256, 0.0);
    for (int i = 0; i < 256; ++i)
    {
        double acc = 0.0;
        for (int j = 0; j < ksize; ++j)
        {
            int idx = i - half + j;
            // mirror padding
            if (idx < 0)   idx = -idx;
            if (idx > 255) idx = 510 - idx;
            idx = std::clamp(idx, 0, 255);
            acc += kernel[j] * hist[idx];
        }
        out[i] = acc;
    }
    return out;
}

std::vector<int> Thresholding::findPeaks(const std::vector<double>& hist)
{
    std::vector<int> peaks;
    for (int i = 1; i < 255; ++i)
        if (hist[i] > hist[i - 1] && hist[i] > hist[i + 1])
            peaks.push_back(i);
    return peaks;
}

int Thresholding::findValley(const std::vector<double>& hist, int left, int right)
{
    int minIdx = left + 1;
    double minVal = hist[minIdx];
    for (int i = left + 1; i < right; ++i)
    {
        if (hist[i] < minVal) { minVal = hist[i]; minIdx = i; }
    }
    return minIdx;
}

cv::Mat Thresholding::applyThreshold(const cv::Mat& gray, double thresh)
{
    cv::Mat out(gray.size(), CV_8UC1);
    uchar t = static_cast<uchar>(std::clamp(thresh, 0.0, 255.0));
    for (int r = 0; r < gray.rows; ++r)
    {
        const uchar* src = gray.ptr<uchar>(r);
        uchar*       dst = out.ptr<uchar>(r);
        for (int c = 0; c < gray.cols; ++c)
            dst[c] = (src[c] > t) ? 255 : 0;
    }
    return out;
}

cv::Mat Thresholding::applyThresholdColor(const cv::Mat& bgr,
                                           const std::vector<double>& thresholds)
{
    std::vector<cv::Mat> channels(3);
    cv::split(bgr, channels);
    std::vector<cv::Mat> binChannels(3);
    for (int ch = 0; ch < 3; ++ch)
        binChannels[ch] = applyThreshold(channels[ch],
                                          thresholds.size() > (size_t)ch
                                              ? thresholds[ch]
                                              : thresholds[0]);
    cv::Mat merged;
    cv::merge(binChannels, merged);
    return merged;
}

double Thresholding::computeOptimal(const cv::Mat& gray, int& outIter)
{
    // Initial threshold = mean of (min + max) pixel values
    double minVal, maxVal;
    // Manual min/max to avoid builtins
    minVal = 255; maxVal = 0;
    for (int r = 0; r < gray.rows; ++r)
    {
        const uchar* row = gray.ptr<uchar>(r);
        for (int c = 0; c < gray.cols; ++c)
        {
            if (row[c] < minVal) minVal = row[c];
            if (row[c] > maxVal) maxVal = row[c];
        }
    }
    double T = (minVal + maxVal) / 2.0;

    outIter = 0;
    while (true)
    {
        ++outIter;
        double sumBg = 0, cntBg = 0;
        double sumFg = 0, cntFg = 0;
        for (int r = 0; r < gray.rows; ++r)
        {
            const uchar* row = gray.ptr<uchar>(r);
            for (int c = 0; c < gray.cols; ++c)
            {
                double v = row[c];
                if (v <= T) { sumBg += v; cntBg += 1; }
                else        { sumFg += v; cntFg += 1; }
            }
        }
        double meanBg = (cntBg > 0) ? sumBg / cntBg : 0.0;
        double meanFg = (cntFg > 0) ? sumFg / cntFg : 255.0;
        double newT   = (meanBg + meanFg) / 2.0;
        if (std::fabs(newT - T) < 0.5) { T = newT; break; }
        T = newT;
        if (outIter > 200) break; // safety
    }
    return T;
}

double Thresholding::computeOtsu(const cv::Mat& gray)
{
    auto hist = buildHistogram(gray);

    double bestVar = -1.0;
    double bestT   = 0.0;
    double totalMean = 0.0;
    for (int i = 0; i < 256; ++i) totalMean += i * hist[i];

    double w0 = 0.0, mean0 = 0.0;
    for (int t = 0; t < 255; ++t)
    {
        w0     += hist[t];
        mean0  += t * hist[t];
        double w1 = 1.0 - w0;
        if (w0 < 1e-10 || w1 < 1e-10) continue;
        double m0  = mean0 / w0;
        double m1  = (totalMean - mean0) / w1;
        double var = w0 * w1 * (m0 - m1) * (m0 - m1);
        if (var > bestVar) { bestVar = var; bestT = t; }
    }
    return bestT;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════════

// ── Optimal ──────────────────────────────────────────────────────────────────
Thresholding::Result Thresholding::optimalThreshold(const cv::Mat& src,
                                                     bool applyToColor)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    Result res;

    if (src.channels() == 3 && applyToColor)
    {
        std::vector<cv::Mat> channels(3);
        cv::split(src, channels);
        std::vector<double> thresholds(3);
        int iters = 0;
        for (int ch = 0; ch < 3; ++ch)
            thresholds[ch] = computeOptimal(channels[ch], iters);
        res.thresholds = thresholds;
        res.threshold  = thresholds[1]; // Green channel representative
        res.iterations = iters;
        res.binary     = applyThresholdColor(src, thresholds);
    }
    else
    {
        cv::Mat gray;
        if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        else                     gray = src.clone();

        int iters = 0;
        double T  = computeOptimal(gray, iters);
        res.threshold  = T;
        res.thresholds = {T};
        res.iterations = iters;
        res.binary     = applyThreshold(gray, T);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.timingMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

// ── Otsu ─────────────────────────────────────────────────────────────────────
Thresholding::Result Thresholding::otsuThreshold(const cv::Mat& src,
                                                  bool applyToColor)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    Result res;
    res.iterations = 1;

    if (src.channels() == 3 && applyToColor)
    {
        std::vector<cv::Mat> channels(3);
        cv::split(src, channels);
        std::vector<double> thresholds(3);
        for (int ch = 0; ch < 3; ++ch)
            thresholds[ch] = computeOtsu(channels[ch]);
        res.thresholds = thresholds;
        res.threshold  = thresholds[1];
        res.binary     = applyThresholdColor(src, thresholds);
    }
    else
    {
        cv::Mat gray;
        if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        else                     gray = src.clone();

        double T       = computeOtsu(gray);
        res.threshold  = T;
        res.thresholds = {T};
        res.binary     = applyThreshold(gray, T);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.timingMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

// ── Spectral ──────────────────────────────────────────────────────────────────
Thresholding::Result Thresholding::spectralThreshold(const cv::Mat& src,
                                                      bool applyToColor,
                                                      double smoothSigma)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    Result res;
    res.iterations = 1;

    // Helper lambda: compute spectral thresholds for a single-channel image
    auto computeSpectral = [&](const cv::Mat& gray) -> std::vector<double>
    {
        auto hist    = buildHistogram(gray);
        auto smoothH = smoothHistogram(hist, smoothSigma);
        auto peaks   = findPeaks(smoothH);

        // Need at least 2 peaks to place a threshold
        if (peaks.size() < 2)
        {
            // Fallback to Otsu if histogram is not multi-modal
            return {computeOtsu(gray)};
        }

        std::vector<double> thresholds;
        for (size_t i = 0; i + 1 < peaks.size(); ++i)
        {
            int valley = findValley(smoothH, peaks[i], peaks[i + 1]);
            thresholds.push_back(static_cast<double>(valley));
        }
        return thresholds;
    };

    // Apply multi-threshold binarisation:
    // For N thresholds T1 < T2 < ... < TN:
    //   pixel < T1     → 0
    //   T1 ≤ pixel < T2→ intensity_step (e.g. 85 for 2 thresholds → 3 levels)
    //   ...
    //   pixel ≥ TN     → 255
    auto multiThreshApply = [](const cv::Mat& gray,
                                const std::vector<double>& thresholds) -> cv::Mat
    {
        int n = static_cast<int>(thresholds.size()); // number of thresholds
        int levels = n + 1;
        cv::Mat out(gray.size(), CV_8UC1);
        for (int r = 0; r < gray.rows; ++r)
        {
            const uchar* src = gray.ptr<uchar>(r);
            uchar*       dst = out.ptr<uchar>(r);
            for (int c = 0; c < gray.cols; ++c)
            {
                double v = src[c];
                int level = 0;
                for (int t = 0; t < n; ++t)
                    if (v >= thresholds[t]) level = t + 1;
                // map level to 0..255
                dst[c] = static_cast<uchar>(
                    std::round(255.0 * level / (levels - 1)));
            }
        }
        return out;
    };

    if (src.channels() == 3 && applyToColor)
    {
        std::vector<cv::Mat> channels(3);
        cv::split(src, channels);
        // Use green channel for valley detection (representative)
        auto thresholds = computeSpectral(channels[1]);

        // Apply same thresholds to all channels
        std::vector<cv::Mat> outChannels(3);
        for (int ch = 0; ch < 3; ++ch)
            outChannels[ch] = multiThreshApply(channels[ch], thresholds);

        cv::merge(outChannels, res.binary);
        res.thresholds = thresholds;
        res.threshold  = thresholds[0];
    }
    else
    {
        cv::Mat gray;
        if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        else                     gray = src.clone();

        auto thresholds = computeSpectral(gray);
        res.binary      = multiThreshApply(gray, thresholds);
        res.thresholds  = thresholds;
        res.threshold   = thresholds[0];
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.timingMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

// ── Local (Adaptive) ─────────────────────────────────────────────────────────
Thresholding::Result Thresholding::localThreshold(const cv::Mat& src,
                                                   int windowSize,
                                                   int method)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    Result res;
    res.iterations = 1;

    // Ensure windowSize is odd and at least 3
    if (windowSize < 3)  windowSize = 3;
    if (windowSize % 2 == 0) windowSize += 1;

    // Convert to grayscale
    cv::Mat gray;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else                     gray = src.clone();

    // Reduce high-frequency noise before local thresholding
    cv::GaussianBlur(gray, gray, cv::Size(5,5), 1.0);

    cv::Mat out = cv::Mat::zeros(gray.size(), CV_8UC1);

    int halfW = windowSize / 2;
    double sumThresh = 0.0;
    int tileCount    = 0;

    for (int r = 0; r < gray.rows; r += windowSize)
    {
        for (int c = 0; c < gray.cols; c += windowSize)
        {
            // Tile boundaries (clamp to image)
            int r2 = std::min(r + windowSize, gray.rows);
            int c2 = std::min(c + windowSize, gray.cols);
            cv::Mat tile = gray(cv::Rect(c, r, c2 - c, r2 - r));

            // Skip tiles that are too small or uniform
            if (tile.rows < 2 || tile.cols < 2) continue;

            double T = 0.0;
            int dummy = 0;
            if (method == 1) T = computeOtsu(tile);
            else             T = computeOptimal(tile, dummy);

            sumThresh += T;
            ++tileCount;

            // Apply threshold within tile
            for (int tr = 0; tr < tile.rows; ++tr)
            {
                const uchar* tSrc = tile.ptr<uchar>(tr);
                uchar*       tDst = out.ptr<uchar>(r + tr) + c;
                for (int tc = 0; tc < tile.cols; ++tc)
                    tDst[tc] = (tSrc[tc] > static_cast<uchar>(T)) ? 255 : 0;
            }
        }
    }

    res.binary     = out;
    res.threshold  = (tileCount > 0) ? sumThresh / tileCount : 128.0;
    res.thresholds = {res.threshold};

    auto t1 = std::chrono::high_resolution_clock::now();
    res.timingMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}
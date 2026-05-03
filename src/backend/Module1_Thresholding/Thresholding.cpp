#include "Thresholding.h"

#include <opencv2/imgproc.hpp>
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
    int half  = static_cast<int>(std::ceil(3.0 * sigma));
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

    std::vector<double> out(256, 0.0);
    for (int i = 0; i < 256; ++i)
    {
        double acc = 0.0;
        for (int j = 0; j < ksize; ++j)
        {
            int idx = i - half + j;
            if (idx < 0)   idx = -idx;
            if (idx > 255) idx = 510 - idx;
            idx = std::clamp(idx, 0, 255);
            acc += kernel[j] * hist[idx];
        }
        out[i] = acc;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  BUG 3 FIXED: was returning every tiny local bump as a "peak".
//  Fix: only keep peaks whose height exceeds 2× the mean histogram density.
//  This filters out noise bumps caused by quantisation and image noise.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<int> Thresholding::findPeaks(const std::vector<double>& hist)
{
    // mean bin density = 1/256 (since histogram is normalised).
    // A real mode must contain meaningfully more pixels than an average bin.
    double meanVal = 0.0;
    for (double v : hist) meanVal += v;
    meanVal /= 256.0;
    double minProminence = meanVal * 2.0;  // peaks must be at least 2× average

    std::vector<int> peaks;
    for (int i = 1; i < 255; ++i)
    {
        if (hist[i] > hist[i - 1] &&
            hist[i] > hist[i + 1] &&
            hist[i] > minProminence)
        {
            peaks.push_back(i);
        }
    }
    return peaks;
}

int Thresholding::findValley(const std::vector<double>& hist, int left, int right)
{
    int minIdx    = left + 1;
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

// ─────────────────────────────────────────────────────────────────────────────
//  BUG 1 FIXED (two sub-issues):
//
//  Sub-issue A — wrong init:
//    Old: T = (minVal + maxVal) / 2
//    Problem: minVal is almost always 0 (black border in medical images) and
//    maxVal is ~255. So T always starts at ~127, far from the true background.
//    Fix: T = mean pixel value — stays close to the actual image centre of mass.
//
//  Sub-issue B — wrong convergence:
//    Old:  if (|newT - T| < 0.5) { T = newT; break; }
//    Problem: overwrites T with newT before breaking, so the returned value is
//    the last newT (may oscillate between two values ±0.49).
//    Fix:  if (|newT - T| < 0.5) break;   // keep T, which is already settled
// ─────────────────────────────────────────────────────────────────────────────
double Thresholding::computeOptimal(const cv::Mat& gray, int& outIter)
{
    // Mean-based initial threshold (robust for images with large dark borders)
    double totalSum = 0.0;
    const int totalPix = gray.rows * gray.cols;
    for (int r = 0; r < gray.rows; ++r)
    {
        const uchar* row = gray.ptr<uchar>(r);
        for (int c = 0; c < gray.cols; ++c)
            totalSum += row[c];
    }
    double T = totalSum / totalPix;

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

        if (std::fabs(newT - T) < 0.5) break;  // BUG FIX: break WITHOUT overwriting T
        T = newT;
        if (outIter > 200) break;
    }
    return T;
}

// ─────────────────────────────────────────────────────────────────────────────
//  BUG 2 FIXED:
//    Old: bestT = t    (stores split index, off-by-one)
//    Fix: bestT = t+1  (pixels ABOVE t+1 are foreground — matches applyThreshold)
//
//    The loop range t=0..255 is correct; t=255 has empty class-1 and is
//    filtered by the w1<1e-10 guard, so no out-of-bounds risk.
// ─────────────────────────────────────────────────────────────────────────────
double Thresholding::computeOtsu(const cv::Mat& gray)
{
    auto hist = buildHistogram(gray);

    double bestVar   = -1.0;
    double bestT     = 128.0;
    double totalMean = 0.0;
    for (int i = 0; i < 256; ++i) totalMean += i * hist[i];

    double w0 = 0.0, mean0 = 0.0;
    for (int t = 0; t < 256; ++t)
    {
        w0    += hist[t];
        mean0 += t * hist[t];
        double w1 = 1.0 - w0;
        if (w0 < 1e-10 || w1 < 1e-10) continue;
        double m0  = mean0 / w0;
        double m1  = (totalMean - mean0) / w1;
        double var = w0 * w1 * (m0 - m1) * (m0 - m1);
        if (var > bestVar)
        {
            bestVar = var;
            bestT   = t + 1;  // BUG FIX: was `t`, now `t+1`
        }
    }
    return bestT;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  PUBLIC API  (unchanged structure, bugs fixed in helpers above)
// ═══════════════════════════════════════════════════════════════════════════════

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
        res.threshold  = thresholds[1];
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

Thresholding::Result Thresholding::spectralThreshold(const cv::Mat& src,
                                                      bool applyToColor,
                                                      double smoothSigma)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    Result res;
    res.iterations = 1;

    auto computeSpectral = [&](const cv::Mat& gray) -> std::vector<double>
    {
        auto hist    = buildHistogram(gray);
        auto smoothH = smoothHistogram(hist, smoothSigma);
        auto peaks   = findPeaks(smoothH);  // BUG 3 FIXED: only real peaks returned

        if (peaks.size() < 2)
            return {computeOtsu(gray)};

        std::vector<double> thresholds;
        for (size_t i = 0; i + 1 < peaks.size(); ++i)
        {
            int valley = findValley(smoothH, peaks[i], peaks[i + 1]);
            thresholds.push_back(static_cast<double>(valley));
        }
        return thresholds;
    };

    auto multiThreshApply = [](const cv::Mat& gray,
                                const std::vector<double>& thresholds) -> cv::Mat
    {
        int n      = static_cast<int>(thresholds.size());
        int levels = n + 1;
        cv::Mat out(gray.size(), CV_8UC1);
        for (int r = 0; r < gray.rows; ++r)
        {
            const uchar* src = gray.ptr<uchar>(r);
            uchar*       dst = out.ptr<uchar>(r);
            for (int c = 0; c < gray.cols; ++c)
            {
                double v     = src[c];
                int    level = 0;
                for (int t = 0; t < n; ++t)
                    if (v >= thresholds[t]) level = t + 1;
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
        auto thresholds = computeSpectral(channels[1]);
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

Thresholding::Result Thresholding::localThreshold(const cv::Mat& src,
                                                   int windowSize,
                                                   int method)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    Result res;
    res.iterations = 1;

    if (windowSize < 3)      windowSize = 3;
    if (windowSize % 2 == 0) windowSize += 1;

    cv::Mat gray;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else                     gray = src.clone();

    cv::Mat out = cv::Mat::zeros(gray.size(), CV_8UC1);

    double sumThresh = 0.0;
    int tileCount    = 0;

    for (int r = 0; r < gray.rows; r += windowSize)
    {
        for (int c = 0; c < gray.cols; c += windowSize)
        {
            int r2 = std::min(r + windowSize, gray.rows);
            int c2 = std::min(c + windowSize, gray.cols);
            cv::Mat tile = gray(cv::Rect(c, r, c2 - c, r2 - r));

            if (tile.rows < 2 || tile.cols < 2) continue;

            double T  = 0.0;
            int dummy = 0;
            if (method == 1) T = computeOtsu(tile);
            else             T = computeOptimal(tile, dummy);

            sumThresh += T;
            ++tileCount;

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
#pragma once
#include <opencv2/core.hpp>
#include <vector>

/**
 * Thresholding  –  Module 1
 * All algorithms implemented from scratch (no cv::threshold / cv::adaptiveThreshold).
 */
class Thresholding
{
public:
    // ── Result bundle ────────────────────────────────────────────────────────
    struct Result {
        cv::Mat   binary;        // 8-bit binary image (0 / 255)
        double    threshold;     // primary threshold used  (or first threshold)
        std::vector<double> thresholds; // all thresholds (spectral may have >1)
        double    timingMs;      // wall-clock time in milliseconds
        int       iterations;    // for iterative methods (Optimal)
    };

    // ── Global methods ───────────────────────────────────────────────────────

    /**
     * Optimal (Iterative) thresholding.
     * Starts at the mean of min+max, then iterates: split image into two
     * classes, compute each mean, average them as new threshold.
     * Converges when threshold stops changing.
     *
     * @param src       Grayscale or colour image.
     * @param applyToColor  If true AND src is colour, threshold each channel
     *                      independently; otherwise convert to grey first.
     */
    static Result optimalThreshold(const cv::Mat& src, bool applyToColor = false);

    /**
     * Otsu's thresholding.
     * Exhaustively maximises inter-class variance over the histogram.
     * Works on grayscale (or per-channel when applyToColor is true).
     */
    static Result otsuThreshold(const cv::Mat& src, bool applyToColor = false);

    /**
     * Spectral (multi-modal) thresholding.
     * Smooths the histogram, finds all local maxima (peaks), then places
     * thresholds at the valleys between consecutive peaks.
     * Handles images with >2 modes (more than 2 histogram peaks).
     *
     * @param smoothSigma   Gaussian sigma for histogram smoothing (default 3).
     */
    static Result spectralThreshold(const cv::Mat& src,
                                    bool applyToColor = false,
                                    double smoothSigma = 3.0);

    /**
     * Local (Adaptive) thresholding.
     * Divides the image into non-overlapping tiles of size windowSize×windowSize.
     * Within each tile the chosen global method (Optimal or Otsu) is applied.
     *
     * @param windowSize  Tile size (odd, ≥3).  Clamped internally if even.
     * @param method      0 = Optimal per-tile, 1 = Otsu per-tile.
     */
    static Result localThreshold(const cv::Mat& src,
                                 int windowSize = 11,
                                 int method = 0);

private:
    // ── Low-level helpers (grayscale only) ───────────────────────────────────

    /** Build a 256-bin normalised histogram from a single-channel 8-bit image. */
    static std::vector<double> buildHistogram(const cv::Mat& gray);

    /** Smooth histogram with a Gaussian kernel of given sigma. */
    static std::vector<double> smoothHistogram(const std::vector<double>& hist,
                                               double sigma);

    /** Find local-maxima indices in a histogram vector. */
    static std::vector<int> findPeaks(const std::vector<double>& hist);

    /** Find the valley (minimum) index between two peak positions. */
    static int findValley(const std::vector<double>& hist, int left, int right);

    /** Apply a single scalar threshold to a grayscale Mat → binary Mat. */
    static cv::Mat applyThreshold(const cv::Mat& gray, double thresh);

    /** Apply independent thresholds to each channel of a colour image. */
    static cv::Mat applyThresholdColor(const cv::Mat& bgr,
                                       const std::vector<double>& thresholds);

    /** Optimal threshold on a grayscale Mat; returns threshold value. */
    static double computeOptimal(const cv::Mat& gray, int& outIter);

    /** Otsu threshold on a grayscale Mat; returns threshold value. */
    static double computeOtsu(const cv::Mat& gray);
};
#pragma once
#include <opencv2/core.hpp>
#include <vector>

/**
 * Segmentation  –  Module 3  (Student C)
 * Implements Mean Shift and Agglomerative Clustering from scratch.
 * No cv::pyrMeanShiftFiltering or similar high-level OpenCV calls are used.
 */

namespace cv {
    using Vec5f = Vec<float, 5>;
}
class Segmentation
{
public:
    // ── Result bundle ────────────────────────────────────────────────────────
    struct Result {
        cv::Mat   segmented;      // 8-bit BGR segmented image (label colours)
        cv::Mat   labelMap;       // 32SC1 map: each pixel's cluster label
        int       numSegments;    // number of distinct segments found
        double    timingMs;       // wall-clock time in milliseconds
    };

    /**
     * Mean Shift Segmentation (from scratch).
     *
     * For each pixel, iteratively shifts a window in joint spatial-colour
     * space until convergence, then merges pixels whose final modes are
     * within a merge tolerance.
     *
     * @param src          Input image (grayscale or BGR).
     * @param spatialRad   Spatial bandwidth (hs) in pixels.
     * @param colorRad     Colour  bandwidth (hr) in intensity / colour units.
     * @param maxIter      Maximum shift iterations per pixel (default 50).
     * @param epsilon      Convergence threshold (default 1.0).
     */
    static Result meanShift(const cv::Mat& src,
                            double spatialRad = 10.0,
                            double colorRad   = 10.0,
                            int    maxIter    = 50,
                            double epsilon    = 1.0);

    /**
     * Agglomerative (Hierarchical) Clustering Segmentation (from scratch).
     *
     * Starts with a downsampled superpixel grid, then greedily merges the
     * two closest clusters by average colour distance until the merge
     * distance exceeds colorRad or no spatial neighbours remain.
     *
     * @param src          Input image (grayscale or BGR).
     * @param spatialRad   Controls initial superpixel grid spacing (in px).
     * @param colorRad     Maximum colour distance for merging clusters.
     */
    static Result agglomerative(const cv::Mat& src,
                                double spatialRad = 10.0,
                                double colorRad   = 20.0);

private:
    // ── Mean Shift helpers ───────────────────────────────────────────────────

    /** Convert BGR to a floating-point Lab-like [L,a,b] vector (simple). */
    static cv::Vec3f bgrToFloat(const cv::Vec3b& bgr);

    /** Colour distance squared between two float colour vectors. */
    static float colorDistSq(const cv::Vec3f& a, const cv::Vec3f& b);

    /** Run one mean-shift iteration; returns new (col, row, c0, c1, c2). */
    static void shiftOnce(const cv::Mat& floatImg,
                          float& x, float& y,
                          float& c0, float& c1, float& c2,
                          float hs2, float hr2);

    /** Assign colour labels from the mode map and build label image. */
    static Result buildResult(const cv::Mat& src,
                              const std::vector<cv::Vec5f>& modes,
                              double timingMs);

    // ── Agglomerative helpers ────────────────────────────────────────────────

    struct Cluster {
        cv::Vec3f   color;          // mean BGR (float)
        std::vector<int> members;   // pixel indices belonging to this cluster
        bool        active = true;
    };

    /** Colour distance between two clusters (average colour). */
    static float clusterDist(const Cluster& a, const Cluster& b);

    /** Paint each pixel with its cluster's mean colour. */
    static cv::Mat paintClusters(const cv::Mat& src,
                                 const std::vector<int>& labels,
                                 const std::vector<Cluster>& clusters);
};
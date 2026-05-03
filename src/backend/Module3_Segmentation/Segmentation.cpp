#include "Segmentation.h"

#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <limits>

// ═══════════════════════════════════════════════════════════════════════════════
//  PRIVATE HELPERS – Mean Shift
// ═══════════════════════════════════════════════════════════════════════════════

cv::Vec3f Segmentation::bgrToFloat(const cv::Vec3b& bgr)
{
    // Normalise to [0, 255] float – keeps the same scale as spatial coords
    // so colorRad is directly comparable to pixel intensity distance.
    return cv::Vec3f(static_cast<float>(bgr[0]),
                     static_cast<float>(bgr[1]),
                     static_cast<float>(bgr[2]));
}

float Segmentation::colorDistSq(const cv::Vec3f& a, const cv::Vec3f& b)
{
    float d0 = a[0] - b[0];
    float d1 = a[1] - b[1];
    float d2 = a[2] - b[2];
    return d0 * d0 + d1 * d1 + d2 * d2;
}

// ─────────────────────────────────────────────────────────────────────────────
//  shiftOnce
//  Performs a single mean-shift update in joint (x, y, c0, c1, c2) space.
//  Uses a flat kernel (Epanechnikov-style): all pixels inside the window
//  contribute equally.  This is cheaper than Gaussian and converges well.
// ─────────────────────────────────────────────────────────────────────────────
void Segmentation::shiftOnce(const cv::Mat& floatImg,
                              float& x, float& y,
                              float& c0, float& c1, float& c2,
                              float hs2, float hr2)
{
    double sumX  = 0, sumY  = 0;
    double sumC0 = 0, sumC1 = 0, sumC2 = 0;
    int    count = 0;

    int hs = static_cast<int>(std::sqrt(hs2));

    int r0 = std::max(0,              static_cast<int>(y) - hs);
    int r1 = std::min(floatImg.rows-1, static_cast<int>(y) + hs);
    int c_0 = std::max(0,              static_cast<int>(x) - hs);
    int c_1 = std::min(floatImg.cols-1, static_cast<int>(x) + hs);

    cv::Vec3f curColor(c0, c1, c2);

    for (int r = r0; r <= r1; ++r)
    {
        const cv::Vec3f* row = floatImg.ptr<cv::Vec3f>(r);
        for (int c = c_0; c <= c_1; ++c)
        {
            // Spatial distance check
            float dx = c - x, dy = r - y;
            if (dx * dx + dy * dy > hs2) continue;

            // Colour distance check
            if (colorDistSq(row[c], curColor) > hr2) continue;

            sumX  += c;
            sumY  += r;
            sumC0 += row[c][0];
            sumC1 += row[c][1];
            sumC2 += row[c][2];
            ++count;
        }
    }

    if (count > 0)
    {
        x  = static_cast<float>(sumX  / count);
        y  = static_cast<float>(sumY  / count);
        c0 = static_cast<float>(sumC0 / count);
        c1 = static_cast<float>(sumC1 / count);
        c2 = static_cast<float>(sumC2 / count);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  buildResult
//  Takes the per-pixel converged modes, merges modes that are close enough,
//  assigns integer labels, then paints the output image.
// ─────────────────────────────────────────────────────────────────────────────
Segmentation::Result Segmentation::buildResult(const cv::Mat& src,
                                               const std::vector<cv::Vec5f>& modes,
                                               double timingMs)
{
    const int rows = src.rows;
    const int cols = src.cols;
    const int N    = rows * cols;

    // ── Merge close modes ────────────────────────────────────────────────────
    // Two modes are in the same segment if their spatial AND colour distances
    // are both within half a bandwidth.  We do a simple union-find approach.
    std::vector<int> labels(N, -1);
    std::vector<cv::Vec5f> centroids;   // representative mode per label

    for (int i = 0; i < N; ++i)
    {
        const cv::Vec5f& m = modes[i];
        int assigned = -1;

        for (int k = 0; k < (int)centroids.size(); ++k)
        {
            const cv::Vec5f& ck = centroids[k];
            float dx = m[0] - ck[0], dy = m[1] - ck[1];
            float dc0 = m[2] - ck[2], dc1 = m[3] - ck[3], dc2 = m[4] - ck[4];
            float spatD2  = dx * dx + dy * dy;
            float colorD2 = dc0*dc0 + dc1*dc1 + dc2*dc2;

            // merge if close to an existing centroid
            if (spatD2 < 4.0f && colorD2 < 400.0f) {
                assigned = k;
                break;
            }
        }

        if (assigned == -1)
        {
            assigned = static_cast<int>(centroids.size());
            centroids.push_back(m);
        }
        labels[i] = assigned;
    }

    // ── Build label map ──────────────────────────────────────────────────────
    cv::Mat labelMap(rows, cols, CV_32SC1);
    for (int i = 0; i < N; ++i)
        labelMap.at<int>(i / cols, i % cols) = labels[i];

    // ── Build coloured output ────────────────────────────────────────────────
    // Each segment gets the average colour of its converged mode.
    int numSeg = static_cast<int>(centroids.size());
    std::vector<cv::Vec3b> palette(numSeg);
    for (int k = 0; k < numSeg; ++k)
    {
        palette[k] = cv::Vec3b(
            static_cast<uchar>(std::clamp(centroids[k][2], 0.f, 255.f)),
            static_cast<uchar>(std::clamp(centroids[k][3], 0.f, 255.f)),
            static_cast<uchar>(std::clamp(centroids[k][4], 0.f, 255.f))
        );
    }

    cv::Mat out(rows, cols, CV_8UC3);
    for (int r = 0; r < rows; ++r)
    {
        cv::Vec3b* outRow = out.ptr<cv::Vec3b>(r);
        const int* lblRow = labelMap.ptr<int>(r);
        for (int c = 0; c < cols; ++c)
            outRow[c] = palette[lblRow[c]];
    }

    // If src was grayscale, convert output to gray display
    if (src.channels() == 1)
        cv::cvtColor(out, out, cv::COLOR_BGR2GRAY);

    return { out, labelMap, numSeg, timingMs };
}

// ═══════════════════════════════════════════════════════════════════════════════
//  PUBLIC – Mean Shift
// ═══════════════════════════════════════════════════════════════════════════════
Segmentation::Result Segmentation::meanShift(const cv::Mat& src,
                                             double spatialRad,
                                             double colorRad,
                                             int    maxIter,
                                             double epsilon)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    // ── Prepare floating-point image ─────────────────────────────────────────
    cv::Mat workImg;
    if (src.channels() == 1)
        cv::cvtColor(src, workImg, cv::COLOR_GRAY2BGR);
    else
        workImg = src.clone();

    // Downsample for speed, process, then upsample labels back
    // Scale factor: limit working resolution to ~200px on longest side
    const int MAX_DIM = 200;
    float scale = 1.0f;
    if (std::max(workImg.rows, workImg.cols) > MAX_DIM)
        scale = static_cast<float>(MAX_DIM) / std::max(workImg.rows, workImg.cols);

    cv::Mat small;
    if (scale < 1.0f)
        cv::resize(workImg, small, cv::Size(), scale, scale, cv::INTER_AREA);
    else
        small = workImg.clone();

    // Convert to float Vec3f for arithmetic
    cv::Mat floatImg(small.rows, small.cols, CV_32FC3);
    for (int r = 0; r < small.rows; ++r)
        for (int c = 0; c < small.cols; ++c)
            floatImg.at<cv::Vec3f>(r, c) = bgrToFloat(small.at<cv::Vec3b>(r, c));

    const float hs  = static_cast<float>(spatialRad * scale);
    const float hr  = static_cast<float>(colorRad);
    const float hs2 = hs * hs;
    const float hr2 = hr * hr;
    const float eps2= static_cast<float>(epsilon * epsilon);

    const int rows = small.rows;
    const int cols = small.cols;
    const int N    = rows * cols;

    std::vector<cv::Vec5f> modes(N);   // (x, y, c0, c1, c2) per pixel

    // ── Per-pixel mean-shift ─────────────────────────────────────────────────
    for (int i = 0; i < N; ++i)
    {
        int r0 = i / cols;
        int c0 = i % cols;

        float x  = static_cast<float>(c0);
        float y  = static_cast<float>(r0);
        cv::Vec3f col = floatImg.at<cv::Vec3f>(r0, c0);
        float fc0 = col[0], fc1 = col[1], fc2 = col[2];

        for (int iter = 0; iter < maxIter; ++iter)
        {
            float px = x, py = y, pc0 = fc0, pc1 = fc1, pc2 = fc2;
            shiftOnce(floatImg, x, y, fc0, fc1, fc2, hs2, hr2);

            float dx = x - px, dy = y - py;
            float dc0 = fc0 - pc0, dc1 = fc1 - pc1, dc2 = fc2 - pc2;
            float shift2 = dx*dx + dy*dy + dc0*dc0 + dc1*dc1 + dc2*dc2;
            if (shift2 < eps2) break;
        }

        modes[i] = cv::Vec5f(x, y, fc0, fc1, fc2);
    }

    // ── Build result on small image ──────────────────────────────────────────
    // Create a temporary src-sized placeholder at small resolution
    cv::Mat smallSrc;
    if (src.channels() == 1)
        cv::cvtColor(small, smallSrc, cv::COLOR_BGR2GRAY);
    else
        smallSrc = small;

    Result smallResult = buildResult(smallSrc, modes,  0.0);

    // ── Upsample label map back to original size ─────────────────────────────
    cv::Mat finalSeg;
    if (scale < 1.0f)
        cv::resize(smallResult.segmented, finalSeg,
                   cv::Size(src.cols, src.rows), 0, 0, cv::INTER_NEAREST);
    else
        finalSeg = smallResult.segmented;

    cv::Mat finalLabels;
    if (scale < 1.0f)
        cv::resize(smallResult.labelMap, finalLabels,
                   cv::Size(src.cols, src.rows), 0, 0, cv::INTER_NEAREST);
    else
        finalLabels = smallResult.labelMap;

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return { finalSeg, finalLabels, smallResult.numSegments, ms };
}

// ═══════════════════════════════════════════════════════════════════════════════
//  PRIVATE HELPERS – Agglomerative
// ═══════════════════════════════════════════════════════════════════════════════

float Segmentation::clusterDist(const Cluster& a, const Cluster& b)
{
    float d0 = a.color[0] - b.color[0];
    float d1 = a.color[1] - b.color[1];
    float d2 = a.color[2] - b.color[2];
    return std::sqrt(d0*d0 + d1*d1 + d2*d2);
}

cv::Mat Segmentation::paintClusters(const cv::Mat& src,
                                    const std::vector<int>& labels,
                                    const std::vector<Cluster>& clusters)
{
    cv::Mat out(src.rows, src.cols, CV_8UC3);
    for (int r = 0; r < src.rows; ++r)
    {
        cv::Vec3b* outRow = out.ptr<cv::Vec3b>(r);
        for (int c = 0; c < src.cols; ++c)
        {
            int lbl = labels[r * src.cols + c];
            const cv::Vec3f& col = clusters[lbl].color;
            outRow[c] = cv::Vec3b(
                static_cast<uchar>(std::clamp(col[0], 0.f, 255.f)),
                static_cast<uchar>(std::clamp(col[1], 0.f, 255.f)),
                static_cast<uchar>(std::clamp(col[2], 0.f, 255.f))
            );
        }
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  PUBLIC – Agglomerative Clustering
// ═══════════════════════════════════════════════════════════════════════════════
Segmentation::Result Segmentation::agglomerative(const cv::Mat& src,
                                                  double spatialRad,
                                                  double colorRad)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    // ── Prepare image ────────────────────────────────────────────────────────
    cv::Mat workImg;
    if (src.channels() == 1)
        cv::cvtColor(src, workImg, cv::COLOR_GRAY2BGR);
    else
        workImg = src.clone();

    // Downsample to keep the algorithm tractable
    const int MAX_DIM = 150;
    float scale = 1.0f;
    if (std::max(workImg.rows, workImg.cols) > MAX_DIM)
        scale = static_cast<float>(MAX_DIM) / std::max(workImg.rows, workImg.cols);

    cv::Mat small;
    if (scale < 1.0f)
        cv::resize(workImg, small, cv::Size(), scale, scale, cv::INTER_AREA);
    else
        small = workImg.clone();

    const int rows = small.rows;
    const int cols = small.cols;
    const int N    = rows * cols;

    // ── Step 1: Superpixel grid initialisation ───────────────────────────────
    // Each pixel starts as its own micro-cluster, then we do a first pass
    // grouping into a grid of cells of size ~gridStep × gridStep.
    int gridStep = std::max(1, static_cast<int>(spatialRad * scale));

    std::vector<int> pixelLabel(N);     // pixel → cluster index
    std::vector<Cluster> clusters;

    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            int gridR = r / gridStep;
            int gridC = c / gridStep;
            int gridCols = (cols + gridStep - 1) / gridStep;
            int cellIdx  = gridR * gridCols + gridC;

            // Grow clusters vector on demand
            while (static_cast<int>(clusters.size()) <= cellIdx)
            {
                Cluster cl;
                cl.color  = cv::Vec3f(0, 0, 0);
                cl.active = true;
                clusters.push_back(cl);
            }

            int pixIdx = r * cols + c;
            cv::Vec3b bgr = small.at<cv::Vec3b>(r, c);
            clusters[cellIdx].color[0] += bgr[0];
            clusters[cellIdx].color[1] += bgr[1];
            clusters[cellIdx].color[2] += bgr[2];
            clusters[cellIdx].members.push_back(pixIdx);
            pixelLabel[pixIdx] = cellIdx;
        }
    }

    // Average the colours inside each initial cluster
    for (auto& cl : clusters)
    {
        if (!cl.members.empty())
        {
            float inv = 1.0f / cl.members.size();
            cl.color[0] *= inv;
            cl.color[1] *= inv;
            cl.color[2] *= inv;
        }
    }

    int K = static_cast<int>(clusters.size());

    // ── Step 2: Build spatial adjacency list ─────────────────────────────────
    // Two clusters are neighbours if any of their pixels are 4-connected.
    int gridCols = (cols + gridStep - 1) / gridStep;
    int gridRows = (rows + gridStep - 1) / gridStep;

    // For grid cells, cells (r,c) and (r,c+1) or (r+1,c) are neighbours.
    std::vector<std::vector<int>> adj(K);
    auto addAdj = [&](int a, int b) {
        if (a >= K || b >= K) return;
        // avoid duplicates (simple check)
        for (int x : adj[a]) if (x == b) return;
        adj[a].push_back(b);
        adj[b].push_back(a);
    };

    for (int gr = 0; gr < gridRows; ++gr)
        for (int gc = 0; gc < gridCols; ++gc)
        {
            int idx = gr * gridCols + gc;
            if (gc + 1 < gridCols) addAdj(idx, gr * gridCols + (gc + 1));
            if (gr + 1 < gridRows) addAdj(idx, (gr + 1) * gridCols + gc);
        }

    // ── Step 3: Greedy agglomerative merging ─────────────────────────────────
    // Repeatedly find the neighbouring pair with minimum colour distance,
    // merge if distance < colorRad, stop when no such pair exists.
    const float mergeThresh = static_cast<float>(colorRad);

    bool merged = true;
    while (merged)
    {
        merged = false;
        float  bestDist = std::numeric_limits<float>::max();
        int    bestA = -1, bestB = -1;

        for (int a = 0; a < K; ++a)
        {
            if (!clusters[a].active) continue;
            for (int b : adj[a])
            {
                if (!clusters[b].active || b <= a) continue;
                float d = clusterDist(clusters[a], clusters[b]);
                if (d < bestDist) { bestDist = d; bestA = a; bestB = b; }
            }
        }

        if (bestA == -1 || bestDist >= mergeThresh) break;

        // Merge bestB into bestA
        Cluster& ca = clusters[bestA];
        Cluster& cb = clusters[bestB];

        // Weighted average colour
        float na = static_cast<float>(ca.members.size());
        float nb = static_cast<float>(cb.members.size());
        float total = na + nb;
        ca.color[0] = (ca.color[0] * na + cb.color[0] * nb) / total;
        ca.color[1] = (ca.color[1] * na + cb.color[1] * nb) / total;
        ca.color[2] = (ca.color[2] * na + cb.color[2] * nb) / total;

        // Transfer members; update pixel labels
        for (int px : cb.members)
        {
            pixelLabel[px] = bestA;
            ca.members.push_back(px);
        }
        cb.members.clear();
        cb.active = false;

        // Merge adjacency lists: neighbours of B become neighbours of A
        for (int nb_idx : adj[bestB])
        {
            if (nb_idx == bestA || !clusters[nb_idx].active) continue;
            addAdj(bestA, nb_idx);
        }
        adj[bestB].clear();

        merged = true;
    }

    // ── Step 4: Re-index labels to 0..numSeg-1 ──────────────────────────────
    std::unordered_map<int, int> remap;
    int nextLabel = 0;
    for (int a = 0; a < K; ++a)
        if (clusters[a].active) remap[a] = nextLabel++;

    std::vector<Cluster> finalClusters(nextLabel);
    for (int a = 0; a < K; ++a)
        if (clusters[a].active) finalClusters[remap[a]] = clusters[a];

    std::vector<int> finalLabels(N);
    for (int i = 0; i < N; ++i)
        finalLabels[i] = remap[pixelLabel[i]];

    // ── Step 5: Paint output ─────────────────────────────────────────────────
    cv::Mat smallOut = paintClusters(small, finalLabels, finalClusters);

    // Build small label map
    cv::Mat smallLabelMap(rows, cols, CV_32SC1);
    for (int i = 0; i < N; ++i)
        smallLabelMap.at<int>(i / cols, i % cols) = finalLabels[i];

    // Upsample back to original size
    cv::Mat finalSeg, finalLabelMap;
    if (scale < 1.0f)
    {
        cv::resize(smallOut,      finalSeg,      cv::Size(src.cols, src.rows),
                   0, 0, cv::INTER_NEAREST);
        cv::resize(smallLabelMap, finalLabelMap, cv::Size(src.cols, src.rows),
                   0, 0, cv::INTER_NEAREST);
    }
    else
    {
        finalSeg      = smallOut;
        finalLabelMap = smallLabelMap;
    }

    // If original was grayscale, return grayscale
    if (src.channels() == 1)
        cv::cvtColor(finalSeg, finalSeg, cv::COLOR_BGR2GRAY);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return { finalSeg, finalLabelMap, nextLabel, ms };
}
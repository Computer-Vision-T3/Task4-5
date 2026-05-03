#include "AppController.h"
#include "../MainWindow.h"
#include "../components/TopTaskBar.h"
#include "../components/ImagePanel.h"
#include "../components/ParameterBox.h"
#include <QFileDialog>

// Backend Headers
#include "../../backend/Module1_Thresholding/Thresholding.h"
#include "../../backend/Module2_Clustering/Clustering.h"
#include "../../backend/Module3_Segmentation/Segmentation.h"
#include "../../backend/Module4_FaceDetection/FaceDetection.h"
#include "../../backend/Module5_FaceRecognition/FaceRecognition.h"

AppController::AppController(MainWindow *window, QObject *parent) : QObject(parent), m_window(window) {
    auto *bar = m_window->getTopTaskBar();
    connect(bar, &TopTaskBar::taskChanged, this, &AppController::handleTaskChange);
    connect(bar, &TopTaskBar::applyRequested, this, &AppController::handleApply);
    connect(bar, &TopTaskBar::clearRequested, this, &AppController::handleClear);
    connect(bar, &TopTaskBar::saveRequested, this, &AppController::handleSave);

    connect(m_window->getPanelA(), &ImagePanel::imageLoaded, this, [this](const cv::Mat &img) { m_state.setImageA(img); });
    connect(m_window->getPanelB(), &ImagePanel::imageLoaded, this, [this](const cv::Mat &img) { m_state.setImageB(img); });
}

void AppController::handleTaskChange(int taskIndex) {
    m_currentTask = taskIndex;
    m_window->updateLayoutForTask(taskIndex);
}

void AppController::handleApply() {
    if (!m_state.hasImageA()) { m_window->setStatusMessage("Load image A!", false); return; }
    if ((m_currentTask >= 4) && !m_state.hasImageB()) { m_window->setStatusMessage("Load image B!", false); return; }

    m_window->getTopTaskBar()->setProcessing(true);
    try {
        switch (m_currentTask) {
            case 1: runModule1(); break;
            case 2: runModule2(); break;
            case 3: runModule3(); break;
            case 4: runModule4(); break;
            case 5: runModule5(); break;
        }
    } catch (...) { m_window->setStatusMessage("Backend Error", false); }
    m_window->getTopTaskBar()->setProcessing(false);
}

// ── Module 1 ─────────────────────────────────────────────────────────

void AppController::runModule1()
{
    auto* params = m_window->getTopTaskBar()->getParameterBox();
    cv::Mat src  = m_state.getImageA();
 
    // ── Read parameters ──────────────────────────────────────────────────────
    int  methodIdx   = params->comboIndex("threshMethod",  0);
    bool applyColor  = params->boolValue ("threshColor",   false);
    int  windowSize  = params->intValue  ("threshWindow",  11);   // local threshold tile
    int  localMethod = params->comboIndex("threshLocalMethod", 0); // 0=Optimal, 1=Otsu
 
    Thresholding::Result result;
 
    // ── Dispatch ─────────────────────────────────────────────────────────────
    switch (methodIdx)
    {
        case 0: // Optimal
            result = Thresholding::optimalThreshold(src, applyColor);
            break;
        case 1: // Otsu
            result = Thresholding::otsuThreshold(src, applyColor);
            break;
        case 2: // Spectral
            result = Thresholding::spectralThreshold(src, applyColor);
            break;
        case 3: // Local
            result = Thresholding::localThreshold(src, windowSize, localMethod);
            break;
        default:
            m_window->setStatusMessage("Unknown method", false);
            return;
    }
 
    // ── Push result to output panel ───────────────────────────────────────────
    m_state.setOutput(result.binary);
    m_window->getPanelOut()->displayImage(result.binary);
 
    // ── Build status / info report ───────────────────────────────────────────
    static const char* methodNames[] = {
        "Optimal (Iterative)", "Otsu", "Spectral (Multi-Modal)", "Local (Adaptive)"
    };
 
    // Threshold string  –  may be multiple values for spectral
    QString threshStr;
    if (result.thresholds.size() == 1)
        threshStr = QString("T = %1").arg(result.thresholds[0], 0, 'f', 1);
    else
    {
        QStringList parts;
        for (double t : result.thresholds)
            parts << QString::number(t, 'f', 1);
        threshStr = "T = [" + parts.join(", ") + "]";
    }
 
    QString extra;
    if (methodIdx == 0)
        extra = QString("Converged in <b>%1</b> iterations.<br>%2")
                    .arg(result.iterations)
                    .arg(threshStr);
    else if (methodIdx == 2)
        extra = QString("<b>%1</b> threshold(s) found (multi-modal).<br>%2")
                    .arg(result.thresholds.size())
                    .arg(threshStr);
    else if (methodIdx == 3)
        extra = QString("Tile size: <b>%1 × %1</b> px  |  Avg %2")
                    .arg(windowSize)
                    .arg(threshStr);
    else
        extra = threshStr;
 
    showDetectionReport(
        QString(methodNames[methodIdx]),
        0,                  // no keypoints for thresholding
        result.timingMs,
        (src.channels() == 3 && applyColor),
        extra
    );
 
    m_window->setStatusMessage(
        QString("Threshold applied  (%1)  |  %2")
            .arg(methodNames[methodIdx])
            .arg(QString("%1 ms").arg(result.timingMs, 0, 'f', 1)),
        true);
}
// ── Module 2 ─────────────────────────────────────────────────────────
void AppController::runModule2()
{
    m_window->setStatusMessage("Module 2: Not Implemented Yet", false);
}

// ── Module 3 ─────────────────────────────────────────────────────────
void AppController::runModule3()
{
    m_window->setStatusMessage("Module 3: Not Implemented Yet", false);
}

// ── Module 4 ─────────────────────────────────────────────────────────
void AppController::runModule4()
{
    m_window->setStatusMessage("Module 4: Not Implemented Yet", false);
}

// ── Module 5 ─────────────────────────────────────────────────────────
void AppController::runModule5()
{
    m_window->setStatusMessage("Module 5: Not Implemented Yet", false);
}

// ── Clear ─────────────────────────────────────────────────────────────────────
void AppController::handleClear()
{
    m_window->getPanelOut()->clear();
    m_window->getPanelA()->clearKeyPoints();
    m_window->getPanelA()->clearTiming();
    m_window->getPanelB()->clearKeyPoints();
    m_window->getPanelB()->clearTiming();
    m_state.clearOutput();
    m_window->setStatusMessage("Outputs cleared", true);
    m_window->updateLayoutForTask(m_currentTask);
}

// ── Save ──────────────────────────────────────────────────────────────────────
void AppController::handleSave()
{
    cv::Mat out = m_state.getOutput();
    if (out.empty())
    {
        m_window->setStatusMessage("Nothing to save", false);
        return;
    }
    QString path = QFileDialog::getSaveFileName(
        m_window, "Save Result", "",
        "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)");
    if (!path.isEmpty())
    {
        cv::imwrite(path.toStdString(), out);
        m_window->setStatusMessage("Saved ✓", true);
    }
}

// ── HTML report builders ──────────────────────────────────────────────────────
void AppController::showDetectionReport(const QString &methodName,
                                        int kpCount, double timingMs,
                                        bool isColor, const QString &extra)
{
    QString timingStr = timingMs < 1000.0
                            ? QString("%1 ms").arg(timingMs, 0, 'f', 2)
                            : QString("%1 s").arg(timingMs / 1000.0, 0, 'f', 3);

    QString colorStr = isColor ? "Color (3-ch)" : "Grayscale";

    QString extraHtml;
    if (!extra.isEmpty())
    {
        extraHtml = QString(R"(
        <div class='card'>
            <h3>Descriptor Info</h3>
            <p>%1</p>
        </div>)")
                        .arg(extra);
    }

    bool dark = m_window->isDark();
    QString textColor = dark ? "#E0E0E0" : "#2C2825";
    QString cardBg    = dark ? "#252529" : "#FFFFFF";
    QString cardBord  = dark ? "#3A3A3F" : "#E6E0F7";
    QString titleCol  = dark ? "#FFFFFF" : "#2C2825";
    QString pCol      = dark ? "#A0A0A0" : "#7A7268";

    QString html = QString(R"(
<style>
  body { font-family: 'DM Sans', sans-serif; color: %1; margin: 0; padding: 0; }
  .card { background: %2; border: 1px solid %3; border-radius: 12px;
          padding: 14px; margin-bottom: 10px; }
  h3 { font-size: 14px; font-weight: 800; color: %4; margin: 0 0 6px; }
  p  { font-size: 12px; color: %5; line-height: 1.7; margin: 4px 0; }
  .badge      { display:inline-block; background:#EDE8FF; color:#5B4FCF;
                border-radius:6px; padding:3px 9px; font-size:11px; font-weight:800; margin:2px; }
  .badge-gold { display:inline-block; background:#FEF3C7; color:#B45309;
                border-radius:6px; padding:3px 9px; font-size:11px; font-weight:800; margin:2px; }
  .badge-sage { display:inline-block; background:#E8F5F0; color:#2D9B6F;
                border-radius:6px; padding:3px 9px; font-size:11px; font-weight:800; margin:2px; }
  .row { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
</style>
<div class='card'>
  <h3>%6</h3>
  <p>Algorithm Analysis</p>
</div>
<div class='card'>
  <h3>Metrics</h3>
  <div class='row'>
    <span class='badge'>⬡ %7 results</span>
    <span class='badge-gold'>⏱ %8</span>
    <span class='badge-sage'>%9</span>
  </div>
</div>
%10
<div class='card'>
  <h3>Color Legend</h3>
  <p>● <span style='color:#2D9B6F;font-weight:700;'>Green</span> — small (&lt;10 px)</p>
  <p>● <span style='color:#5B4FCF;font-weight:700;'>Purple</span> — medium (10–25 px)</p>
  <p>● <span style='color:#D85A30;font-weight:700;'>Coral</span> — large (&gt;25 px)</p>
  <p style='margin-top:8px;font-size:11px;color:#C4BDB4;'>Analysis Complete.</p>
</div>
)")
                       .arg(textColor)
                       .arg(cardBg)
                       .arg(cardBord)
                       .arg(titleCol)
                       .arg(pCol)
                       .arg(methodName)
                       .arg(kpCount)
                       .arg(timingStr)
                       .arg(colorStr)
                       .arg(extraHtml);

    if (auto *sb = m_window->getInfoSidebar())
        sb->setHtml(html);
}

void AppController::showMatchingReport(const QString &methodName,
                                       int matchCount, double timingMs,
                                       const QString &extra)
{
    QString timingStr = timingMs < 1000.0
                            ? QString("%1 ms").arg(timingMs, 0, 'f', 2)
                            : QString("%1 s").arg(timingMs / 1000.0, 0, 'f', 3);

    QString extraHtml;
    if (!extra.isEmpty())
        extraHtml = QString("<div class='card'><p>%1</p></div>").arg(extra);

    bool dark = m_window->isDark();
    QString textColor = dark ? "#E0E0E0" : "#2C2825";
    QString cardBg    = dark ? "#252529" : "#FFFFFF";
    QString cardBord  = dark ? "#3A3A3F" : "#E6E0F7";
    QString titleCol  = dark ? "#FFFFFF" : "#2C2825";
    QString pCol      = dark ? "#A0A0A0" : "#7A7268";

    QString html = QString(R"(
<style>
  body { font-family: 'DM Sans', sans-serif; color: %1; margin: 0; padding: 0; }
  .card { background: %2; border: 1px solid %3; border-radius: 12px;
          padding: 14px; margin-bottom: 10px; }
  h3 { font-size: 14px; font-weight: 800; color: %4; margin: 0 0 6px; }
  p  { font-size: 12px; color: %5; line-height: 1.7; margin: 4px 0; }
  .badge      { display:inline-block; background:#EDE8FF; color:#5B4FCF;
                border-radius:6px; padding:3px 9px; font-size:11px; font-weight:800; margin:2px; }
  .badge-gold { display:inline-block; background:#FEF3C7; color:#B45309;
                border-radius:6px; padding:3px 9px; font-size:11px; font-weight:800; margin:2px; }
  .row { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
</style>
<div class='card'>
  <h3>%1 Matching</h3>
  <p>Pairwise descriptor matching results</p>
</div>
<div class='card'>
  <h3>Metrics</h3>
  <div class='row'>
    <span class='badge'>⟺ %2 matches</span>
    <span class='badge-gold'>⏱ %3</span>
  </div>
</div>
%4
<div class='card'>
  <h3>About %1</h3>
  <p>%5</p>
</div>
)")
                       .arg(textColor)
                       .arg(cardBg)
                       .arg(cardBord)
                       .arg(titleCol)
                       .arg(pCol)
                       .arg(methodName)
                       .arg(matchCount)
                       .arg(timingStr)
                       .arg(extraHtml)
                       .arg(methodName == "SSD"
                                ? "Sum of Squared Differences measures descriptor similarity by summing pixel-wise squared differences. Lower scores = better matches."
                                : "Normalized Cross-Correlation measures the cosine similarity between descriptor vectors. Scores near 1.0 indicate strong matches.");

    if (auto *sb = m_window->getInfoSidebar())
        sb->setHtml(html);
}

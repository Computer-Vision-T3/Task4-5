#include "AppController.h"
#include <QUrl>
#include <QDebug>
#include <QTimer>
#include <QByteArray>
#include <QBuffer>
#include <QImage>
#include <QFileDialog>
#include <QtConcurrent/QtConcurrent>

// Backend Headers
#include "../../backend/Module1_Thresholding/Thresholding.h"
#include "../../backend/Module2_Clustering/Clustering.h"
#include "../../backend/Module3_Segmentation/Segmentation.h"
#include "../../backend/Module4_FaceDetection/FaceDetection.h"
#include "../../backend/Module5_FaceRecognition/FaceRecognition.h"

// ── NEW HELPER FUNCTIONS ─────────────────────────────────────────────────

// Helper to convert cv::Mat to QImage
QImage matToQImage(const cv::Mat& mat) {
    if (mat.empty()) return QImage();
    if (mat.type() == CV_8UC1) {
        return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Grayscale8).copy();
    } else if (mat.type() == CV_8UC3) {
        return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_BGR888).copy();
    }
    return QImage();
}

// Helper to convert cv::Mat to Base64 String for the HTML Canvas
QString matToBase64(const cv::Mat& mat) {
    if (mat.empty()) return "";
    QImage img = matToQImage(mat);
    QByteArray byteArray;
    QBuffer buffer(&byteArray);
    buffer.open(QIODevice::WriteOnly);
    img.save(&buffer, "PNG"); // PNG preserves crisp edges for threshold masks
    return QString("data:image/png;base64,") + byteArray.toBase64();
}

// ────────────────────────────────────────────────────────────────────────

AppController::AppController(QObject *parent) : QObject(parent) {}

void AppController::setCurrentTask(int index) {
    if (m_currentTask != index) {
        m_currentTask = index;
        emit taskChanged();
    }
}
void AppController::requestImageLoad(const QString& role) {
    QString fileName = QFileDialog::getOpenFileName(nullptr,
        "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)");
    
    if (!fileName.isEmpty()) {
        // Use your existing loadImage logic
        this->loadImage(role, fileName);
    }
}

void AppController::loadImage(const QString& panelRole, const QString& localPath) {
    // Robust path handling: Check if it's already a local path or a file:// URL
    QString path = localPath;
    if (path.startsWith("file://")) {
        path = QUrl(localPath).toLocalFile();
    }
    
    cv::Mat img = cv::imread(path.toStdString(), cv::IMREAD_COLOR);
    
    if (img.empty()) {
        // This was likely emitting an error because 'path' became empty
        emit errorOccurred("Failed to load image at: " + path);
        return;
    }

    if (panelRole == "A" || panelRole == "A2") {
        m_state.setImageA(img);
        emit imageReady("canvasA", matToBase64(img));
        emit imageReady("canvasA2", matToBase64(img));
    } else {
        m_state.setImageB(img);
        emit imageReady("canvasB", matToBase64(img));
    }
}

void AppController::handleApply(const QVariantMap& params) {
    m_currentParams = params; 

    if (!m_state.hasImageA()) {
        emit errorOccurred("Please load Image A first.");
        return;
    }

    emit processingStarted(); 

    // Background thread to keep the UI from freezing
    QtConcurrent::run([this]() {
        try {
            // This 'm_currentTask' is what we updated in Step 1
            switch (m_currentTask) {
                case 1: runModule1(); break;
                case 2: runModule2(); break;
                case 3: runModule3(); break; // Agglomerative lives here
                case 4: runModule4(); break;
                case 5: runModule5(); break;
                default: emit errorOccurred("Invalid Module Selected."); break;
            }
        } catch (const std::exception& e) {
            emit errorOccurred(QString::fromStdString(e.what()));
        }
    });
}

void AppController::handleClear() {
    m_state.clearAll();
    
    // UPDATE: Send empty strings to clear the images from the HTML UI
    emit imageReady("canvasA", "");
    emit imageReady("canvasA2", "");
    emit imageReady("canvasB", "");
    emit imageReady("canvasOut", "");
}

void AppController::handleSave(const QString& filePath) {
    QString path = QUrl(filePath).toLocalFile();
    cv::Mat out = m_state.getOutput();
    
    if (out.empty()) {
        emit errorOccurred("Nothing to save!");
        return;
    }

    if (!path.isEmpty()) {
        if (cv::imwrite(path.toStdString(), out)) {
            qDebug() << "Successfully saved to:" << path;
        } else {
            emit errorOccurred("Failed to save image.");
        }
    }
}

void AppController::runModule1() {
    cv::Mat src = m_state.getImageA();
    int methodIdx = m_currentParams.value("method", 1).toInt();
    bool color = m_currentParams.value("color", false).toBool();
    int window = m_currentParams.value("tileSize", 31).toInt();
    int localMethod = m_currentParams.value("localMethod", 1).toInt();
    
    Thresholding::Result res;
    QString methodName;

    if (methodIdx == 0) { res = Thresholding::optimalThreshold(src, color); methodName = "Optimal"; }
    else if (methodIdx == 1) { res = Thresholding::otsuThreshold(src, color); methodName = "Otsu"; }
    else if (methodIdx == 2) { res = Thresholding::spectralThreshold(src, color); methodName = "Spectral"; }
    else { res = Thresholding::localThreshold(src, window, localMethod); methodName = "Local"; }

    // UPDATE: Save the output state and emit it to canvasOut
    m_state.setOutput(res.binary);
    emit imageReady("canvasOut", matToBase64(res.binary));
    emit processingFinished(methodName, res.timingMs, 1, res.threshold, "Success");
}

void AppController::runModule2() {
    cv::Mat src = m_state.getImageA();
    int methodIdx = m_currentParams.value("method", 2).toInt();
    int k = m_currentParams.value("k", 3).toInt();
    int window = m_currentParams.value("window", 11).toInt();

    Clustering::Result res;
    QString methodName;

    if (methodIdx == 0) { res = Clustering::localThresholding(src, window); methodName = "Local Threshold"; }
    else if (methodIdx == 1) { res = Clustering::regionGrowing(src, window); methodName = "Region Growing"; }
    else { res = Clustering::kMeans(src, k); methodName = "K-Means"; }

    // UPDATE: Save the output state and emit it to canvasOut
    m_state.setOutput(res.clustered);
    emit imageReady("canvasOut", matToBase64(res.clustered));
    emit processingFinished(methodName, res.timingMs, res.numClusters, k, "Success");
}

void AppController::runModule3() {
    cv::Mat src = m_state.getImageA();
    int methodIdx = m_currentParams.value("method", 0).toInt();
    double spat = m_currentParams.value("spatialRad", 10.0).toDouble();
    double col = m_currentParams.value("colorRad", 10.0).toDouble();

    Segmentation::Result res;
    QString methodName;

    if (methodIdx == 0) { res = Segmentation::meanShift(src, spat, col); methodName = "Mean Shift"; }
    else { res = Segmentation::agglomerative(src, spat, col); methodName = "Agglomerative"; }

    // UPDATE: Save the output state and emit it to canvasOut
    m_state.setOutput(res.segmented);
    emit imageReady("canvasOut", matToBase64(res.segmented));
    emit processingFinished(methodName, res.timingMs, res.numSegments, 0.0, "Success");
}

void AppController::runModule4() {
    emit errorOccurred("Face Detection module stub.");
    emit processingFinished("Face Detection", 0, 0, 0, "Not implemented");
}

void AppController::runModule5() {
    emit errorOccurred("Face Recognition module stub.");
    emit processingFinished("Face Recognition", 0, 0, 0, "Not implemented");
}

QString AppController::getMethodName(int task, int methodIdx) const {
    if (task == 1) {
        static const char* n[] = {"Optimal","Otsu","Spectral","Local"};
        return n[methodIdx];
    } else if (task == 2) {
        static const char* n[] = {"Local Thresholding","Region Growing","K-Means"};
        return n[methodIdx];
    }
    return "Unknown";
}
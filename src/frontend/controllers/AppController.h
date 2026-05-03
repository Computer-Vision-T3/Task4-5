#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QVariantMap>
#include "ImageStateManager.h"

class AppController : public QObject {
    Q_OBJECT
    Q_PROPERTY(int currentTask READ currentTask WRITE setCurrentTask NOTIFY taskChanged)

public:
    explicit AppController(QObject* parent = nullptr);

    int currentTask() const { return m_currentTask; }
    void setCurrentTask(int index);

    ImageStateManager& getState() { return m_state; }

    // ── QML/Web Invokable Functions ──────────────────────────────────
    Q_INVOKABLE void handleApply(const QVariantMap& params = QVariantMap());
    Q_INVOKABLE void handleClear();
    Q_INVOKABLE void handleSave(const QString& filePath);
    Q_INVOKABLE void loadImage(const QString& panelRole, const QString& localPath);
    Q_INVOKABLE QString getMethodName(int task, int methodIdx) const;
    Q_INVOKABLE void requestImageLoad(const QString& role);

signals:
    void taskChanged();
    void processingStarted();
    void processingFinished(QString methodName, double timeMs, int resultCount, double threshold, QString extra);
    void errorOccurred(QString message);
    
    // NEW: Signal to send Base64 image data to the HTML Canvas!
    void imageReady(QString canvasId, QString base64Data); 

private:
    void runModule1();
    void runModule2();
    void runModule3();
    void runModule4();
    void runModule5();

    ImageStateManager m_state;
    int               m_currentTask = 1;
    QVariantMap       m_currentParams; 
};
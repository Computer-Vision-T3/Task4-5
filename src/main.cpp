#include <QApplication>
#include <QWebEngineView>
#include <QWebChannel>
#include <QWebEnginePage>
#include <QDir>
#include <QUrl>

#include "frontend/controllers/AppController.h"

int main(int argc, char *argv[]) {
    // Fixes the FileDialog QSettings warning
    QCoreApplication::setOrganizationName("VisionLab");
    QCoreApplication::setOrganizationDomain("visionlab.com");

    // NOTE: We MUST use QApplication (not QGuiApplication) for WebEngineWidgets
    QApplication app(argc, argv);

    // Create your existing backend controller
    AppController appController;

    // 1. Create the Web View
    QWebEngineView* view = new QWebEngineView();
    view->resize(1200, 800);
    view->setWindowTitle("Vision Lab - WebEngine");

    // 2. Setup the WebChannel bridge
    QWebChannel* channel = new QWebChannel(view->page());
    
    // 3. Expose your C++ object to JavaScript
    // "backend" is the name JS will use to access appController
    channel->registerObject("backend", &appController);
    view->page()->setWebChannel(channel);

    // 4. Load your HTML file
    // Ensure this path points to where vision_lab_enhanced.html is relative to your build directory
    QString htmlPath = QDir::currentPath() + "/../src/frontend/vision_lab_enhanced.html";
    view->setUrl(QUrl::fromLocalFile(htmlPath));

    view->show();

    return app.exec();
}
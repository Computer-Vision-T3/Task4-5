#include "ParameterBox.h"
#include <QVBoxLayout>
#include <QHBoxLayout>

ParameterBox::ParameterBox(QWidget* parent) : QWidget(parent) {
    m_layout = new QGridLayout(this);
    m_layout->setContentsMargins(0, 0, 0, 0);
    m_layout->setSpacing(14);
    m_layout->setColumnStretch(5, 1); // right spacer
}

// ── Clear all parameter widgets ───────────────────────────────────────────────
void ParameterBox::clearLayout() {
    QLayoutItem* child;
    while ((child = m_layout->takeAt(0)) != nullptr) {
        if (child->widget()) delete child->widget();
        delete child;
    }
}

// ── Rebuild parameters per task ───────────────────────────────────────────────
void ParameterBox::updateForTask(int taskIndex) {
    clearLayout();

    switch (taskIndex) {
    case 1: // ── Global + Local Thresholding ───────────────────────────────
        addCombo    ("threshMethod",      "Method",
                     {"Optimal", "Otsu", "Spectral", "Local"},            0, 0);
        addCheckBox ("threshColor",       "Apply to Color Channels", false, 0, 1);
 
        // Local-specific controls  (always shown; visible context depends on
        // the user selecting "Local" — you may hide/show them dynamically)
        addSpinBox  ("threshWindow",      "Tile Size (Local)", 3, 201, 31,  0, 2);
        addCombo    ("threshLocalMethod", "Local Method",
                 {"Optimal per-tile", "Otsu per-tile"},               0, 3);
        // Default to Otsu for per-tile local thresholding (user can still change)
        if (auto* cb = findChild<QComboBox*>("threshLocalMethod")) cb->setCurrentIndex(1);
        break;
        
    case 2: { //   Spatial & Basic Clustering 
        addCombo("clusterMethod", "Method", {"Local Thresholding", "Region Growing", "K-Means"}, 0, 0);
        
        // Expanded the max range of the window/tolerance to 255 for Region Growing
        addSpinBox("clusterK",      "Clusters (K)", 2, 50, 3,  0, 1);
        addSpinBox("clusterWindow", "Window Size",  1, 255, 11, 0, 2);

        // Dynamic UI switching
        if (auto* methodCombo = findChild<QComboBox*>("clusterMethod")) {
            connect(methodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx) {
                
                auto* kBox   = findChild<QSpinBox*>("clusterK");
                auto* winBox = findChild<QSpinBox*>("clusterWindow");
                
                if (!kBox || !winBox) return;

                // The spinboxes are wrapped in a generic QWidget container alongside their labels.
                // We need to hide/show the parent container to toggle both the label and the box.
                QWidget* kContainer   = kBox->parentWidget();
                QWidget* winContainer = winBox->parentWidget();
                QLabel*  winLabel     = winContainer->findChild<QLabel*>("paramLabel");

                if (idx == 0) { 
                    // 0: Local Thresholding
                    kContainer->hide();
                    winContainer->show();
                    if (winLabel) winLabel->setText("Window Size");
                } 
                else if (idx == 1) { 
                    // 1: Region Growing
                    kContainer->hide();
                    winContainer->show();
                    if (winLabel) winLabel->setText("Color Tolerance");
                } 
                else if (idx == 2) { 
                    // 2: K-Means
                    kContainer->show();
                    winContainer->hide();
                }
            });
            
            // Trigger it once manually to set the correct initial state when the tab loads
            methodCombo->currentIndexChanged(0);
        }
        break;
    }

    case 3: // ── Advanced Segmentation ───────────────────────────────────────
        addCombo        ("segMethod",     "Method", {"Mean Shift", "Agglomerative"}, 0, 0);
        addDoubleSpinBox("segSpatialRad", "Spatial Radius", 1.0, 100.0, 10.0, 1.0, 0, 1);
        addDoubleSpinBox("segColorRad",   "Color Radius",   1.0, 100.0, 10.0, 1.0, 0, 2);
        break;

    case 4: // ── Face Detection ──────────────────────────────────────────────
        addCombo        ("faceDetMode",   "Mode", {"Default"},                      0, 0);
        addCheckBox     ("faceDetColor",  "Color Detection", false,                 0, 1);
        break;

    case 5: // ── Face Recognition ────────────────────────────────────────────
        addSpinBox      ("faceRecPCA",    "Eigenvectors (PCA)", 1, 500, 50,         0, 0);
        addCheckBox     ("faceRecROC",    "Plot ROC Curve", true,                   0, 1);
        break;

    default: break;
    }
}

// ── Value accessors ───────────────────────────────────────────────────────────
int ParameterBox::intValue(const QString& id, int fallback) const {
    if (auto* w = findChild<QSpinBox*>(id))   return w->value();
    if (auto* w = findChild<QSlider*>(id))    return w->value();
    return fallback;
}
double ParameterBox::dblValue(const QString& id, double fallback) const {
    if (auto* w = findChild<QDoubleSpinBox*>(id)) return w->value();
    if (auto* w = findChild<QSlider*>(id))        return w->value() / 100.0;
    return fallback;
}
bool ParameterBox::boolValue(const QString& id, bool fallback) const {
    if (auto* w = findChild<QCheckBox*>(id)) return w->isChecked();
    return fallback;
}
int ParameterBox::comboIndex(const QString& id, int fallback) const {
    if (auto* w = findChild<QComboBox*>(id)) return w->currentIndex();
    return fallback;
}

// ── Widget builder helpers ────────────────────────────────────────────────────
void ParameterBox::addSpinBox(const QString& id, const QString& label,
                               int min, int max, int def, int row, int col) {
    QWidget* c = new QWidget(this);
    QVBoxLayout* v = new QVBoxLayout(c);
    v->setContentsMargins(0,0,0,0); v->setSpacing(4);

    QLabel* lbl = new QLabel(label, c);
    lbl->setObjectName("paramLabel");

    QSpinBox* sb = new QSpinBox(c);
    sb->setObjectName(id);
    sb->setRange(min, max);
    sb->setValue(def);

    v->addWidget(lbl);
    v->addWidget(sb);
    m_layout->addWidget(c, row, col);
}

void ParameterBox::addDoubleSpinBox(const QString& id, const QString& label,
                                     double min, double max, double def,
                                     double step, int row, int col) {
    QWidget* c = new QWidget(this);
    QVBoxLayout* v = new QVBoxLayout(c);
    v->setContentsMargins(0,0,0,0); v->setSpacing(4);

    QLabel* lbl = new QLabel(label, c);
    lbl->setObjectName("paramLabel");

    QDoubleSpinBox* sb = new QDoubleSpinBox(c);
    sb->setObjectName(id);
    sb->setRange(min, max);
    sb->setSingleStep(step);
    sb->setDecimals(3);
    sb->setValue(def);

    v->addWidget(lbl);
    v->addWidget(sb);
    m_layout->addWidget(c, row, col);
}

void ParameterBox::addSlider(const QString& id, const QString& label,
                              int min, int max, int def,
                              int row, int col, bool isFloat) {
    QWidget* c = new QWidget(this);
    QVBoxLayout* v = new QVBoxLayout(c);
    v->setContentsMargins(0,0,0,0); v->setSpacing(4);

    QWidget* lblRow = new QWidget(c);
    QHBoxLayout* h = new QHBoxLayout(lblRow);
    h->setContentsMargins(0,0,0,0);

    QLabel* lbl = new QLabel(label, lblRow);
    lbl->setObjectName("paramLabel");

    QLabel* val = new QLabel(isFloat
        ? QString::number(def / 10.0, 'f', 1)
        : QString::number(def), lblRow);
    val->setObjectName("valueLabel");
    val->setAlignment(Qt::AlignRight);

    h->addWidget(lbl);
    h->addWidget(val);

    QSlider* sl = new QSlider(Qt::Horizontal, c);
    sl->setObjectName(id);
    sl->setRange(min, max);
    sl->setValue(def);
    connect(sl, &QSlider::valueChanged, [val, isFloat](int v) {
        val->setText(isFloat ? QString::number(v / 10.0, 'f', 1) : QString::number(v));
    });

    v->addWidget(lblRow);
    v->addWidget(sl);
    m_layout->addWidget(c, row, col);
}

void ParameterBox::addCombo(const QString& id, const QString& label,
                             const QStringList& items, int row, int col) {
    QWidget* c = new QWidget(this);
    QVBoxLayout* v = new QVBoxLayout(c);
    v->setContentsMargins(0,0,0,0); v->setSpacing(4);

    QLabel* lbl = new QLabel(label, c);
    lbl->setObjectName("paramLabel");

    QComboBox* cb = new QComboBox(c);
    cb->setObjectName(id);
    cb->addItems(items);

    v->addWidget(lbl);
    v->addWidget(cb);
    m_layout->addWidget(c, row, col);
}

void ParameterBox::addCheckBox(const QString& id, const QString& label,
                                bool def, int row, int col) {
    QCheckBox* cb = new QCheckBox(label, this);
    cb->setObjectName(id);
    cb->setChecked(def);
    m_layout->addWidget(cb, row, col, Qt::AlignBottom);
}

void ParameterBox::addButton(const QString& text, int targetTask, int row, int col) {
    QPushButton* btn = new QPushButton(text, this);
    btn->setObjectName("loadBtn");
    btn->setCursor(Qt::PointingHandCursor);
    connect(btn, &QPushButton::clicked, [this, targetTask]() {
        emit quickActionRequested(targetTask);
    });
    m_layout->addWidget(btn, row, col, Qt::AlignBottom);
}
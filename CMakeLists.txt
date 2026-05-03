cmake_minimum_required(VERSION 3.10)
project(Task4_5_Studio VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─── Qt5 Configuration ──────────────────────────────────────────────
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# WebEngine components for the HTML-based UI bridge
find_package(Qt5 COMPONENTS Core Gui Widgets WebEngineWidgets WebChannel Concurrent REQUIRED)

# ─── OpenCV & CURL ──────────────────────────────────────────────────
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)

# ─── Include Directories ────────────────────────────────────────────
include_directories(
    src
    ${Qt5Widgets_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
)

# ─── Source Files ───────────────────────────────────────────────────
# Cleaned up: Only scanning for controllers and backend logic
file(GLOB_RECURSE FRONTEND_SOURCES
    "src/frontend/controllers/*.cpp"
    "src/frontend/controllers/*.h"
)

file(GLOB_RECURSE BACKEND_SOURCES
    "src/backend/*.cpp"
    "src/backend/*.h"
)

# ─── Executable & Linking ───────────────────────────────────────────
# The 'launch' target now builds the WebEngine entry point
add_executable(launch src/main.cpp ${FRONTEND_SOURCES} ${BACKEND_SOURCES})

target_link_libraries(launch PRIVATE
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    Qt5::WebEngineWidgets
    Qt5::WebChannel
    Qt5::Concurrent
    ${OpenCV_LIBS}
    ${CURL_LIBRARIES}
    tiff                 # Fixes GDAL/TIFF undefined references
    pthread              # Multi-threading support for OpenCV
    -lcurl
)

# Tool for testing thresholding logic independently
add_executable(threshold_probe
    tools/threshold_probe.cpp
    src/backend/Module1_Thresholding/Thresholding.cpp
)

target_include_directories(threshold_probe PRIVATE src)
target_link_libraries(threshold_probe PRIVATE ${OpenCV_LIBS})
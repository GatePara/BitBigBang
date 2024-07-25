cmake_minimum_required(VERSION 3.0...3.26)

project(hnswlib
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

add_library(hnswlib INTERFACE)
add_library(hnswlib::hnswlib ALIAS hnswlib)

target_include_directories(hnswlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

set(CMAKE_CXX_STANDARD 11)

SET( CMAKE_CXX_FLAGS "-Ofast -lrt -std=c++11 -DHAVE_CXX0X -march=native -mtune=native -finline-functions -fipa-pta -funroll-loops -mdaz-ftz -ffast-math -mprefer-vector-width=512 -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )

add_executable(mt_filter6 mt_filter6.cpp)
target_link_libraries(mt_filter6 hnswlib)

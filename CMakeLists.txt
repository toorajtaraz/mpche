cmake_minimum_required(VERSION 3.10)
PROJECT (mpche LANGUAGES CXX)
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/submodules/opencv/build/")
find_package( OpenCV REQUIRED )
find_package(OpenMP REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${OpenMP_CXX_INCLUDE_DIRS} )
include_directories( ${OpenMP_C_INCLUDE_DIRS} )

set( NAME_SRC
  mpche.cpp
  utils/ocv_wrappers.cpp
  utils/SerialLHE.cpp
  utils/ParallelLHE.cpp
)

set( NAME_HEADERS
  headers/ocv_wrappers.h
  headers/SerialLHE.h
  headers/ParallelLHE.h
)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/headers )
link_directories( ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable( mpche ${NAME_SRC} ${NAME_HEADERS} )

target_link_libraries( mpche PRIVATE OpenMP::OpenMP_CXX ${OpenCV_LIBS})

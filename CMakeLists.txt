cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov5_demo_linux)

set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -s -O3")

if (CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH lib64)
else()
  set(LIB_ARCH lib)
endif()


# drm
set(DRM_DIR ${CMAKE_SOURCE_DIR}/3rdparty/drm)
include_directories(${DRM_DIR}/include)
include_directories(${DRM_DIR}/include/libdrm)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/objDet)

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/3rdparty/librknn_api)
include_directories(${RKNN_API_PATH}/include)
set(RKNN_API_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknn_api.so)


#opencv
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/opencv/include)
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)


set(CMAKE_INSTALL_RPATH "lib")

add_executable(rknn_yolov5_demo
        utils/drm_func.c
        utils/postprocess.cc
        src/objDet/objectDet.cpp
        src/main.cc
        )

target_link_libraries(rknn_yolov5_demo
	${RKNN_API_LIB}
  ${OpenCV_LIBS}
  dl
)

# install target and libraries
# set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo)
# install(TARGETS rknn_yolov5_demo DESTINATION ./)
# install(DIRECTORY model DESTINATION ./)
# install(PROGRAMS ${RKNN_API_LIB} DESTINATION lib)

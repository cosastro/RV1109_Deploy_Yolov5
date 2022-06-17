#pragma once

#include "drm_func.h"
#include "postprocess.h"
#include "rknn_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <vector>
#include "postprocess.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

class ObjDet{
    public:
        ObjDet();
        ~ObjDet();
        int Init(const char* model_path);
        int DetProcess(cv::Mat & input_img, std::vector<cv::Rect> & rects);
        int ReleaseModel();

    private:
        int            status     = 0;
        const char*          model_name = NULL;
        rknn_context   ctx;
        void*          drm_buf = NULL;
        int            drm_fd  = -1;
        int            buf_fd  = -1; // converted from buffer handle
        unsigned int   handle;
        size_t         actual_size = 0;
        int            img_width   = 0;
        int            img_height  = 0;
        int            img_channel = 0;
        
        drm_context    drm_ctx;
        const float    nms_threshold      = NMS_THRESH;
        const float    box_conf_threshold = BOX_THRESH;
        struct timeval start_time, stop_time;
        int            ret;
        rknn_sdk_version version;
        rknn_input_output_num io_num;
        rknn_tensor_attr input_attrs[1];
        rknn_tensor_attr output_attrs[3];
        unsigned char* model_data;
        
};
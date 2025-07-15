/*
 * @file    prl_tracker.cpp
 * @brief   PRL tabanlı nesne takip algoritması
 * @author  Hacı Veli Türcan
 * @date    12.07.2025
*/


#ifndef PRL_TRACKER_H
#define PRL_TRACKER_H

#include <onnxruntime_cxx_api.h>
#include <torch/torch.h>
#include <map>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "tracker.hpp"



//status -3 not initalized and allocated memory
//status -2 tracker is free for use
//status -1 object is inactive 
//status 0 confidence building mode
//status 1 tracker is active

class PRLTracker {
    //Burası 1
    public: PRLTracker(std::string onnx_model_path, Ort::SessionOptions & sessionOptions,
        Ort::Env & env);

    
    ~PRLTracker();

    //Parametlereler değişecek


    void init(cv::Mat img, cv::Rect bbox, track_prl_config* tconfig);
    

    //parametre değişecek.
    std::pair < cv::Rect,
    float> track(cv::Mat img, track_prl_config* tconfig);

    private: std::pair < std::vector < float > ,
    size_t > get_subwindow(const cv::Mat & im,
        std::vector < float > pos,
        int model_sz,
        int original_sz,
        cv::Scalar avg_chans);


    std::vector < float > generate_anchor(const torch::Tensor & mapp,int output_size, float search_size);


    torch::Tensor _convert_bbox(torch::Tensor delta, torch::Tensor anchor);
    at::Tensor _convert_score(torch::Tensor & score);
    std::vector < float > _bbox_clip(float cx, float cy, float width, float height,
        std::vector < int > boundary);


    int status;
    int score_size;
    int anchor_num;
    std::vector < float > window;

    std::pair <std::vector <float> ,size_t> z_crop_blob;

    std::vector < int64_t > * dims_template;
    std::vector < int64_t > * dims_search;

    Ort::AllocatorWithDefaultOptions allocator;

    std::string & onnx_model_path_prl;
    std::vector < float > center_pos;
    std::vector < float > size;
    float scaleaa;
    cv::Scalar channel_average;

    std::vector < Ort::Value > inputTensors;
    std::vector < Ort::Value > outputTensors;
    Ort::MemoryInfo * memoryInfo;

    Ort::SessionOptions & sessionOptions;
    Ort::Env & env;

    Ort::Session session;
    int64_t batchSize;
};

#endif

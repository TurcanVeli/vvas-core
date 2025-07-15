/*
 *
 * Copyright (C) 2022 Xilinx, Inc.
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include <opencv2/opencv.hpp>
#include "tracker_int.hpp"
#include "prl_tracker.hpp"

cv::Rect rect_from_rectf(const Rectf& rf);
cv::Mat mat_from_matimg(const Mat_img& m);


void track_by_detection(vvas_tracker *tracker, objs_data new_objs, int *ids, Mat_img img, track_prl_config tconfig) {

  tracker->deinit_tracker();
  Rectf bbox1 = { 0 };
  bbox1.x = new_objs.objs[0].x;
  bbox1.y = new_objs.objs[0].y;
  bbox1.width = new_objs.objs[0].width;
  bbox1.height = new_objs.objs[0].height;
  tracker->init_tracker(tconfig, bbox1, img);

}

void vvas_tracker::init_tracker(track_prl_config tconfig, Rectf bbox, Mat_img img) {
	
	if (tconfig.tracker_type == ALGO_PRL) {
        if (PRL_tracker != nullptr) {
            delete PRL_tracker;
            PRL_tracker = nullptr;
        }
        
        cv::Rect cv_bbox = rect_from_rectf(bbox);
        cv::Mat cv_img = mat_from_matimg(img);
		PRL_tracker->init(cv_img, cv_bbox,  &tconfig);

    } else {
        std::cerr << "PRL ALGO FAILED" << std::endl;
    }
	
}
cv::Rect rect_from_rectf(const Rectf& rf) {
    return cv::Rect(
        static_cast<int>(rf.x),
        static_cast<int>(rf.y),
        static_cast<int>(rf.width),
        static_cast<int>(rf.height)
    );
}

cv::Mat mat_from_matimg(const Mat_img& m) {
    // channels == 3 ise, 8-bit unsigned BGR varsayalım
    return cv::Mat(
        m.height,         // rows
        m.width,          // cols
        CV_8UC3,          // type (8-bit, 3 channel)
        m.data,           // data pointer
        m.width * m.channels // step (genelde stride ama dikkat: stride mı width*channels mı?)
    );
}

//Burayı düzenle
void vvas_tracker::deinit_tracker() {
  std::cout<<"deinitilazitation tracker" << std::endl;
}


int init_tracker(tracker_handle *tracker_data) {
    
    vvas_tracker *tracker = (vvas_tracker *)malloc(sizeof(vvas_tracker));

    if (!tracker)
        return -1; // Bellek hatası

    // Varsayılan başlangıç değerleri
    tracker->status = -3;
    if (tracker_data->tconfig.tracker_type == ALGO_PRL)
    {
      tracker->tracker_type = 0; 

    }
    else{
        tracker->tracker_type = -1;
    } 

	tracker->PRL_tracker = new PRLTracker(tracker_data->tconfig.MODEL_PATH,
               *tracker_data->tconfig.onnxContext.options, // pointerdan referans!
               *tracker_data->tconfig.onnxContext.env);
    tracker_data->tracker_info = (char *)tracker;

    return 0;
}


//PRL de initilization
int deinit_tracker(tracker_handle *tracker_data) {
  vvas_tracker *tracker;
  tracker = (vvas_tracker *)tracker_data->tracker_info;
  tracker->deinit_tracker();
  free(tracker);
  tracker_data->tracker_info = NULL;
  return 0;
}

/*
void out_object_tracker_info(vvas_tracker *tracker, track_prl_config tconfig, objs_data *trk_objs) {
  int obj_cnt = 0;
  if(tracker->status >= 0) {
    trk_objs->objs[obj_cnt].x = tracker->obj_rect.x;
    trk_objs->objs[obj_cnt].y = tracker->obj_rect.y;
    trk_objs->objs[obj_cnt].width = tracker->obj_rect.width;
    trk_objs->objs[obj_cnt].height = tracker->obj_rect.height;
    trk_objs->objs[obj_cnt].map_id = tracker->obj_rect.map_id;
    trk_objs->objs[obj_cnt].status = tracker->status;
    trk_objs->objs[obj_cnt].trk_id = tracker->id;
    
  }
  trk_objs->num_objs = obj_cnt;
  //Burdan Emin Değilim
}

*/

int objects_detect_update(vvas_tracker *tracker, objs_data new_objs, int *ids, Mat_img img, track_prl_config tconfig) {
  track_by_detection(tracker, new_objs, ids, img, tconfig);
  
  return 0;
}



int run_tracker(Mat_img img, tracker_handle *tracker_data, bool detect_flag) {
  	vvas_tracker *tracker;
  	tracker = (vvas_tracker *)tracker_data->tracker_info;
	objects_detect_update (tracker, tracker_data->new_objs, &tracker_data->ids,
        img, tracker_data->tconfig);
	//Sadece prl_track çalışacak burada. 
	//init kısmıdna doğru tanımlandığından emin ol.
    cv::Mat cv_img = mat_from_matimg(img);


	//tracker->PRL_tracker->track(cv_img, &tracker_data->tconfig);
  auto track_result = tracker->PRL_tracker->track(cv_img, &tracker_data->tconfig);

  // Burada track_result'taki bbox ve score'u bir yere yaz
  tracker->obj_rect.x = track_result.first.x;
  tracker->obj_rect.y = track_result.first.y;
  tracker->obj_rect.width = track_result.first.width;
  tracker->obj_rect.height = track_result.first.height;
  tracker->conf_score = track_result.second;  
  tracker->status = 1;
	
  return 0;
}

#include "vvas_yolov6.hpp"
#include <algorithm>

vvas_yolov6::vvas_yolov6 (void * handle, const std::string & model_name,
    bool need_preprocess)
{
  VvasDpuInferPrivate *kpriv = (VvasDpuInferPrivate *)handle;
  log_level = kpriv->log_level;
  kpriv->labelflags = VVAS_XLABEL_REQUIRED;
  LOG_MESSAGE (LOG_LEVEL_DEBUG, kpriv->log_level, "enter");

  if (kpriv->labelptr == NULL) {
    LOG_MESSAGE (LOG_LEVEL_ERROR, kpriv->log_level, "label not found");
    kpriv->labelflags |= VVAS_XLABEL_NOT_FOUND;
  } else
    kpriv->labelflags |= VVAS_XLABEL_FOUND;


  model = vitis::ai::YOLOv6::create (model_name, need_preprocess);
}



/*
struct YOLOv6Result {
  struct BoundingBox {
    /// Classification.
    int label;
    /// Confidence. The value ranges from 0 to 1.
    float score;
    /// (x0,y0,x1,y1). x0, x1 Range from 0 to the input image columns.
    /// y0,y1. Range from 0 to the input image rows.
    std::vector<float> box;
  };
  /// All objects, The vector of BoundingBox.
  std::vector<BoundingBox> bboxes;
};
*/
bool
compare_by_area (const vitis::ai::YOLOv6Result::BoundingBox & box1,
    const vitis::ai::YOLOv6Result::BoundingBox & box2)
{
  float area1 = (box1.box[2] - box1.box[0]) * (box1.box[3] - box1.box[1]);
  float area2 = (box2.box[2] - box2.box[0]) * (box2.box[3] - box2.box[1]);
  return (area1 > area2);
}



int
vvas_yolov6::run (void *handle, std::vector < cv::Mat > &images,
    VvasInferPrediction ** predictions)
{
  VvasDpuInferPrivate *kpriv = (VvasDpuInferPrivate *)handle;
  LOG_MESSAGE (LOG_LEVEL_DEBUG, kpriv->log_level, "enter batch");
  auto results = model->run (images);

  labels *lptr;
  char *pstr;                   /* prediction string */

  if (kpriv->labelptr == NULL) {
    LOG_MESSAGE (LOG_LEVEL_ERROR, kpriv->log_level, "label not found");
    return false;
  }

  if (kpriv->objs_detection_max > 0) {
    LOG_MESSAGE (LOG_LEVEL_DEBUG, kpriv->log_level,
        "sort detected objects based on bbox area");

    /* sort objects based on dimension to pick objects with bigger bbox */
    for (unsigned int i = 0u; i < results.size (); i++) {
      std::sort (results[i].bboxes.begin (), results[i].bboxes.end (),
          compare_by_area);
    }
  } else {
    LOG_MESSAGE (LOG_LEVEL_WARNING, kpriv->log_level,
        "max-objects count is zero. So, not doing any metadata processing");
    return true;
  }

  for (auto i = 0u; i < results.size (); i++) {
    VvasInferPrediction *parent_predict = NULL;
    unsigned int cur_objs = 0;

    LOG_MESSAGE (LOG_LEVEL_INFO, kpriv->log_level, "objects detected %lu",
        results[i].bboxes.size ());

    if (results[i].bboxes.size ()) {
      VvasBoundingBox parent_bbox;
      int cols = images[i].cols;
      int rows = images[i].rows;

      parent_predict = predictions[i];

    for (auto & box:results[i].bboxes) {

        lptr = kpriv->labelptr + box.label;
        if (kpriv->filter_labels.size ()) {
          bool found_label = false;

          for (unsigned int n = 0; n < kpriv->filter_labels.size (); n++) {
            const char *filter_label = kpriv->filter_labels[n].c_str ();
            const char *current_label = lptr->display_name.c_str ();
            if (!strncmp (current_label, filter_label, strlen (filter_label))) {
              LOG_MESSAGE (LOG_LEVEL_DEBUG, kpriv->log_level,
                  "current label %s is in filter_label list", current_label);
              found_label = true;
            }
          }

          if (!found_label)
            continue;
        }

        if (!parent_predict) {
          parent_bbox.x = parent_bbox.y = 0;
          parent_bbox.width = cols;
          parent_bbox.height = rows;
          parent_predict = vvas_inferprediction_new ();
          parent_predict->bbox = parent_bbox;
        }


        int label = box.label;
        float xmin = box.box[0];
        float ymin = box.box[1];
        float xmax = box.box[2];
        float ymax = box.box[3];

        
        if (xmin < 0.)
        xmin = 1.;
        if (ymin < 0.)
        ymin = 1.;
        if (xmax > cols)
        xmax = cols;
        if (ymax > rows)
        ymax = rows;
        float confidence = box.score;
        
        VvasBoundingBox bbox;
        VvasInferPrediction *predict;
        VvasInferClassification *c = NULL;

        bbox.x = xmin;
        bbox.y = ymin;
        bbox.width = xmax - xmin;
        bbox.height = ymax - ymin;

        predict = vvas_inferprediction_new ();
        predict->bbox = bbox;

        c = vvas_inferclassification_new ();
        c->class_id = label;
        c->class_prob = confidence;
        c->class_label = strdup (lptr->display_name.c_str ());
        c->num_classes = 0;
        predict->classifications = vvas_list_append (predict->classifications, c);

        /* add class and name in prediction node */
        predict->model_class = (VvasClass) kpriv->modelclass;
        predict->model_name = strdup (kpriv->modelname.c_str ());
        vvas_inferprediction_append (parent_predict, predict);

        LOG_MESSAGE (LOG_LEVEL_INFO, kpriv->log_level,
            "RESULT: %s(%d) %f %f %f %f (%f)", lptr->display_name.c_str (),
            label, xmin, ymin, xmax, ymax, confidence);

        cur_objs++;
        if (cur_objs == kpriv->objs_detection_max) {
          LOG_MESSAGE (LOG_LEVEL_DEBUG, kpriv->log_level,
              "reached max limit of objects to add to metadata");
          break;
        }
      }

      if (parent_predict) {
        pstr = vvas_inferprediction_to_string (parent_predict);
        LOG_MESSAGE (LOG_LEVEL_DEBUG, kpriv->log_level,
            "prediction tree : \n%s", pstr);
        free (pstr);
      }
    }
    predictions[i] = parent_predict;
  }

  LOG_MESSAGE (LOG_LEVEL_INFO, kpriv->log_level, " ");

  return true;
}



int
vvas_yolov6::requiredwidth (void)
{
  LOG_MESSAGE (LOG_LEVEL_DEBUG, log_level, "enter");
  return model->getInputWidth ();
}

int
vvas_yolov6::requiredheight (void)
{
  LOG_MESSAGE (LOG_LEVEL_DEBUG, log_level, "enter");
  return model->getInputHeight ();
}

int
vvas_yolov6::supportedbatchsz (void)
{
  LOG_MESSAGE (LOG_LEVEL_DEBUG, log_level, "enter");
  return model->get_input_batch ();
}

int
vvas_yolov6::close (void)
{
  LOG_MESSAGE (LOG_LEVEL_DEBUG, log_level, "enter");
  return true;
}

vvas_yolov6::~vvas_yolov6 ()
{
  LOG_MESSAGE (LOG_LEVEL_DEBUG, log_level, "enter");
}

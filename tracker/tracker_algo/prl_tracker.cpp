

#include <cmath>
#include <iostream>

#include "prl_tracker.hpp"



// 16.7 KB memory leak Direclty



template <typename T>
T vectorProduct(const std::vector<T> &v) {
  return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

PRLTracker::PRLTracker(std::string onnx_model_path,
                       Ort::SessionOptions &sessionOptions, Ort::Env &env)
    : score_size(11), //buraya tconfig ver
      anchor_num(1),
      onnx_model_path_prl(onnx_model_path),
      sessionOptions(this->sessionOptions),
      env(this->env),
      session(env, onnx_model_path_prl.c_str(), sessionOptions) {
  batchSize = 1;

  size_t numInputNodes = session.GetInputCount();
  size_t numOutputNodes = session.GetOutputCount();

  // Input 0
  Ort::AllocatedStringPtr inputNamePtr_template =
      session.GetInputNameAllocated(0, allocator);
  const char *inputName_template = inputNamePtr_template.get();

  Ort::TypeInfo inputTypeInfo_template = session.GetInputTypeInfo(0);
  auto inputTensorInfo_template =
      inputTypeInfo_template.GetTensorTypeAndShapeInfo();

  Ort::MemoryInfo tmp = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  memoryInfo = new Ort::MemoryInfo(std::move(tmp));

  ONNXTensorElementDataType inputType_template =
      inputTensorInfo_template.GetElementType();

  std::vector<int64_t> inputDims_template = inputTensorInfo_template.GetShape();
  dims_template = new std::vector<int64_t>(inputDims_template);
  if (dims_template->at(0) == -1) {
    dims_template->at(0) = batchSize;
  } else {
    batchSize = dims_template->at(0);
  }

  // Input 1
  Ort::AllocatedStringPtr inputNamePtr_search =
      session.GetInputNameAllocated(1, allocator);
  const char *inputName_search = inputNamePtr_search.get();

  Ort::TypeInfo inputTypeInfo_search = session.GetInputTypeInfo(1);
  auto inputTensorInfo_search =
      inputTypeInfo_search.GetTensorTypeAndShapeInfo();

  ONNXTensorElementDataType inputType_search =
      inputTensorInfo_search.GetElementType();

  std::vector<int64_t> inputDims_search = inputTensorInfo_search.GetShape();
  dims_search = new std::vector<int64_t>(inputDims_search);
  if (inputDims_search.at(0) == -1) {
    inputDims_search.at(0) = batchSize;
  }

  // Output 0
  Ort::AllocatedStringPtr outputNamePtr_loc =
      session.GetOutputNameAllocated(0, allocator);
  const char *outputName_loc = outputNamePtr_loc.get();

  Ort::TypeInfo outputTypeInfo_loc = session.GetOutputTypeInfo(0);
  auto outputTensorInfo_loc = outputTypeInfo_loc.GetTensorTypeAndShapeInfo();

  ONNXTensorElementDataType outputType_loc =
      outputTensorInfo_loc.GetElementType();

  std::vector<int64_t> outputDims_loc = outputTensorInfo_loc.GetShape();
  if (outputDims_loc.at(0) == -1) {
    outputDims_loc.at(0) = batchSize;
  }

  // Output 1
  Ort::AllocatedStringPtr outputNamePtr_cls1 =
      session.GetOutputNameAllocated(1, allocator);
  const char *outputName_cls1 = outputNamePtr_cls1.get();

  Ort::TypeInfo outputTypeInfo_cls1 = session.GetOutputTypeInfo(1);
  auto outputTensorInfo_cls1 = outputTypeInfo_cls1.GetTensorTypeAndShapeInfo();

  ONNXTensorElementDataType outputType_cls1 =
      outputTensorInfo_cls1.GetElementType();

  std::vector<int64_t> outputDims_cls1 = outputTensorInfo_cls1.GetShape();
  if (outputDims_cls1.at(0) == -1) {
    outputDims_cls1.at(0) = batchSize;
  }

  // Output 2
  Ort::AllocatedStringPtr outputNamePtr_cls2 =
      session.GetOutputNameAllocated(2, allocator);
  const char *outputName_cls2 = outputNamePtr_cls2.get();

  Ort::TypeInfo outputTypeInfo_cls2 = session.GetOutputTypeInfo(2);
  auto outputTensorInfo_cls2 = outputTypeInfo_cls2.GetTensorTypeAndShapeInfo();

  ONNXTensorElementDataType outputType_cls2 =
      outputTensorInfo_cls2.GetElementType();

  std::vector<int64_t> outputDims_cls2 = outputTensorInfo_cls2.GetShape();
  if (outputDims_cls2.at(0) == -1) {
    outputDims_cls2.at(0) = batchSize;
  }

  // Create hanning window
  std::vector<float> hanning(score_size);
  for (int i = 0; i < score_size; ++i) {
    hanning[i] = 0.5 * (1 - std::cos(2 * M_PI * i / (score_size - 1)));
  }

  // Create 2D window
  window.resize(score_size * score_size);
  for (int i = 0; i < score_size; ++i) {
    for (int j = 0; j < score_size; ++j) {
      window[i * score_size + j] = hanning[i] * hanning[j];
    }
  }
}

PRLTracker::~PRLTracker() {
  if (memoryInfo != nullptr) {
    delete memoryInfo;
    memoryInfo = nullptr;
  }

  if (dims_search != nullptr) {
    delete dims_search;
    dims_search = nullptr;
  }
  if (dims_template != nullptr) {
    delete dims_template;
    dims_template = nullptr;
  }
}

void PRLTracker::init(cv::Mat img, cv::Rect bbox, track_prl_config* tconfig) {
  center_pos = {static_cast<float>(bbox.x) + (bbox.width - 1) / 2.0f,
                static_cast<float>(bbox.y) + (bbox.height - 1) / 2.0f};
  size = {static_cast<float>(bbox.width), static_cast<float>(bbox.height)};
  double w_z = size[0] + tconfig->CONTEXT_AMOUNT * (size[0] + size[1]);
  double h_z = size[1] + tconfig->CONTEXT_AMOUNT * (size[0] + size[1]);
  double s_z = std::round(std::sqrt(w_z * h_z));
  scaleaa = s_z;

  cv::Scalar mean = cv::mean(img);
  channel_average = mean;

  z_crop_blob = get_subwindow(img, center_pos, tconfig->EXEMPLAR_SIZE,
                              s_z, channel_average);
}

std::pair<std::vector<float>, size_t> PRLTracker::get_subwindow(
    const cv::Mat &im, std::vector<float> center_pos, int model_sz,
    int original_sz, cv::Scalar avg_chans) {

  if (center_pos.size() == 1) {
    center_pos.push_back(center_pos[0]);
  }

  cv::Size im_sz = im.size();
  float half_sz = (original_sz + 1) / 2.0f;

  float center_pos_x = center_pos[0];
  float center_pos_y = center_pos[1];

  float context_xmin = std::floor(center_pos_x - half_sz + 0.5f);
  float context_xmax = context_xmin + original_sz - 1;
  float context_ymin = std::floor(center_pos_y - half_sz + 0.5f);
  float context_ymax = context_ymin + original_sz - 1;

  int left_pad = static_cast<int>(std::max(0.0f, -context_xmin));
  int top_pad = static_cast<int>(std::max(0.0f, -context_ymin));
  int right_pad =
      static_cast<int>(std::max(0.0f, context_xmax - im_sz.width + 1));
  int bottom_pad =
      static_cast<int>(std::max(0.0f, context_ymax - im_sz.height + 1));

  context_xmin += left_pad;
  context_xmax += left_pad;
  context_ymin += top_pad;
  context_ymax += top_pad;

  cv::Mat te_im;
  cv::Mat im_patch;

  int r = im.rows;
  int c = im.cols;
  int channel = im.channels();

  if (top_pad > 0 || bottom_pad > 0 || left_pad > 0 || right_pad > 0) {
    cv::Size padded_size(c + left_pad + right_pad, r + top_pad + bottom_pad);
    te_im = cv::Mat::zeros(padded_size, im.type());

    cv::Rect roi_dst(left_pad, top_pad, c, r);
    im.copyTo(te_im(roi_dst));

    if (top_pad > 0) {
      cv::Rect top_region(left_pad, 0, c, top_pad);
      te_im(top_region).setTo(avg_chans);
    }

    if (bottom_pad > 0) {
      cv::Rect bottom_region(left_pad, r + top_pad, c, bottom_pad);
      te_im(bottom_region).setTo(avg_chans);
    }

    if (left_pad > 0) {
      cv::Rect left_region(0, 0, left_pad, r + top_pad + bottom_pad);
      te_im(left_region).setTo(avg_chans);
    }

    if (right_pad > 0) {
      cv::Rect right_region(c + left_pad, 0, right_pad,
                            r + top_pad + bottom_pad);
      te_im(right_region).setTo(avg_chans);
    }

    int y1 = static_cast<int>(context_ymin);
    int y2 = static_cast<int>(context_ymax + 1);
    int x1 = static_cast<int>(context_xmin);
    int x2 = static_cast<int>(context_xmax + 1);

    cv::Rect patch_roi(x1, y1, x2 - x1, y2 - y1);

    im_patch = te_im(patch_roi).clone();
  } else {
    int y1 = static_cast<int>(context_ymin);
    int y2 = static_cast<int>(context_ymax + 1);
    int x1 = static_cast<int>(context_xmin);
    int x2 = static_cast<int>(context_xmax + 1);

    cv::Rect patch_roi(x1, y1, x2 - x1, y2 - y1);
    im_patch = im(patch_roi);
  }

  if (model_sz != original_sz) {
    cv::resize(im_patch, im_patch, cv::Size(model_sz, model_sz));
  }

  cv::Mat im_patch_f = im_patch.clone();
  im_patch.convertTo(im_patch_f, CV_32FC3);

  cv::cvtColor(im_patch_f, im_patch_f, cv::COLOR_RGB2BGR);

  cv::Mat im_patch_contig = im_patch_f.clone();

  cv::Mat channels[3];
  cv::split(im_patch_contig, channels);
  cv::merge(channels, 3, im_patch_contig);

  cv::Mat blob;
  cv::dnn::blobFromImage(im_patch_contig, blob);

  size_t tensorSize = blob.total();
  std::vector<float> inputTensorValues(tensorSize);
  std::memcpy(inputTensorValues.data(), blob.ptr<float>(),
              tensorSize * sizeof(float));

  return std::pair<std::vector<float>, size_t>(inputTensorValues, tensorSize);
}

std::vector<float> PRLTracker::generate_anchor(const torch::Tensor &mapp, int output_size, float search_size) {
  std::function<torch::Tensor(torch::Tensor)> dcon = [](torch::Tensor x) {
    x = torch::clamp(x, -0.99, 0.99);
    return (torch::log1p(x) - torch::log1p(-x)) / 2;
  };

  const int size = output_size;
  const int total = size * size;

  torch::Tensor grid = torch::arange(0, size, torch::kFloat32);
  torch::Tensor x_base = (8 * grid + 63) - (search_size/ 2);
  torch::Tensor x = x_base.repeat({size}).reshape({-1});
  torch::Tensor y_base = (8 * grid + 63) - (search_size/ 2);
  torch::Tensor y = y_base.reshape({-1, 1}).repeat({1, size}).reshape({-1});
  torch::Tensor shap = dcon(mapp[0].detach().cpu()) * 143;
  torch::Tensor xx =
      torch::arange(0, size, torch::kLong).repeat({size}).reshape({-1});
  torch::Tensor yy = torch::arange(0, size, torch::kLong)
                         .reshape({-1, 1})
                         .repeat({1, size})
                         .reshape({-1});
  torch::Tensor shap0 = shap.index({0, yy, xx});
  torch::Tensor shap1 = shap.index({1, yy, xx});
  torch::Tensor shap2 = shap.index({2, yy, xx});
  torch::Tensor shap3 = shap.index({3, yy, xx});

  torch::Tensor w = shap0 + shap1;
  torch::Tensor h = shap2 + shap3;
  x = x - shap0 + w / 2;
  y = y - shap2 + h / 2;

  torch::Tensor anchor = torch::zeros({total, 4}, torch::kFloat32);
  anchor.index_put_({torch::indexing::Slice(), 0}, x);
  anchor.index_put_({torch::indexing::Slice(), 1}, y);
  anchor.index_put_({torch::indexing::Slice(), 2}, w);
  anchor.index_put_({torch::indexing::Slice(), 3}, h);

  torch::Tensor anchor_contiguous = anchor.contiguous();
  const float *data = anchor_contiguous.data_ptr<float>();
  std::vector<float> result(data, data + anchor.numel());

  return result;
}

torch::Tensor PRLTracker::_convert_bbox(torch::Tensor delta,
                                        torch::Tensor anchor_tensor) {
  delta = delta.permute({1, 2, 3, 0}).contiguous().view({4, -1});
  anchor_tensor =
      anchor_tensor.transpose(0, 1).to(delta.device()).to(delta.dtype());

  auto anchor_x = anchor_tensor.select(1, 0);
  auto anchor_y = anchor_tensor.select(1, 1);
  auto anchor_w = anchor_tensor.select(1, 2);
  auto anchor_h = anchor_tensor.select(1, 3);

  auto dx = delta[0] * anchor_w + anchor_x;
  auto dy = delta[1] * anchor_h + anchor_y;
  auto dw = torch::exp(delta[2]) * anchor_w;
  auto dh = torch::exp(delta[3]) * anchor_h;

  auto out = torch::stack({dx, dy, dw, dh}, 0);

  return out;
}

at::Tensor PRLTracker::_convert_score(torch::Tensor &score) {
  at::Tensor s = score
                     .permute({1, 2, 3, 0})
                     .contiguous()
                     .view({2, -1})
                     .permute({1, 0});

  return torch::softmax(s, /*dim=*/1)
      .select(/*dim=*/1, /*index=*/1)
      .to(torch::kCPU)
      .contiguous();
}

std::vector<float> PRLTracker::_bbox_clip(float cx, float cy, float width,
                                          float height,
                                          std::vector<int> boundary) {
  cx = std::max(0.0f, std::min(static_cast<float>(boundary[1]), cx));
  cy = std::max(0.0f, std::min(static_cast<float>(boundary[0]), cy));
  width = std::max(10.0f, std::min(static_cast<float>(boundary[1]), width));
  height = std::max(10.0f, std::min(static_cast<float>(boundary[0]), height));
  return {cx, cy, width, height};
}

std::pair<cv::Rect, float> PRLTracker::track(cv::Mat img, track_prl_config* tconfig) {
  long double w_z =
      size[0] + tconfig->CONTEXT_AMOUNT * (size[0] + size[1]);
  long double h_z =
      size[1] + tconfig->CONTEXT_AMOUNT * (size[0] + size[1]);
  long double s_z = std::sqrt(w_z * h_z);

  if (size[0] * size[1] > 0.5f * img.rows * img.cols) {
    s_z = scaleaa;
  }

  double scale_z = tconfig->EXEMPLAR_SIZE / s_z;
  double s_x =
      s_z * (tconfig->INSTANCE_SIZE / tconfig->EXEMPLAR_SIZE);

  Ort::Value z_crop = Ort::Value(Ort::Value::CreateTensor<float>(
      *memoryInfo, z_crop_blob.first.data(), z_crop_blob.second,
      dims_template->data(), dims_template->size()));

  std::pair<std::vector<float>, size_t> x_crop_blob =
      get_subwindow(img, center_pos, tconfig->INSTANCE_SIZE,
                    std::round(s_x), channel_average);

  Ort::Value x_crop = Ort::Value(Ort::Value::CreateTensor<float>(
      *memoryInfo, x_crop_blob.first.data(), x_crop_blob.second,
      dims_search->data(), dims_search->size()));

  const auto &type_info_img_zf = z_crop.GetTensorTypeAndShapeInfo();
  auto shape_img_zf = type_info_img_zf.GetShape();

  float *template_data = z_crop.GetTensorMutableData<float>();
  float *search_data = x_crop.GetTensorMutableData<float>();

  std::vector<Ort::Value> inputTensors;

  inputTensors.push_back(std::move(z_crop));
  inputTensors.push_back(std::move(x_crop));

  std::vector<Ort::AllocatedStringPtr> inputNamePtrs;
  inputNamePtrs.push_back(session.GetInputNameAllocated(0, allocator));
  inputNamePtrs.push_back(session.GetInputNameAllocated(1, allocator));

  std::vector<const char *> inputNamesCStr;
  for (const auto &ptr : inputNamePtrs) inputNamesCStr.push_back(ptr.get());

  std::vector<Ort::AllocatedStringPtr> outputNamePtrs;
  outputNamePtrs.push_back(session.GetOutputNameAllocated(0, allocator));
  outputNamePtrs.push_back(session.GetOutputNameAllocated(1, allocator));
  outputNamePtrs.push_back(session.GetOutputNameAllocated(2, allocator));

  std::vector<const char *> outputNamesCStr;
  for (const auto &ptr : outputNamePtrs) outputNamesCStr.push_back(ptr.get());

  std::vector<Ort::Value> outputTensors = session.Run(
      Ort::RunOptions{nullptr}, inputNamesCStr.data(), inputTensors.data(),
      inputNamesCStr.size(), outputNamesCStr.data(), outputNamesCStr.size());

  float *loc = outputTensors[0].GetTensorMutableData<float>();
  float *cls1 = outputTensors[1].GetTensorMutableData<float>();
  float *cls2 = outputTensors[2].GetTensorMutableData<float>();

  auto loc_shape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
  auto cls1_shape = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
  auto cls2_shape = outputTensors[2].GetTensorTypeAndShapeInfo().GetShape();

  torch::Tensor loc_tensor =
      torch::from_blob(loc, loc_shape, torch::kFloat32).clone();

  torch::Tensor cls1_tensor =
      torch::from_blob(cls1, cls1_shape, torch::kFloat32).clone();
  torch::Tensor cls2_tensor =
      torch::from_blob(cls2, cls2_shape, torch::kFloat32).clone();

  std::vector<float> predbox = generate_anchor(loc_tensor, tconfig->OUTPUT_SIZE, tconfig->SEARCH_SIZE);

  at::Tensor score1 = _convert_score(cls1_tensor) * tconfig->w2;
  at::Tensor score2 = (cls2_tensor.view({-1}) * tconfig->w3)
                          .to(torch::kCPU);

  at::Tensor score = (score1 + score2) * 0.5f;

  int num_anchors = score_size * score_size;

  torch::Tensor predbox_tensor =
      torch::from_blob(predbox.data(), {num_anchors, 4}, torch::kFloat32)
          .clone();

  torch::Tensor predbox_transposed = predbox_tensor.transpose(0, 1).clone();

  auto change = [](const torch::Tensor &r) {
    return torch::max(r, 1.0 / (r + 1e-5));
  };

  auto sz = [](float w, float h) {
    float pad = 0.5f * (w + h);
    return std::sqrt((w + pad) * (h + pad));
  };

  auto sz_tensor = [](const torch::Tensor &w, const torch::Tensor &h) {
    torch::Tensor pad = 0.5 * (w + h);
    return torch::sqrt((w + pad) * (h + pad));
  };
  const int num_anchor_r = predbox_transposed.size(1);

  float target_sz = sz(size[0] * scale_z, size[1] * scale_z);
  torch::Tensor pw = predbox_transposed[2];
  torch::Tensor ph = predbox_transposed[3];
  torch::Tensor sz_pred = sz_tensor(pw, ph);
  torch::Tensor s_c = sz_pred / target_sz;

  s_c = change(s_c);
  float target_ratio = size[0] / (size[1] + 1e-5f);
  torch::Tensor r_c = (target_ratio) / (pw / (ph + 1e-5));
  r_c = change(r_c);
  torch::Tensor penalty =
      torch::exp(-(r_c * s_c - 1) * tconfig->PENALTY_K);
  torch::Tensor pscore = penalty * score;

  torch::Tensor window_tensor =
      torch::from_blob(const_cast<float *>(window.data()),
                       {score_size * score_size},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .clone();

  pscore = pscore * (1 - tconfig->WINDOW_INFLUENCE) +
           window_tensor * tconfig->WINDOW_INFLUENCE;
  int best_idx = pscore.argmax().item<int>();
  float best_score = score[best_idx].item<float>();

  torch::Tensor bbox =
      predbox_transposed.index({torch::indexing::Slice(), best_idx})
          .div(scale_z);

  float lr = penalty[best_idx].item<float>() * best_score * tconfig->LR;
  float cx = bbox[0].item<float>() + center_pos[0];
  float cy = bbox[1].item<float>() + center_pos[1];
  float width = size[0] * (1 - lr) + bbox[2].item<float>() * lr;
  float height = size[1] * (1 - lr) + bbox[3].item<float>() * lr;

  std::vector<float> clipped =
      _bbox_clip(cx, cy, width, height, {img.cols, img.rows});
  center_pos = {clipped[0], clipped[1]};
  size = {clipped[2], clipped[3]};

  cv::Rect bbox_rect(static_cast<int>(center_pos[0] - size[0] / 2),
                     static_cast<int>(center_pos[1] - size[1] / 2),
                     static_cast<int>(size[0]), static_cast<int>(size[1]));

  return {bbox_rect, best_score};
}

#include "layer.h"
#include "net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <map>
#include <algorithm> // std::max

// ====================== Config ========================

float pixel_mean[3] = {0, 0, 0}; // 均值
float pixel_std[3] = {1, 1, 1};  // 方差
bool dense_anchor = false;
float cls_threshold = 0.8;
float nms_threshold = 0.4;

// ====================== Config End ========================

// ======================== CRect2f ==========================

class CRect2f
{
public:
    CRect2f(float x1, float y1, float x2, float y2)
    {
        val[0] = x1;
        val[1] = y1;
        val[2] = x2;
        val[3] = y2;
    }

    float& operator[](int i)
    {
        return val[i];
    }

    float operator[](int i) const
    {
        return val[i];
    }

    float val[4];

    void print()
    {
        // printf("rect %f %f %f %f\n", val[0], val[1], val[2], val[3]);
    }
};

// ======================== CRect2f End ==========================

// ======================== Anchor ==========================

class Anchor
{
public:
    Anchor()
    {
    }

    ~Anchor()
    {
    }

    bool operator<(const Anchor& t) const
    {
        return score < t.score;
    }

    bool operator>(const Anchor& t) const
    {
        return score > t.score;
    }

    float& operator[](int i)
    {
        assert(0 <= i && i <= 4);

        if (i == 0)
            return finalbox.x;
        if (i == 1)
            return finalbox.y;
        if (i == 2)
            return finalbox.width;
        if (i == 3)
            return finalbox.height;
    }

    float operator[](int i) const
    {
        assert(0 <= i && i <= 4);

        if (i == 0)
            return finalbox.x;
        if (i == 1)
            return finalbox.y;
        if (i == 2)
            return finalbox.width;
        if (i == 3)
            return finalbox.height;
    }

    cv::Rect_<float> anchor;      // x1,y1,x2,y2
    float reg[4];                 // offset reg
    cv::Point center;             // anchor feat center
    float score;                  // cls score
    std::vector<cv::Point2f> pts; // pred pts

    cv::Rect_<float> finalbox; // final box res

    void print()
    {
        printf("finalbox %f %f %f %f, score %f\n", finalbox.x, finalbox.y,
               finalbox.width, finalbox.height, score);
        printf("landmarks ");
        for (int i = 0; i < pts.size(); ++i)
        {
            printf("%f %f, ", pts[i].x, pts[i].y);
        }
        printf("\n");
    }
};

// ======================== Anchor End ==========================

// ======================== AnchorCfg ==========================
class AnchorCfg
{
public:
    std::vector<float> SCALES;
    std::vector<float> RATIOS;
    int BASE_SIZE;

    AnchorCfg()
    {
    }
    ~AnchorCfg()
    {
    }
    AnchorCfg(const std::vector<float> s, const std::vector<float> r, int size)
    {
        SCALES = s;
        RATIOS = r;
        BASE_SIZE = size;
    }
};

// ======================== AnchorEnd ==========================

// ====================== AnchorGenerator ======================

class AnchorGenerator
{
public:
    AnchorGenerator();
    ~AnchorGenerator();

    // init different anchors
    int Init(int stride, const AnchorCfg& cfg, bool dense_anchor);

    // anchor plane
    int Generate(int fwidth, int fheight, int stride, float step,
                 std::vector<int>& size, std::vector<float>& ratio,
                 bool dense_anchor);

    // filter anchors and return valid anchors
    int FilterAnchor(ncnn::Mat& cls, ncnn::Mat& reg, ncnn::Mat& pts,
                     std::vector<Anchor>& result);

private:
    void _ratio_enum(const CRect2f& anchor, const std::vector<float>& ratios,
                     std::vector<CRect2f>& ratio_anchors);

    void _scale_enum(const std::vector<CRect2f>& ratio_anchor,
                     const std::vector<float>& scales,
                     std::vector<CRect2f>& scale_anchors);

    void bbox_pred(const CRect2f& anchor, const CRect2f& delta,
                   cv::Rect_<float>& box);

    void landmark_pred(const CRect2f anchor,
                       const std::vector<cv::Point2f>& delta,
                       std::vector<cv::Point2f>& pts);

    std::vector<std::vector<Anchor> > anchor_planes; // corrspont to channels

    std::vector<int> anchor_size;
    std::vector<float> anchor_ratio;
    float anchor_step; // scale step
    int anchor_stride; // anchor tile stride
    int feature_w;     // feature map width
    int feature_h;     // feature map height

    std::vector<CRect2f> preset_anchors;
    int anchor_num; // anchor type num
};

AnchorGenerator::AnchorGenerator()
{
}

AnchorGenerator::~AnchorGenerator()
{
}

// anchor plane
int AnchorGenerator::Generate(int fwidth, int fheight, int stride, float step,
                              std::vector<int>& size, std::vector<float>& ratio,
                              bool dense_anchor)
{
    /*
  anchor_planes.resize(anchor_num);
  cv::Mat xs = cv::Mat(fheight, fwidth, CV_32FC1);
  cv::Mat ys = cv::Mat(fheight, fwidth, CV_32FC1);
  for (int w = 0; w < fwidth; ++w) {
      xs.col(w).setTo(float(w));
  }
  for (int h = 0; h < fheight; ++h) {
      ys.row(w).setTo(float(h));
  }
  xs = xs * stride;
  ys = ys * stride;

  for (int i = 0; i < anchor_num; ++i) {
      anchor_planes[i] = std::vector<cv::Mat>(4);
      anchor_planes[i][0] = xs + anchors[i][0];
      anchor_planes[i][1] = ys + anchors[i][1];
      anchor_planes[i][2] = xs + anchors[i][2];
      anchor_planes[i][3] = ys + anchors[i][3];
  }
  */

    return 0;
}

// init different anchors
int AnchorGenerator::Init(int stride, const AnchorCfg& cfg, bool dense_anchor)
{
    CRect2f base_anchor(0, 0, cfg.BASE_SIZE - 1, cfg.BASE_SIZE - 1);
    std::vector<CRect2f> ratio_anchors;
    // get ratio anchors
    _ratio_enum(base_anchor, cfg.RATIOS, ratio_anchors);
    _scale_enum(ratio_anchors, cfg.SCALES, preset_anchors);

    // save as x1,y1,x2,y2
    if (dense_anchor)
    {
        assert(stride % 2 == 0);
        int num = preset_anchors.size();
        for (int i = 0; i < num; ++i)
        {
            CRect2f anchor = preset_anchors[i];
            preset_anchors.push_back(
                CRect2f(anchor[0] + int(stride / 2), anchor[1] + int(stride / 2),
                        anchor[2] + int(stride / 2), anchor[3] + int(stride / 2)));
        }
    }

    anchor_stride = stride;

    anchor_num = preset_anchors.size();
    // std::cout << "anchor_num=" << anchor_num << std::endl;
    for (int i = 0; i < anchor_num; ++i)
    {
        preset_anchors[i].print();
    }
    return anchor_num;
}

int AnchorGenerator::FilterAnchor(ncnn::Mat& cls, ncnn::Mat& reg,
                                  ncnn::Mat& pts, std::vector<Anchor>& result)
{
    assert(cls.c == anchor_num * 2);
    assert(reg.c == anchor_num * 4);
    int pts_length = 0;

    assert(pts.c % anchor_num == 0);
    pts_length = pts.c / anchor_num / 2;

    int w = cls.w;
    int h = cls.h;

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            int id = i * w + j;
            for (int a = 0; a < anchor_num; ++a)
            {
                //            	std::cout<< j << i << id<<cls.channel(anchor_num +
                //            a)[id]<<",";
                if (cls.channel(anchor_num + a)[id] >= cls_threshold)
                {
                    // printf("cls %f\n", cls.channel(anchor_num + a)[id]);
                    CRect2f box(j * anchor_stride + preset_anchors[a][0],
                                i * anchor_stride + preset_anchors[a][1],
                                j * anchor_stride + preset_anchors[a][2],
                                i * anchor_stride + preset_anchors[a][3]);
                    // printf("%f %f %f %f\n", box[0], box[1], box[2], box[3]);
                    CRect2f delta(reg.channel(a * 4 + 0)[id], reg.channel(a * 4 + 1)[id],
                                  reg.channel(a * 4 + 2)[id], reg.channel(a * 4 + 3)[id]);

                    Anchor res;
                    res.anchor = cv::Rect_<float>(box[0], box[1], box[2], box[3]);
                    bbox_pred(box, delta, res.finalbox);
                    // printf("bbox pred\n");
                    res.score = cls.channel(anchor_num + a)[id];
                    res.center = cv::Point(j, i);

                    // printf("center %d %d\n", j, i);

                    if (1)
                    {
                        std::vector<cv::Point2f> pts_delta(pts_length);
                        for (int p = 0; p < pts_length; ++p)
                        {
                            pts_delta[p].x = pts.channel(a * pts_length * 2 + p * 2)[id];
                            pts_delta[p].y = pts.channel(a * pts_length * 2 + p * 2 + 1)[id];
                        }
                        // printf("ready landmark_pred\n");
                        landmark_pred(box, pts_delta, res.pts);
                        // printf("landmark_pred\n");
                    }
                    result.push_back(res);
                }
            }
        }
    }

    return 0;
}

void AnchorGenerator::_ratio_enum(const CRect2f& anchor,
                                  const std::vector<float>& ratios,
                                  std::vector<CRect2f>& ratio_anchors)
{
    float w = anchor[2] - anchor[0] + 1;
    float h = anchor[3] - anchor[1] + 1;
    float x_ctr = anchor[0] + 0.5 * (w - 1);
    float y_ctr = anchor[1] + 0.5 * (h - 1);

    ratio_anchors.clear();
    float sz = w * h;
    for (int s = 0; s < ratios.size(); ++s)
    {
        float r = ratios[s];
        float size_ratios = sz / r;
        float ws = std::sqrt(size_ratios);
        float hs = ws * r;
        ratio_anchors.push_back(
            CRect2f(x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                    x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)));
    }
}

void AnchorGenerator::_scale_enum(const std::vector<CRect2f>& ratio_anchor,
                                  const std::vector<float>& scales,
                                  std::vector<CRect2f>& scale_anchors)
{
    scale_anchors.clear();
    for (int a = 0; a < ratio_anchor.size(); ++a)
    {
        CRect2f anchor = ratio_anchor[a];
        float w = anchor[2] - anchor[0] + 1;
        float h = anchor[3] - anchor[1] + 1;
        float x_ctr = anchor[0] + 0.5 * (w - 1);
        float y_ctr = anchor[1] + 0.5 * (h - 1);

        for (int s = 0; s < scales.size(); ++s)
        {
            float ws = w * scales[s];
            float hs = h * scales[s];
            scale_anchors.push_back(
                CRect2f(x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                        x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)));
        }
    }
}

void AnchorGenerator::bbox_pred(const CRect2f& anchor, const CRect2f& delta,
                                cv::Rect_<float>& box)
{
    float w = anchor[2] - anchor[0] + 1;
    float h = anchor[3] - anchor[1] + 1;
    float x_ctr = anchor[0] + 0.5 * (w - 1);
    float y_ctr = anchor[1] + 0.5 * (h - 1);

    float dx = delta[0];
    float dy = delta[1];
    float dw = delta[2];
    float dh = delta[3];

    float pred_ctr_x = dx * w + x_ctr;
    float pred_ctr_y = dy * h + y_ctr;
    float pred_w = std::exp(dw) * w;
    float pred_h = std::exp(dh) * h;

    box = cv::Rect_<float>(
        pred_ctr_x - 0.5 * (pred_w - 1.0), pred_ctr_y - 0.5 * (pred_h - 1.0),
        pred_ctr_x + 0.5 * (pred_w - 1.0), pred_ctr_y + 0.5 * (pred_h - 1.0));
}

void AnchorGenerator::landmark_pred(const CRect2f anchor,
                                    const std::vector<cv::Point2f>& delta,
                                    std::vector<cv::Point2f>& pts)
{
    float w = anchor[2] - anchor[0] + 1;
    float h = anchor[3] - anchor[1] + 1;
    float x_ctr = anchor[0] + 0.5 * (w - 1);
    float y_ctr = anchor[1] + 0.5 * (h - 1);

    pts.resize(delta.size());
    for (int i = 0; i < delta.size(); ++i)
    {
        pts[i].x = delta[i].x * w + x_ctr;
        pts[i].y = delta[i].y * h + y_ctr;
    }
}

// ====================== AnchorGenerator End ======================

// ================ InitData =================================
std::vector<int> _feat_stride_fpn = {32, 16, 8};
std::map<int, AnchorCfg> anchor_cfg = {
    {32, AnchorCfg(std::vector<float>{32, 16}, std::vector<float>{1}, 16)},
    {16, AnchorCfg(std::vector<float>{8, 4}, std::vector<float>{1}, 16)},
    {8, AnchorCfg(std::vector<float>{2, 1}, std::vector<float>{1}, 16)}};

// ================ InitData End =================================

// ====================== Utils ========================

// 把图短边进行填充为正方形,并进行缩放.保证缩放后的图横纵比不变
std::pair<cv::Mat, float> square_crop(cv::Mat im, int S)
{
    // im: 传入原始图片
    // S: 要缩放到的尺寸
    // return: <修正后的图片,缩放比例>
    int height, width;
    float scale; //缩放比例
    if (im.rows > im.cols)
    {
        height = S;
        width = int(float(im.cols) / im.rows * S);
        scale = float(S) / im.rows;
    }
    else
    {
        width = S;
        height = int(float(im.rows) / im.cols * S);
        scale = float(S) / im.cols;
    }
    cv::Mat resized_im;
    cv::resize(im, resized_im, cv::Size(width, height), 0, 0, cv::INTER_AREA);
    cv::Mat det_im = cv::Mat::zeros(cv::Size(S, S), im.type());
    cv::Rect roi(0, 0, resized_im.cols, resized_im.rows);
    resized_im.clone().copyTo(det_im(roi));
    std::pair<cv::Mat, float> pp(det_im, scale);
    return pp;
}

// pic_transform
std::pair<cv::Mat, cv::Mat> pic_transform(cv::Mat img, cv::Point center, float scale,
                                          int image_size, float rotate)
{
    float rot = (rotate * M_PI) / 180.0;
    float cx = center.x * scale;
    float cy = center.y * scale;
    cv::Mat t1 = (cv::Mat_<float>(3, 3) << scale, -0., 0., 0., scale, 0., 0., 0., 1.0);
    cv::Mat t2 = (cv::Mat_<float>(3, 3) << 0, -0., -1 * cx, 0., 0., -1 * cy, 0., 0., 1.0);
    cv::Mat t3 = (cv::Mat_<float>(3, 3) << 0, -0., 0., 0, 0, 0., 0., 0., 1.0);
    cv::Mat t4 = (cv::Mat_<float>(3, 3) << 0, -0., float(image_size) / 2, 0, 0,
                  float(image_size) / 2, 0., 0., 1.0);
    cv::Mat t = t1 + t2 + t3 + t4;
    cv::Mat M = t.rowRange(0, 2); //只要前2行
    // cout << "M=" << M << endl;
    cv::Mat cropped;
    cv::warpAffine(img, cropped, M, cv::Size(image_size, image_size));
    // cv::imshow("cropped", cropped);
    // cv::waitKey(0);
    return std::make_pair(cropped, M);
}

// Summary:把输出存入容器中,元素为点
// Parameters:
//  out: 特征点网络2d06det输出,相对于仿射变化后的rimg下的相对坐标(-1,1)
// Return: rimg图下的特征点放入容器中
std::vector<cv::Point> get_point_2d(ncnn::Mat out)
{
    std::vector<cv::Point> outpoint; // 特征点容器,元素为点
    int xx, yy;
    const float* ptr = out.channel(0);
    for (int y = 0; y < out.h; y++)
    {
        for (int x = 0; x < out.w; x++)
        {
            if (x % 2 == 0)
            {
                xx = int((ptr[x] + 1) * (192.0 / 2));
            }
            else
            {
                yy = int((ptr[x] + 1) * (192.0 / 2));
                outpoint.push_back(cv::Point(xx, yy));
            }
        }
        ptr += out.w;
    }
    return outpoint;
}

// 把输出点通过仿射变换的矩阵逆矩阵,得到原图的坐标点
// points: 仿射图rimg下特征点的
// IM: 仿射变换矩阵的逆矩阵
// return: 原图下的特征点坐标点
std::vector<cv::Point> trans_point(std::vector<cv::Point> points, cv::Mat IM)
{
    std::vector<cv::Point> new_pts;
    for (int i = 0; i < points.size(); i++)
    {
        cv::Point pt = points[i];
        cv::Mat new_pt = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1.0);
        new_pt = IM * new_pt;
        cv::Point p(new_pt.at<float>(0, 0), new_pt.at<float>(1, 0));
        new_pts.push_back(p);
    }
    return new_pts;
}

// recover_point
std::pair<cv::Rect, std::vector<cv::Point2f> > recover_point(Anchor face, float det_scale)
{
    cv::Rect box = face.finalbox;            // 人脸框x1 y1 x2 y2 score
    std::vector<cv::Point2f> pts = face.pts; // 5个 x y
    float x1 = box.x / det_scale;
    float y1 = box.y / det_scale;
    float x2 = box.width / det_scale;
    float y2 = box.height / det_scale;
    cv::Rect new_box(x1, y1, x2, y2);
    std::vector<cv::Point2f> new_landmark;
    for (int i = 0; i < pts.size(); i++)
    {
        cv::Point2f point = pts[i];
        cv::Point2f p(point.x / det_scale, point.y / det_scale);
        new_landmark.push_back(p);
    }
    return std::make_pair(new_box, new_landmark);
}

void nms_cpu(std::vector<Anchor>& boxes, float threshold,
             std::vector<Anchor>& filterOutBoxes)
{
    filterOutBoxes.clear();
    if (boxes.size() == 0)
        return;
    std::vector<size_t> idx(boxes.size());

    for (unsigned i = 0; i < idx.size(); i++)
    {
        idx[i] = i;
    }

    // descending sort
    sort(boxes.begin(), boxes.end(), std::greater<Anchor>());

    while (idx.size() > 0)
    {
        int good_idx = idx[0];
        filterOutBoxes.push_back(boxes[good_idx]);

        std::vector<size_t> tmp = idx;
        idx.clear();
        for (unsigned i = 1; i < tmp.size(); i++)
        {
            int tmp_i = tmp[i];
            float inter_x1 = std::max(boxes[good_idx][0], boxes[tmp_i][0]);
            float inter_y1 = std::max(boxes[good_idx][1], boxes[tmp_i][1]);
            float inter_x2 = std::min(boxes[good_idx][2], boxes[tmp_i][2]);
            float inter_y2 = std::min(boxes[good_idx][3], boxes[tmp_i][3]);

            float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
            float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

            float inter_area = w * h;
            float area_1 = (boxes[good_idx][2] - boxes[good_idx][0] + 1) * (boxes[good_idx][3] - boxes[good_idx][1] + 1);
            float area_2 = (boxes[tmp_i][2] - boxes[tmp_i][0] + 1) * (boxes[tmp_i][3] - boxes[tmp_i][1] + 1);
            float o = inter_area / (area_1 + area_2 - inter_area);
            if (o <= threshold)
                idx.push_back(tmp_i);
        }
    }
}

void draw_points(cv::Mat& img, std::vector<std::vector<cv::Point> > outpoints)
{
    // 绘制特征点
    // img: 要绘制的图板
    // outpoints:所有人脸的特征点
    for (int i = 0; i < outpoints.size(); i++)
    {
        std::vector<cv::Point> outpoint = outpoints[i];
        for (int j = 0; j < outpoint.size(); j++)
        {
            cv::circle(img, outpoint[j], 2, cv::Scalar(0, 0, 255), -1, 8, 0);
        }
    }
}

// ====================== Utils End ========================

int main(int args, char** argv)
{
    std::cout << "Face Landmark Detection Begin" << std::endl;

    // 人脸检测模型
    std::string param_path = "retina.param";
    std::string model_path = "retina.bin";
    ncnn::Net _net;
    int ret_param = _net.load_param(param_path.data());
    int ret_model = _net.load_model(model_path.data());
    if (ret_param == 0 && ret_model == 0)
    {
        std::cout << "人脸检测模型 Retinaface 加载成功" << std::endl;
    }
    else
    {
        std::cout << "人脸检测模型 Retinaface 加载失败" << std::endl;
        return -1;
    }

    // 加载特征点检测模型
    ncnn::Net det_net;
    int det_param = det_net.load_param("2d106det.param");
    int det_model = det_net.load_model("2d106det.bin");
    if (det_param == 0 && det_model == 0)
    {
        std::cout << "人脸关键点检测模型 2d106det 加载成功" << std::endl;
    }
    else
    {
        std::cout << "人脸关键点检测模型 2d106det 加载失败" << std::endl;
        return -1;
    }

    // extern float pixel_mean[3]; //均值,在config规定了,这里无法改变值
    // extern float pixel_std[3];  //方差,在config规定了,这里无法改变值

    cv::Mat img = cv::imread("/Users/ringcrl/Documents/assets/02-图片/01.jpg");
    if (!img.data)
    {
        printf("load img error");
        return -1;
    }

    cv::Mat img_c = img.clone();
    int detim_size = 640;                                                 // 原始照片要缩放后的尺寸
    std::pair<cv::Mat, float> detIm_scala = square_crop(img, detim_size); // 进过 square_crop 保证图片保持衡纵比
    cv::Mat det_im = detIm_scala.first;                                   // 缩放后的图 640x640
    float det_scale = detIm_scala.second;                                 // 缩放比例
    std::cout << "det_scale: " << det_scale << std::endl;

    ncnn::Mat input = ncnn::Mat::from_pixels(
        det_im.data, ncnn::Mat::PIXEL_BGR2RGB, det_im.cols, det_im.rows);

    input.substract_mean_normalize(pixel_mean, pixel_std);
    ncnn::Extractor _extractor = _net.create_extractor();
    _extractor.set_light_mode(true);
    _extractor.set_num_threads(4);
    _extractor.input("data", input); // 可以从.param查看网络结构,查看输入名称

    std::vector<AnchorGenerator> ac(
        _feat_stride_fpn.size()); // _feat_stride_fpn = {32, 16, 8}  .size=3
    for (int i = 0; i < _feat_stride_fpn.size(); ++i)
    {
        int stride = _feat_stride_fpn[i]; // 32, 16, 8
        ac[i].Init(stride, anchor_cfg[stride], false);
    }

    std::vector<Anchor> proposals;
    proposals.clear();

    for (int i = 0; i < _feat_stride_fpn.size(); ++i)
    {
        ncnn::Mat cls;
        ncnn::Mat reg;
        ncnn::Mat pts;

        // get blob output
        char clsname[100];
        sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d",
                _feat_stride_fpn[i]); // sprintf 打印到字符串中,即赋值
        char regname[100];
        sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
        char ptsname[100];
        sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
        _extractor.extract(clsname, cls); // 得到分类
        _extractor.extract(regname, reg); // 得到目标框
        _extractor.extract(ptsname, pts); // 得到特征点

        ac[i].FilterAnchor(cls, reg, pts, proposals); // 筛选
    }

    // nms
    std::vector<Anchor> result;
    nms_cpu(proposals, nms_threshold,
            result);                              // 得到最终的人脸,  原函数result传入的是地址,
    printf("final result %lld\n", result.size()); //人脸个数
    // result = choose_one(result);

    std::vector<std::vector<cv::Point> > all_points;
    for (int i = 0; i < result.size(); i++) //遍历人脸
    {
        result[i].print();
        Anchor face = result[i];                                                                  // 这里的人脸还是在det_im(640x640)图中的坐标
        std::pair<cv::Rect, std::vector<cv::Point2f> > new_face = recover_point(face, det_scale); // 得到原始img下的人脸框信息
        cv::Rect new_box = new_face.first;                                                        //  img下人脸框
        std::vector<cv::Point2f> new_pts = new_face.second;                                       // img下的5个特征点
        float x1 = new_box.x, y1 = new_box.y;
        float x2 = new_box.width, y2 = new_box.height;

        float w = x2 - x1, h = y2 - y1;
        cv::Point center((x1 + x2) / 2, (y1 + y2) / 2); //得到人脸框的中心
        // cout << "center=" << center;
        float rotate = 0;
        float _scale = 192 * 2 / 3.0 / std::max(w, h);
        std::pair<cv::Mat, cv::Mat> rimg_M = pic_transform(img, center, _scale, 192, rotate);
        cv::Mat rimg = rimg_M.first;                 //仿射变换后192x192的图
        cv::Mat M = rimg_M.second;                   //仿射变换矩阵
        cv::cvtColor(rimg, rimg, cv::COLOR_BGR2RGB); //可有可无
        ncnn::Mat indet = ncnn::Mat::from_pixels(rimg.data, ncnn::Mat::PIXEL_BGR, rimg.cols,
                                                 rimg.rows); // rimg已经是192x192x3
        ncnn::Extractor exdet = det_net.create_extractor();
        exdet.set_light_mode(true);
        exdet.set_num_threads(4);
        // indet.substract_mean_normalize(pixel_mean, pixel_std);
        // //数据标准化,可有可无
        exdet.input("data", indet);
        ncnn::Mat outdet;
        exdet.extract("fc1", outdet);
        cv::Mat IM;
        cv::invertAffineTransform(M, IM); //得到仿射变换逆矩阵
        std::vector<cv::Point> pred = get_point_2d(outdet);
        std::vector<cv::Point> outpoint = trans_point(pred, IM);
        all_points.push_back(outpoint);
    }

    std::cout << "人脸数" << all_points.size() << std::endl;
    std::cout << "单人脸点数" << all_points[0].size() << std::endl;

    draw_points(img_c, all_points); //绘制 关键点

    cv::namedWindow("img", 3);
    cv::imshow("img", img_c);
    cv::waitKey(0);
    return 0;
}

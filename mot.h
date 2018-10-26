#ifndef MOT_H_
#define MOT_H_

#include <opencv2/opencv.hpp>
#include <io.h>
#include <direct.h>
#include <iostream>
#include <limits>
#include <vector>
#include <map>


#define STR_LEN 256
#define Q_SIZE 21
#define SAVE_IMG

typedef struct ObjRect {
    int x1;
    int y1;
    int x2;
    int y2;
} ObjRect;

typedef struct MatchInfo {
    int match_id;
    int match_frm;
} MatchInfo;

typedef struct ObjInfo {
    int frm;
    int id;
    int match_id;
    int pre_match_id;  // for track only
    int pre_match_frm; // for track only
    ObjRect rect;
} ObjInfo;

typedef struct MotResult {
    int gt;     // ground truth
    int gt_unq; // num of track
    int tp;     // true positive
    int fn;     // false negative, miss
    int fp;     // false positive
    int mmc;    // mismatch
    float d;    // distance
    float MOTA; // 1-(fp+fn+mmc)/gt --> [1-(fp+fn)/gt - mmc/gt_unq]
    float MOTP; // d/tp
} MotResult;

typedef struct FrameInfo {
    cv::Mat frame;
    int idx;
} FrameInfo;

typedef struct FrameQueue {
    FrameInfo frames[Q_SIZE];
    int front;
    int rear;
} FrameQueue;


typedef std::map<int, MatchInfo> match_map;
typedef match_map::iterator match_iter;
typedef std::vector<ObjInfo> obj_infos;
typedef obj_infos::iterator obj_iter;
typedef std::vector<obj_infos> obj_infos_vec;


class findx {
 public:
    findx(int id) : id_() {}

    bool operator()(ObjInfo &objvec) {
        if (objvec.id == id_)
            return true;
        return false;
    }

 private:
    int id_;
};


bool ReadData(char *fname, obj_infos_vec &data_vec);
void DrawCurve(std::vector<float> &fvars, std::vector<std::string> &notes);
void ShowMatch(char *vdoname, obj_infos &mark_vec, obj_infos &track_vec, int frm, int delay = 1);

#endif  // MOT_H_

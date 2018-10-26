/***************************************************************
## Evaluating Multiple Object Tracking Performance
   --- Implementation of The CLEAR-MOT metric
## Author: Andrew Lee
## Update: May 24th, 2017
## using this mark format:
   frame_idx obj_num (id x1 y1 x2 y2) (id x1 y1 x2 y2) ...
   (do not appear this line if obj_num is zero)
## Email: code.lilei@gmail.com
***************************************************************/

#include <string.h>
#include "mot.h"

using namespace std;
using namespace cv;

match_map kMatchPre;


float CalcIou(const ObjRect &src, const ObjRect &dst) {
    int x1 = MAX(src.x1, dst.x1);
    int y1 = MAX(src.y1, dst.y1);
    int x2 = MIN(src.x2, dst.x2);
    int y2 = MIN(src.y2, dst.y2);

    float area_src = (src.x2 - src.x1) * (src.y2 - src.y1);
    float area_dst = (dst.x2 - dst.x1) * (dst.y2 - dst.y1);
    float area_intersect = (x2 - x1) * (y2 - y1);
    float area_union = area_src + area_dst - area_intersect;
    float iou = (float)area_intersect / (float)area_union;

    return iou;
}


Mat GetMatchFlag(Mat &iou_mat, float iou_thresh) {
    Point max_loc;
    double max_val = .0;
    Mat flag_mat = Mat::zeros(Size(iou_mat.cols, iou_mat.rows), CV_32SC1);
    Mat iou_copy = iou_mat.clone();

    while (true) {
        minMaxLoc(iou_copy, NULL, &max_val, NULL, &max_loc);
        if (max_val < iou_thresh)
            break;
        flag_mat.at<int>(max_loc.y, max_loc.x) = 1;
        iou_copy.row(max_loc.y).setTo(0);
        iou_copy.col(max_loc.x).setTo(0);
    }

    return flag_mat;
}


#if 0
// version two
void GetMatchFlag(obj_infos &mark_vec, obj_infos &track_vec, float iou_thresh) {
    int i = 0, j = 0;
    Mat iou_mat = Mat::zeros(Size(track_vec.size(), mark_vec.size()), CV_32FC1);
    Mat flag_mat = Mat::zeros(Size(track_vec.size(), mark_vec.size()), CV_32SC1);
    for (i = 0; i < mark_vec.size(); ++i) {
        for (j = 0; j < track_vec.size(); ++j) {
            iou_mat.at<float>(i, j) = CalcIou(mark_vec[i].rect, track_vec[j].rect);
        }
    }

    for (i = 0; i < mark_vec.size(); ++i) {
        int max_idx = -1;
        float max_iou = iou_thresh;
        match_iter itrfind = kMatchPre.find(mark_vec[i].id);
        for (j = 0; j < track_vec.size(); ++j) {
            float iou_now = iou_mat.at<float>(i, j);
            if (iou_now > iou_thresh) {
                if (itrfind != kMatchPre.end() && track_vec[j].id == itrfind->second) {
                    flag_mat.at<int>(i, j) = 1;
                    iou_mat.row(i).setTo(0);
                    iou_mat.col(j).setTo(0);
                    break;
                }
                else if (iou_now > max_iou) {
                    max_iou = iou_now;
                    max_idx = j;
                }
            }
        }

        if (max_idx != -1) {
            flag_mat.at<int>(i, max_idx) = 1;
            iou_mat.row(i).setTo(0);
            iou_mat.col(max_idx).setTo(0);
        }
    }

    return;
}
#endif


Mat GetIOUMat(obj_infos &mark_vec, obj_infos &track_vec, float iou_thresh) {
    Mat iou_mat = Mat::zeros(Size(track_vec.size(), mark_vec.size()), CV_32FC1);
    for (int i = 0; i < (int)mark_vec.size(); ++i) {
        match_iter itrfind = kMatchPre.find(mark_vec[i].id);
        for (int j = 0; j < (int)track_vec.size(); ++j) {
            iou_mat.at<float>(i, j) = CalcIou(mark_vec[i].rect, track_vec[j].rect);
#if 1
            if (iou_mat.at<float>(i, j) > iou_thresh) {
                if (itrfind != kMatchPre.end() && track_vec[j].id == itrfind->second.match_id) {
                    // increase the weights of previous matched
                    iou_mat.at<float>(i, j) += 2;
                }
            }
#endif
        }
    }

    return iou_mat;
}


void EvalMOT(char *markname, char *trkname, float iou_thresh, int delay = -1) {
    obj_infos_vec mark_data, track_data;
    if (!ReadData(markname, mark_data))
        return;
    if (!ReadData(trkname, track_data))
        return;

    MotResult rst;
    memset(&rst, 0, sizeof(MotResult));
    vector<int> unq_id;

    // for drawing matched in video
    char vdoname[512] = { 0 };
    strncpy(vdoname, markname, strlen(markname) - 5);
#ifdef SAVE_IMG
    if (_access(vdoname, 0) != 0)
        _mkdir(vdoname);
#endif
    strcat(vdoname, ".mp4");

    // for drawing curve
    vector<string> str_vec;
    str_vec.push_back("MOTA");
    str_vec.push_back("MOTP");

    int frm_mark = 0, frm_trk = 0;
    while (frm_mark < (int)mark_data.size() || frm_trk < (int)track_data.size()) {
        obj_infos mark_vec, track_vec;

        // synchronize frame
        int frm = 0, mark_now = INT_MAX, trk_now = INT_MAX;
        if (frm_mark < (int)mark_data.size()) {
            mark_vec = mark_data[frm_mark];
            mark_now = mark_vec[0].frm;
        }

        if (frm_trk < (int)track_data.size()) {
            track_vec = track_data[frm_trk];
            trk_now = track_vec[0].frm;
        }

        frm = MIN(mark_now, trk_now);
        if (mark_now > trk_now) {
            mark_vec.clear();
            ++frm_trk;
        }
        else if (mark_now < trk_now) {
            track_vec.clear();
            ++frm_mark;
        }
        else {
            ++frm_mark;
            ++frm_trk;
        }

        rst.gt += (int)mark_vec.size();

        match_map match_cur;
        match_cur.clear();

        // judge match
        if (mark_vec.size() > 0 && track_vec.size() > 0) {
            int i = 0, j = 0;
            Mat iou_mat = GetIOUMat(mark_vec, track_vec, iou_thresh);
            Mat flag_mat = GetMatchFlag(iou_mat, iou_thresh);
            
            //for (i = 0; i < flag_mat.rows; ++i) {
            //    for (j = 0; j < flag_mat.cols; ++j)
            //        cout << flag_mat.at<int>(i, j) << endl;
            //}

            // search by row to find false negatives
            for (i = 0; i < flag_mat.rows; ++i) {
                for (j = 0; j < flag_mat.cols; ++j) {
                    if (1 == flag_mat.at<int>(i, j)) {
                        ++rst.tp;
                        float dst_tmp = iou_mat.at<float>(i, j);
                        dst_tmp = dst_tmp > 2.0 ? float(dst_tmp - 2.0) : dst_tmp;
                        dst_tmp = (float)(1.0 - dst_tmp);
                        rst.d += dst_tmp;
                        mark_vec[i].match_id = track_vec[j].id;

                        MatchInfo tmp_match;
                        tmp_match.match_id = track_vec[j].id;
                        tmp_match.match_frm = frm;
                        match_cur.insert(make_pair(mark_vec[i].id, tmp_match));
                        break;
                    }
                }

                if (-1 == mark_vec[i].match_id)
                    ++rst.fn;
            }

            // search by column to find false positives
            for (j = 0; j < flag_mat.cols; ++j) {
                for (i = 0; i < flag_mat.rows; ++i) {
                    if (1 == flag_mat.at<int>(i, j)) {
                        track_vec[j].match_id = mark_vec[i].id;
                        break;
                    }
                }

                if (-1 == track_vec[j].match_id)
                    ++rst.fp;
            }
           
        }
        else if (mark_vec.size() > 0) {
            rst.fn += (int)mark_vec.size();
        }
        else {
            rst.fp += (int)track_vec.size();
        }

        // judge mismatch
        if (match_cur.size() > 0 && kMatchPre.size() > 0) {
            for (match_iter itrnow = match_cur.begin(); itrnow != match_cur.end(); ++itrnow) {
                match_iter itrpre = kMatchPre.find(itrnow->first);
                if (itrpre != kMatchPre.end()) {
                    if (itrpre->second.match_id != itrnow->second.match_id) {
                        obj_iter itrtmp = find_if(track_vec.begin(), track_vec.end(),
							findx(itrnow->second.match_id));
                        itrtmp->pre_match_id = itrpre->second.match_id;
                        itrtmp->pre_match_frm = itrpre->second.match_frm;

                        ++rst.mmc;
                        itrpre->second = itrnow->second;
                    }
                }
                else {
                    kMatchPre.insert(*itrnow);
                }
            }
        }
        else if (match_cur.size() > 0) {
            for (match_iter itrnow = match_cur.begin(); itrnow != match_cur.end(); ++itrnow)
                kMatchPre.insert(*itrnow);
        }

        cout << "debug" << endl;
        for (obj_iter itrobj = mark_vec.begin(); itrobj != mark_vec.end(); ++itrobj) {
            if (unq_id.end() == find(unq_id.begin(), unq_id.end(), itrobj->id))
                unq_id.push_back(itrobj->id);
        }

        rst.gt_unq = (int)unq_id.size();
        /*rst.MOTA = (float)(1.0 - (float)(rst.fp + rst.fn) / (float)rst.gt
                   - (float)rst.mmc / (float)rst.gt_unq);*/
        rst.MOTA = (float)(1.0 - (float)(rst.fp + rst.fn + (float)rst.mmc) / (float)rst.gt);
        rst.MOTP = rst.d / rst.tp;
        cout << "frame " << frm << endl;
        cout << "gt:" << rst.gt << endl;
        cout << "tp:" << rst.tp << endl;
        cout << "fp:" << rst.fp << endl;
        cout << "fn:" << rst.fn << endl;
        cout << "mmc:" << rst.mmc << endl;
        cout << "MOTA:" << rst.MOTA << endl;
        cout << "MOTP:" << rst.MOTP << endl;
        cout << "-----------------------" << endl;

        if (delay >= 0) {
            ShowMatch(vdoname, mark_vec, track_vec, frm, delay);

            vector<float> vecFL;
            vecFL.push_back(rst.MOTA);
            vecFL.push_back(rst.MOTP);
            DrawCurve(vecFL, str_vec);
        }
    }

    char rstname[512] = {0};
    char *p = strrchr(markname, '\\');
    strncpy(rstname, markname, p - markname);
    strcat(rstname, "\\mot.csv");

    FILE *fprst = fopen(rstname, "a");
    if (NULL == fprst) {
        cout << "file write error!" << endl;
        return;
    }
    //fprintf(fprst, "gt,gt_unq,tp,fp,fn,mmc,MOTA,MOTP\n");
    fprintf(fprst, "video,face_num,person_num,tp,fp,fn,mmc,MOTA,MOTP\n");
    fprintf(fprst, "%s,%d,%d,%d,%d,%d,%d,%f,%f\n\n",
        vdoname, rst.gt, rst.gt_unq, rst.tp, rst.fp, rst.fn, rst.mmc, rst.MOTA, rst.MOTP);
    fclose(fprst);

    cout << endl << "Done! Result saved in:" << endl << rstname << endl << endl;

    waitKey(3000);
    destroyAllWindows();

    return;
}


int main(int argc, char *argv[]) {
    cout << "command line mode: **.exe [mark_file/vdo_file] [vdo_show_delay]" << endl;
    cout << "file name format: example_name.mark, example_name.track" << endl;
    char markname[256] = {0};
    char trkname[256] = {0};
    int delay = -1;

    if (1 < argc) {
        char *p = strrchr(argv[1], '.');
        if (p) {
            strncpy(markname, argv[1], p - argv[1]);
            strcpy(trkname, markname);
            strcat(markname, ".mark");
            strcat(trkname, ".track");
        }
		else {
			cout << "wrong file name format" << endl;
		}

		if (3 == argc) {
			delay = atoi(argv[2]);
		}
    }
    else {
        //strcpy(markname, "D:/Video/test.mp4");
        cout << "now manual input mode" << endl;
        cout << "step 1: input video file name:" << endl;
        cin >> markname;
        cout << "step 2: input track file:" << endl;;
        cin >> trkname;
        cout << "step 3: input video show delay:" << endl;
        cin >> delay;
    }

    char *ptr = strrchr(markname, '/');
    ptr = ptr ? ptr : strrchr(markname, '\\');
    if (ptr) {
        char buffer[512] = {0};
        //_getcwd(buffer, 512);
        strncpy(buffer, markname, ptr - markname);
        _chdir(buffer);
    }

    kMatchPre.clear();
    EvalMOT(markname, trkname, .4, delay);

    //system("pause");

    return 0;
}

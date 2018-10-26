/***************************************************************
## Evaluating Multiple Object Tracking Performance
--- Implementation of The CLEAR-MOT metric
## graphic demo
***************************************************************/

#include "mot.h"

using namespace std;
using namespace cv;

bool ReadData(char *fname, obj_infos_vec &data_vec) {
    ObjInfo obj_tmp;
    obj_infos frm_objs;

    char line_tmp[1024] = { 0 };
    FILE *fp = fopen(fname, "r");
    if (NULL == fp) {
        cout << "file open error!" << endl;
        return false;
    }

    int j = 0, num = 0;
    char *p = NULL;
    while (NULL != fgets(line_tmp, 1024, fp)) {
        if (strlen(line_tmp) < 4)
            continue;

        memset(&obj_tmp, 0, sizeof(ObjInfo));
        obj_tmp.match_id = obj_tmp.pre_match_id = obj_tmp.pre_match_frm = -1;
        frm_objs.clear();

        p = line_tmp;
        sscanf(p, "%d %d %*s", &obj_tmp.frm, &num);

        for (j = 0; j < num; j++) {
            while (*p != '(')
                ++p;

            ++p;
            sscanf(p, "%d %d %d %d %d %*s",
                &obj_tmp.id,
                &obj_tmp.rect.x1, &obj_tmp.rect.y1,
                &obj_tmp.rect.x2, &obj_tmp.rect.y2);
            frm_objs.push_back(obj_tmp);
        }

        data_vec.push_back(frm_objs);
    }
    fclose(fp);

    return true;
}


void DrawCurve(vector<float> &fvars, vector<string> &notes) {
    static int cnt = 0;
    static int var_num = fvars.size();
    static vector<vector<float>> fvars_vec;
    static Mat chart(300, 600, CV_8UC3, Scalar(220, 220, 220));
    static Mat chart_bak;

    int j = 0;
    static vector<Scalar> colors;
    if (0 == cnt) {
        char tmp[8] = { 0 };
        for (j = 0; j < 10; ++j) {
            line(chart,
                Point(0, chart.rows / 10 * (j + 1)),
                Point(chart.cols - 1, chart.rows / 10 * (j + 1)), Scalar(211, 211, 211), 2, CV_AA);
            line(chart,
                Point(chart.cols / 10 * (j + 1), 0),
                Point(chart.cols / 10 * (j + 1), chart.rows - 1), Scalar(211, 211, 211), 2, CV_AA);

            sprintf(tmp, "0.%d", 9 - j);
            putText(chart, tmp, Point(10, chart.rows / 10 * (j + 1)),
                CV_FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 0));
        }

        RNG rng(time(NULL));
        for (j = 0; j < var_num; ++j) {
            colors.push_back(Scalar(rng.uniform(0, 222), rng.uniform(0, 222), rng.uniform(0, 222)));
            line(chart,
                Point(0.75*chart.cols, 0.1*chart.rows*(j + 1)),
                Point(0.80*chart.cols, 0.1*chart.rows*(j + 1)), colors[j], 2, CV_AA);
            Size textsize = getTextSize("Test", FONT_HERSHEY_COMPLEX, 0.5, 1, 0);
            putText(chart, notes[j], Point(0.80*chart.cols + 20, 0.1*chart.rows*(j + 1) + textsize.height / 2),
                CV_FONT_HERSHEY_SIMPLEX, 0.5, colors[j]);
        }
        chart_bak = chart.clone();
    }

    int xstep = 4;
    int xnum = chart.cols / xstep;

    if (cnt < xnum) {
        fvars_vec.push_back(fvars);
        if (cnt > 0) {
            for (j = 0; j < fvars.size(); ++j) {
                line(chart,
                    Point(xstep * (cnt - 1), chart.rows * (1 - fvars_vec[cnt - 1][j]) - 1),
                    Point(xstep * cnt, chart.rows * (1 - fvars_vec[cnt][j]) - 1), colors[j], 2, CV_AA);
            }

        }
    }
    else {
        int sta = cnt % xnum;
        fvars_vec[sta] = fvars;
        sta = (sta + 1) % xnum;
        chart_bak.copyTo(chart);
        for (int k = 0; k < xnum - 1; ++k) {
            for (j = 0; j < fvars.size(); ++j) {
                line(chart,
                    Point(xstep * k, chart.rows * (1 - fvars_vec[(k + sta) % xnum][j])),
                    Point(xstep * (k + 1), chart.rows * (1 - fvars_vec[(k + sta + 1) % xnum][j])),
                    colors[j], 2, CV_AA);
            }
        }
    }

    ++cnt;

    imshow("variable curve", chart);

    return;
}


void CheckRect(Mat &src, Rect &rect) {
    rect.x = MAX(MIN(rect.x, src.cols - 1), 0);
    rect.y = MAX(MIN(rect.y, src.rows - 1), 0);
    rect.width = MAX(MIN(rect.width, src.cols - 1 - rect.x), 1);
    rect.height = MAX(MIN(rect.height, src.rows - 1 - rect.y), 1);
}


void ShowMatch(char *vdoname, obj_infos &mark_vec, obj_infos &track_vec, int frm, int delay/* = 1*/) {
    static VideoCapture sequence(vdoname);
    static Mat frame;
    static FrameQueue q;
    static int last_frm = -1;
    if (last_frm == 0) {
        if (!sequence.isOpened()) {
            cerr << "Failed to open the image sequence!\n" << endl;
            return;
        }
        //memset(&q, 0, sizeof(FrameQueue));
        q.front = q.rear = 0;
    }

    int delta_frm = frm - last_frm;
    for (int i = 0; i < delta_frm; ++i)
        sequence >> frame;
    last_frm = frm;


    RNG rng(time(NULL));
    Scalar color;
    double fontscale = 1.;
    char tmp[STR_LEN] = { 0 };
    char dir_name[STR_LEN] = { 0 };
    strncpy(dir_name, vdoname, strlen(vdoname) - 4);

    memset(tmp, 0, STR_LEN);
    sprintf(tmp, "frame: %d", frm);
    color = Scalar(255, 0, 255);
    putText(frame, tmp, Point(50, 50), CV_FONT_HERSHEY_DUPLEX, 1., color, 2);

    for (obj_iter mark_itr = mark_vec.begin(); mark_itr != mark_vec.end(); ++mark_itr) {
        ObjRect *prect = &(mark_itr->rect);
        Rect rect(Point(prect->x1, prect->y1), Point(prect->x2, prect->y2));
        CheckRect(frame, rect);
        color = Scalar(0, 0, 255);
        rectangle(frame, rect, color, 2);

        if (mark_itr->match_id != -1) {
            memset(tmp, 0, STR_LEN);
            _itoa_s(mark_itr->id, tmp, 10);
            color = Scalar(rng.uniform(0, 222), 255, rng.uniform(0, 222));
            putText(frame, tmp, Point(prect->x1, prect->y1), CV_FONT_HERSHEY_DUPLEX, fontscale, color, 2);

            obj_iter trk_itr = find_if(track_vec.begin(), track_vec.end(), findx(mark_itr->match_id));
            prect = &(trk_itr->rect);
            memset(tmp, 0, STR_LEN);
            _itoa_s(mark_itr->match_id, tmp, 10);
            putText(frame, tmp, Point(prect->x2, prect->y2), CV_FONT_HERSHEY_DUPLEX, fontscale, color, 2);
        }
        else {
            sprintf(tmp, "%d fn", mark_itr->id);
            putText(frame, tmp, Point(prect->x1, prect->y1), CV_FONT_HERSHEY_DUPLEX, fontscale, color, 2);

#ifdef SAVE_IMG
            memset(tmp, 0, STR_LEN);
            sprintf(tmp, "%s\\fn_%d_%d.jpg", dir_name, frm, mark_itr->id);
            Mat fpimg = frame(rect);
            imwrite(tmp, fpimg);
#endif
        }
    }


    bool hasmmc = false;
    for (obj_iter trk_itr = track_vec.begin(); trk_itr != track_vec.end(); ++trk_itr) {
        ObjRect *prect = &(trk_itr->rect);
        Rect rect(Point(prect->x1, prect->y1), Point(prect->x2, prect->y2));
        CheckRect(frame, rect);
        color = Scalar(50, 220, 220);
        rectangle(frame, rect, color, 2);

        if (-1 == trk_itr->match_id) {
            memset(tmp, 0, STR_LEN);
            sprintf(tmp, "%d fp", trk_itr->id);
            putText(frame, tmp, Point(prect->x2, prect->y2), CV_FONT_HERSHEY_DUPLEX, fontscale, color, 2);
#ifdef SAVE_IMG
            memset(tmp, 0, STR_LEN);
            sprintf(tmp, "%s\\fp_%d_%d.jpg", dir_name, frm, trk_itr->id);
            Mat fpimg = frame(rect);
            imwrite(tmp, fpimg);
#endif
        }
        else if (trk_itr->pre_match_id > 0) {
            hasmmc = true;
            memset(tmp, 0, STR_LEN);
            sprintf(tmp, "mmc[pre_match:%d fr:%d]", trk_itr->pre_match_id, trk_itr->pre_match_frm);
            putText(frame, tmp, Point(prect->x2, prect->y2 + 10), CV_FONT_HERSHEY_DUPLEX, fontscale, color, 2);
        }
    }

#ifdef SAVE_IMG
    // enqueue
    if ((q.rear + 1) % Q_SIZE == q.front) {
        // dequeue if queue is full
        q.front = (q.front + 1) % Q_SIZE;
    }
    q.frames[q.rear].frame = frame.clone();
    q.frames[q.rear].idx = frm;
    q.rear = (q.rear + 1) % Q_SIZE;

    if (hasmmc) {
        memset(tmp, 0, STR_LEN);
        sprintf(tmp, "%s\\%d_mmc.jpg", dir_name, frm);
        imwrite(tmp, frame);
        // dequeue last Q_SIZE - 1 frames and save
        int k = 0;
        while (q.front != q.rear) {
            if (q.frames[q.front].idx != frm) {
                memset(tmp, 0, STR_LEN);
                sprintf(tmp, "%s\\%d_b%d_fr%d.jpg", dir_name, frm, ++k, q.frames[q.front].idx);
                imwrite(tmp, q.frames[q.front].frame);
            }
            
            q.front = (q.front + 1) % Q_SIZE;
        }
    }

#if 0
    memset(tmp, 0, STR_LEN);
    sprintf(tmp, "%s\\FR_%d.jpg", dir_name, frm);
    Mat simg;
    resize(frame, simg, Size(frame.cols >> 1, frame.rows >> 1));
    imwrite(tmp, simg);
    cvWaitKey(10);
#endif
#endif

    imshow("matched", frame);
    int key = waitKey(delay);
    if (32 == key)
        waitKey();

    return;
}

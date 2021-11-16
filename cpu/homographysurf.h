#ifndef HOMOGRAPHYSURF_H_
#define HOMOGRAPHYSURF_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h> 
#include <fstream>
#include "const.h"

using namespace std;
using namespace cv;
// using namespace cv::cuda;

class HomographySurf{
    private:
        Mat im1;
        Mat im2;
        Mat im1Reg;
        Mat pano;
        Mat h;
        int nkp1 = 0;
        int nkp2 = 0;
        int inliers_good;
        Mat im1Gray, im2Gray;
        bool move = false;
    public:
        HomographySurf(Mat img1, Mat img2);
        bool warpImage(bool GPU);
        void setReference(Mat img2);
        void getHomographyGPU(int he);
        void getHomographyCPU(int he);
        int getNumberKp1();
        int getNumberKp2();
        Mat getRegistration();
        Mat getMatrix();
        int getNumberMatches();
        bool getMove();                                         // chuyen goc 
        void setMove(bool nmove);
        int getShift();                                         // xuat ra do lech pixel chuyen goc
        int createMoving(string outputpath, string name);       // tao anh chuyen goc
        void createMovingInter(string outputpath, string name); // noi suy ra anh chuyen goc
};

#endif
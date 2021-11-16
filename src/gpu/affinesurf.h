#ifndef AFFINESURF_H_
#define AFFINESURF_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/calib3d.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

class AffineSurf{
    private:
        Mat im1;
        Mat im2;
        Mat im1Reg;
        Mat h;
        int inliers_good;
        Mat im1Gray, im2Gray;
    public:
        AffineSurf(Mat img1, Mat img2);
        bool warpImage();
        void getAffine(int he);
        Mat getRegistration();
        Mat getMatrix();
        int getNumberMatches();
};

#endif
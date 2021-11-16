#include <iostream>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core/affine.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
// using namespace cv::xfeatures2d;
// using namespace cv::cuda;

// int main1(int argc, char *argv[])
// {   
    
//     cuda::setDevice(0);
    
//     int64 t0 = cv::getTickCount();
//     cout << cuda::getCudaEnabledDeviceCount() << endl;
//     Mat im1 = imread("/media/master/hs00/files/1gWO66eUFKs_4GFtwK-nZX-eJPjFSRIxN/o/16097.jpg");
//     Mat im2 = imread("/media/master/hs00/files/1gWO66eUFKs_4GFtwK-nZX-eJPjFSRIxN/f1/"+string(argv[1])+".jpg");
//     Mat im1Gray, im2Gray;

//     cv::cvtColor(im1, im1Gray, cv::COLOR_BGR2GRAY);
//     cv::cvtColor(im2, im2Gray, cv::COLOR_BGR2GRAY);
//     GpuMat img1Gpu(im1Gray);
//     GpuMat img2Gpu(im2Gray);
//     // GpuMat img1;
//     // GpuMat img2;

//     // cv::cuda::cvtColor(img1Gpu, img1, CV_BGR2GRAY);
//     // cv::cuda::cvtColor(img2Gpu, img2, CV_BGR2GRAY);

//     SURF_CUDA surf(400);
//     GpuMat keypoints1GPU, keypoints2GPU;
//     GpuMat descriptors1GPU, descriptors2GPU;

//     surf(img1Gpu, GpuMat(), keypoints1GPU, descriptors1GPU);
//     surf(img2Gpu, GpuMat(), keypoints2GPU, descriptors2GPU);

//     cout << "FOUND " << keypoints1GPU.cols << " keypoints on first image" << endl;
//     cout << "FOUND " << keypoints2GPU.cols << " keypoints on second image" << endl;

//     // matching descriptors
//     Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
//     // vector<vector<DMatch>> knn_matchesGPU;
//     GpuMat knn_matchesGPU;
//     matcher->knnMatchAsync(descriptors1GPU, descriptors2GPU, knn_matchesGPU, 2);
    
    
//     vector<vector<DMatch>> knn_matches;
//     matcher->knnMatchConvert(knn_matchesGPU, knn_matches); //download matches from gpu to cpu

    

//     // downloading results
//     vector<KeyPoint> keypoints1, keypoints2;
//     vector<float> descriptors1, descriptors2;
//     surf.downloadKeypoints(keypoints1GPU, keypoints1);
//     surf.downloadKeypoints(keypoints2GPU, keypoints2);
//     surf.downloadDescriptors(descriptors1GPU, descriptors1);
//     surf.downloadDescriptors(descriptors2GPU, descriptors2);
  

//     std::vector<cv::DMatch> matches;      
//     // matcher->match(descriptors1GPU, descriptors2GPU, matches);                   //vector of good matches between tested images
//     // Filter the matches using the ratio test
//     for(std::vector<std::vector<cv::DMatch> >::const_iterator it = knn_matches.begin(); it != knn_matches.end(); ++it) {
//         if(it->size() > 1 && (*it)[0].distance/(*it)[1].distance < 0.7) {
//             DMatch m =(*it)[0];
//             matches.push_back(m);
//         }
//     }

//     std::vector<Point2f> points1, points2;

//     for (size_t i=0; i<matches.size(); i++){
//         points1.push_back( keypoints1[matches[i].queryIdx].pt);
//         points2.push_back( keypoints2[matches[i].trainIdx].pt);
//     }
//     cout<< matches.size() << endl;
//     vector<uchar> inliers;
//     // cv::Mat h = cv::estimateAffine2D(points1, points2, inliers);
//     cv::Mat h = cv::findHomography(points1, points2, RANSAC, 1.5, inliers);

//     int inliers_good = count(inliers.begin(), inliers.end(), 1);

//     cout << inliers_good <<endl;

//     Mat im1Reg;
//     warpPerspective(im1, im1Reg, h, im2.size());
//     int height = im2.rows;
//     int width = im2.cols;
//     std::vector<Point2f> camera_corners,world_corners;
//     camera_corners.push_back(Point2f(0, 0));
//     camera_corners.push_back(Point2f(0, height - 1));
//     camera_corners.push_back(Point2f(width - 1, height -1));
//     camera_corners.push_back(Point2f(width - 1, 0));

//     perspectiveTransform(camera_corners, world_corners, h);
//     if(abs(world_corners[0].x - world_corners[2].x) < width*0.8 
//          || abs(world_corners[0].y - world_corners[2].y) < height*0.8
//          || abs(world_corners[0].x - world_corners[2].x > width*1.5)
//          || abs(world_corners[0].y - world_corners[2].y > height*1.5)) {
//        cout << "false";
//     }   
//     else cout << "true";
    
//     cout << world_corners << endl;

//     // warpAffine(im1, im1Reg, h, im2.size());
//     // drawing the results
//     Mat img_matches;
//     drawMatches(Mat(im1), keypoints1, Mat(im2), keypoints2, matches, img_matches);
//     imwrite("matches.jpg", img_matches);
//     imwrite("output.jpg", im1Reg);

//     // // namedWindow("matches", 0);
//     // // imshow("matches", img_matches);
//     // // waitKey(0);
//     cout << "-----------------"<<endl;
//     Mat h1 = h.reshape(1, 9);
//     cout << h1 << endl;
//     int64 t1 = cv::getTickCount();
//     double secs = (t1-t0)/cv::getTickFrequency();

//     cout << "computational time : " << secs << endl;
//     // cuda::resetDevice();

//     return 0;
// }

int main(int argc, char *argv[])
{   
    int64 t0 = cv::getTickCount();
    Mat im1 = imread("/mnt/hs01/test/aaa/o/09854.jpg");
    Mat im2 = imread("/mnt/hs01/test/aaa/o/09853.jpg");
    Mat im1Gray, im2Gray;

    cv::cvtColor(im1, im1Gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(im2, im2Gray, cv::COLOR_BGR2GRAY);  


    Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( 400 );
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute( im1Gray, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( im2Gray, noArray(), keypoints2, descriptors2 );

    cout << "FOUND " << keypoints1.size() << " keypoints on first image" << endl;
    cout << "FOUND " << keypoints2.size() << " keypoints on second image" << endl;
    // matching descriptors
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    // vector<vector<DMatch>> knn_matchesGPU;
    vector<vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    std::vector<cv::DMatch> matches;      
    //vector of good matches between tested images
    // Filter the matches using the ratio test
    for(std::vector<std::vector<cv::DMatch> >::const_iterator it = knn_matches.begin(); it != knn_matches.end(); ++it) {
        if(it->size() > 1 && (*it)[0].distance/(*it)[1].distance < 0.7) {
            DMatch m =(*it)[0];
            matches.push_back(m);
        }
    }

    std::vector<Point2f> points1, points2;

    for (size_t i=0; i<matches.size(); i++){
        points1.push_back( keypoints1[matches[i].queryIdx].pt);
        points2.push_back( keypoints2[matches[i].trainIdx].pt);
    }

    cout<< matches.size() << endl;
    vector<uchar> inliers;

    cv::Mat h = cv::findHomography(points1, points2, RANSAC, 1.5, inliers);

    int inliers_good = count(inliers.begin(), inliers.end(), 1);

    cout << inliers_good <<endl;

    Mat im1Reg;
    warpPerspective(im1, im1Reg, h, im2.size());

    cv::imwrite("out.jpg", im1Reg);

    int64 t1 = cv::getTickCount();
    double secs = (t1-t0)/cv::getTickFrequency();

    cout << "computational time : " << secs << endl;
}
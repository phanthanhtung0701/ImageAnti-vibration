#include "affinesurf.h"

AffineSurf::AffineSurf(Mat img1, Mat img2){
    im1 = img1;
    im2 = img2;
    cv::cvtColor(im1, im1Gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(im2, im2Gray, cv::COLOR_BGR2GRAY);
    inliers_good = 0;
}

Mat AffineSurf::getRegistration(){
    return im1Reg;
}

Mat AffineSurf::getMatrix(){
    return h;
}

int AffineSurf::getNumberMatches(){
    return inliers_good;
}

void AffineSurf::getAffine(int he){
    SURF_CUDA surf;
    GpuMat keypoints1GPU, keypoints2GPU;
    GpuMat descriptors1GPU, descriptors2GPU;
    
    GpuMat img1Gpu(im1Gray);
    GpuMat img2Gpu(im2Gray);

    surf(img1Gpu, GpuMat(), keypoints1GPU, descriptors1GPU);
    surf(img2Gpu, GpuMat(), keypoints2GPU, descriptors2GPU);

    if (keypoints1GPU.cols > 0 && keypoints2GPU.cols > 0){
        // matching descriptors
        Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());

        GpuMat knn_matchesGPU;
        matcher->knnMatchAsync(descriptors1GPU, descriptors2GPU, knn_matchesGPU, 2);
        
        vector<vector<DMatch>> knn_matches;
        matcher->knnMatchConvert(knn_matchesGPU, knn_matches); //download matches from gpu to cpu
        // downloading results
        vector<KeyPoint> keypoints1, keypoints2;
        vector<float> descriptors1, descriptors2;
        surf.downloadKeypoints(keypoints1GPU, keypoints1);
        surf.downloadKeypoints(keypoints2GPU, keypoints2);
        surf.downloadDescriptors(descriptors1GPU, descriptors1);
        surf.downloadDescriptors(descriptors2GPU, descriptors2);

        std::vector<cv::DMatch> matches;                         //vector of good matches between tested images
        //Filter the matches using the ratio test
        for(std::vector<std::vector<cv::DMatch> >::const_iterator it = knn_matches.begin(); it != knn_matches.end(); ++it) {
            if(it->size() > 1 && (*it)[0].distance/(*it)[1].distance < 0.7) {
                DMatch m =(*it)[0];
                matches.push_back(m);
            }
        }
        if (matches.size() > 0){
            std::vector<Point2f> points1, points2;

            for (size_t i=0; i<matches.size(); i++){
                points1.push_back( keypoints1[matches[i].queryIdx].pt);
                points2.push_back( keypoints2[matches[i].trainIdx].pt);
            }

            vector<uchar> inliers;
            h = cv::estimateAffine2D(points1, points2, inliers);
            // h = cv::findHomography(points1, points2, RHO,3, inliers);
            inliers_good = count(inliers.begin(), inliers.end(), 1);
        } 
        else inliers_good = 0;
    }
    else inliers_good = 0;
}

bool AffineSurf::warpImage(){
    getAffine(500);
    int n = getNumberMatches();
    int height = im2.rows;
    int width = im2.cols;
    if (n >= 100){
        h = getMatrix();
        warpAffine(im1, im1Reg, h, im2.size());
        // warpPerspective(im1, im1Reg, h, im2.size());
        return true;
    }
    else return false;
}
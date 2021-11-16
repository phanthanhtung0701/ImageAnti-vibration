#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/calib3d.hpp"

#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"

#include <ctime>
#include <string>
#include <fstream>
#include <stdio.h>
#include <dirent.h>
#include <sys/types.h>
#include <nlohmann/json.hpp>

using namespace std;
using namespace cv::cuda;
using namespace cv;
using namespace cv::detail;
using json = nlohmann::json;

const int DEVIATION_ANGLE = 0.15;

bool checkMove(std::vector<Point2f> firstCorners, std::vector<Point2f> secondCorners, int height, int width){
    if (abs(firstCorners[0].x - secondCorners[0].x) > width*DEVIATION_ANGLE
        || abs(firstCorners[1].x - secondCorners[1].x) > width*DEVIATION_ANGLE
        || abs(firstCorners[2].x - secondCorners[2].x) > width*DEVIATION_ANGLE
        || abs(firstCorners[3].x - secondCorners[3].x) > width*DEVIATION_ANGLE
        || abs(firstCorners[0].y - secondCorners[0].y) > height*DEVIATION_ANGLE
        || abs(firstCorners[1].y - secondCorners[1].y) > height*DEVIATION_ANGLE
        || abs(firstCorners[2].y - secondCorners[2].y) > height*DEVIATION_ANGLE
        || abs(firstCorners[3].y - secondCorners[3].y) > height*DEVIATION_ANGLE)
    {
        cout << "move" << endl;
        return true;      
    }  
    else if (secondCorners[0].x - firstCorners[0].x < 100 && secondCorners[0].y - firstCorners[0].y < 100       // check room out
      && secondCorners[1].x - firstCorners[1].x < 100 && secondCorners[1].y - firstCorners[1].y > 100
      && secondCorners[2].x - firstCorners[2].x > 100 && secondCorners[2].y - firstCorners[2].y > 100
      && secondCorners[3].x - firstCorners[3].x > 100 && secondCorners[3].y - firstCorners[3].y < 100)
    {
        cout << "room out" << endl;
        return true;    
    }
    else if (secondCorners[0].x - firstCorners[0].x > 100 && secondCorners[0].y - firstCorners[0].y > 100      // check room in
      && secondCorners[1].x - firstCorners[1].x > 100 && secondCorners[1].y - firstCorners[1].y < 100
      && secondCorners[2].x - firstCorners[2].x < 100 && secondCorners[2].y - firstCorners[2].y < 100
      && secondCorners[3].x - firstCorners[3].x < 100 && secondCorners[3].y - firstCorners[3].y > 100)
      {     
          cout << "room in" << endl;
          return true;
      }
    else return false;
}

int getShift(Mat im2, Mat h){
    int height = im2.rows;
    int width = im2.cols;    
    
    std::vector<Point2f> cornersFirst,cornersWarped;

    cornersFirst.push_back(Point2f(0, 0));
    cornersFirst.push_back(Point2f(0, height - 1));
    cornersFirst.push_back(Point2f(width - 1, height -1));
    cornersFirst.push_back(Point2f(width - 1, 0));

    perspectiveTransform(cornersFirst, cornersWarped, h);
    cout << cornersWarped << endl;

    Rect boudRect;
    boudRect = boundingRect(cornersWarped);

    int bx, by, bwidth, bheight;
    bx =  boudRect.tl().x;
    by =  boudRect.tl().y;
    bheight = boudRect.height;
    bwidth = boudRect.width;

    // std::cout << x_offset << " x " << y_offset << std::endl;
    // cv::imwrite("/opt/code/process/ccode/testview/pano.jpg", pano);

    // create interpolated image
    float sum = 0;
    for (int i=0;i<4; i++){
        sum += abs(cornersWarped[i].x - cornersFirst[i].x)+abs(cornersWarped[i].y - cornersFirst[i].y);
    }
    sum /=8;
    return int(sum);
}

Mat findWarp(int height, int width, std::vector<Point2f> corners){
    Mat h;
    Mat M;

    std::vector<Point2f> preCorners;
    preCorners.push_back(Point2f(0, 0));
    preCorners.push_back(Point2f(0, height - 1));
    preCorners.push_back(Point2f(width - 1, height -1));
    preCorners.push_back(Point2f(width - 1, 0));

    M = cv::getPerspectiveTransform(preCorners, corners);

    Rect boudRect;
    boudRect = boundingRect(corners);

    int bx, by, bwidth, bheight;
    bx =  boudRect.tl().x;
    by =  boudRect.tl().y;
    bheight = boudRect.height;
    bwidth = boudRect.width;

    cv::Mat_<double> T = (cv::Mat_<double>(3, 3) << 1, 0, -bx, 0, 1, -by, 0, 0, 1);
    Mat h1 = T*M;
    h = h1.inv();
    return h;
}

template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{

	std::vector<double> linspaced;

	double start = static_cast<double>(start_in);
	double end = static_cast<double>(end_in);
	double num = static_cast<double>(num_in);

	if (num == 0) { return linspaced; }
	if (num == 1)
	{
		linspaced.push_back(start);
		return linspaced;
	}

	double delta = (end - start) / (num - 1);

	for (int i = 0; i < num - 1; ++i)
	{
		linspaced.push_back(start + delta * i);
	}
	linspaced.push_back(end); // I want to ensure that start and end
							  // are exactly the same as the input
	return linspaced;
}

int findnInterpolated(int pixel, int shift){
    return pixel*shift;
}

int main(int argc, char *argv[]){
    // cuda::setDevice(0);

    // Mat im1;
    // Mat im2;
    Mat im1Gray, im2Gray, h;
    string path1 = argv[1];
    string path2 = argv[2];
    //Mat im2 = imread("/mnt/hs01/test/aaa/f1_test/09860.jpg");
    Mat im2 = imread("/opt/test/"+path1+".jpg");
    //Mat im1 = imread("/mnt/hs01/test/aaa/f1_test/09900.jpg");
    Mat im1 = imread("/opt/test/"+path2+".jpg");
    cv::cvtColor(im1, im1Gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(im2, im2Gray, cv::COLOR_BGR2GRAY);

    SURF_CUDA surf(400);
    GpuMat keypoints1GPU, keypoints2GPU;
    GpuMat descriptors1GPU, descriptors2GPU;
    
    GpuMat img1Gpu(im1Gray);
    GpuMat img2Gpu(im2Gray);

    surf(img1Gpu, GpuMat(), keypoints1GPU, descriptors1GPU);
    surf(img2Gpu, GpuMat(), keypoints2GPU, descriptors2GPU);

    std::cout << "number keypoints 1: "<< keypoints1GPU.cols << std::endl;
    std::cout << "number keypoints 2: "<< keypoints2GPU.cols << std::endl;
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
        
        //draw matchs
        Mat img_matches;
        try
        {
            drawMatches( im1, keypoints1, im2, keypoints2, matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
            cv::imwrite("/opt/test/match.jpg", img_matches);    
        }
        catch(...)
        {
            cout<<"Error in drawMatches "<< endl;
        }

        int inliers_good = 0;
		cout << "match: "<< matches.size() << endl;
        if (matches.size() > 0){
            std::vector<Point2f> points1, points2;
            std::vector<Point2f> pts1, pts2;

            for (size_t i=0; i<matches.size(); i++){
                points1.push_back( keypoints1[matches[i].queryIdx].pt);
                points2.push_back( keypoints2[matches[i].trainIdx].pt);
            }

            vector<uchar> inliers;
            //h = cv::estimateAffine2D(points1, points2, inliers);
            h = cv::findHomography(points1, points2, RANSAC, 0.5, inliers);
            cout << "number match RANSAC 1: "<<count(inliers.begin(), inliers.end(), 1)<< endl;
            for (int i=0; i<inliers.size(); i++){
                if (inliers[i] == 1){
                    pts1.push_back(points1[i]);
                    pts2.push_back(points2[i]);
                }
            }

            // h = cv::findHomography(pts1, pts2, RANSAC, 1.5, inliers);
            // h = cv::findHomography(pts1, pts2, LMEDS, 0, inliers);
            // h = cv::findHomography(points2, points2, LMEDS, 3, inliers);
            inliers_good = count(inliers.begin(), inliers.end(), 1);
        }
        cout << "number match: "<<inliers_good<< endl;
        int height = im2.rows;
        int width = im2.cols;
        Mat pano;
        warpPerspective(im1, pano, h, im1.size());
        int shift = getShift(im2, h);
        cout << shift << endl;
        cv::imwrite("/opt/test/pano1.jpg", pano);
		// return 0;
        std::vector<Point2f> cornersFirst,cornersWarped;
        cornersFirst.push_back(Point2f(0, 0));
        cornersFirst.push_back(Point2f(0, height - 1));
        cornersFirst.push_back(Point2f(width - 1, height -1));
        cornersFirst.push_back(Point2f(width - 1, 0));

        perspectiveTransform(cornersFirst, cornersWarped, h);

        if(abs(cornersWarped[0].x - cornersWarped[2].x) < width*0.3
            || abs(cornersWarped[0].y - cornersWarped[2].y) < height*0.3
            || abs(cornersWarped[0].x - cornersWarped[2].x) > width*1.8
            || abs(cornersWarped[0].y - cornersWarped[2].y) > height*1.8) {
            cout << "Loi" << endl;
        }
        
        bool check = checkMove(cornersFirst, cornersWarped, height, width);
        cout << check << endl;
        Mat mask_pano;
		mask_pano.create(im1.size(), CV_8UC1);
		mask_pano.setTo(Scalar::all(255));
		warpPerspective(mask_pano, mask_pano, h, im1.size());
        int white = cv::countNonZero(mask_pano);
        int total_size = pano.rows*pano.cols;

        cout << "ratio: " << (double)white/(double)total_size << endl;

        Rect boudRect;
        boudRect = boundingRect(cornersWarped);

        

        int bx, by, bwidth, bheight;
        bx =  boudRect.tl().x;
        by =  boudRect.tl().y;
        bheight = boudRect.height;
        bwidth = boudRect.width;
        
        int x_offset, y_offset, n_height, n_width;
        if (bx < 0){
            x_offset = -bx;
            n_width = width + abs(bx);
            if (bwidth - n_width > 0) n_width = bwidth;
        }
        else {
            x_offset = 0;
            n_width = abs(bx) + bwidth;
            if (width - n_width > 0) n_width = width;
        }

        if (by < 0){
            y_offset = -by;
            n_height = height + abs(by);
            if (bheight - n_height > 0) n_height = bheight;
        }
        else {
            y_offset = 0;
            n_height = abs(by) + bheight;
            if (height-n_height > 0) n_height = height;
        }

        cv::Mat_<double> T = (cv::Mat_<double>(3, 3) << 1, 0, x_offset, 0, 1, y_offset, 0, 0, 1);
		
        Mat h1;
        h1 = T*h;
       
        Size s(n_width, n_height);
        // warpPerspective(im1, pano, h1, s, INTER_LINEAR, BORDER_REFLECT);
        warpPerspective(im1, pano, h1, s);
        cv::imwrite("/opt/test/pano.jpg", pano);
		return 0;
        // cv::imwrite("/mnt/hs01/test/aaa/test/pano.jpg", pano);
		
		std::cout << x_offset << " x " << y_offset << std::endl;
        cout << n_height << "x" << n_width <<endl;
		// blender
		Mat mask1, mask2;
		mask1.create(im1.size(), CV_8UC1);
		mask1.setTo(Scalar::all(255));
		//cout << "mask1: " << mask1.size() << endl;
		warpPerspective(mask1, mask1, h1, s);
		
		Mat im2_mask ;
		im2_mask.create(im2.size(), CV_8U);
		//cout << "im2_mask: " << im2_mask.size() << endl;
		im2_mask.setTo(Scalar::all(255));
		Mat image2Updated;
		warpPerspective(im2, image2Updated, T, s, INTER_LINEAR, BORDER_REFLECT); 
		
		Mat gray;
		cv::cvtColor(im2,gray, cv::COLOR_BGR2GRAY);
		cv::threshold(gray,mask2, 0, 255, cv::THRESH_BINARY);
		warpPerspective(mask2, mask2, T, s); 
		
        Mat image1Updated;
		pano.copyTo(image1Updated);
		
		//compositing
		cout << "compositing...." << endl;
		int num_images = 2;
		vector<Point> corners(num_images);
		vector<UMat> masks_warped(num_images);
		vector<UMat> masks_warped_f(num_images);
		vector<UMat> images_warped(num_images);
		vector<Mat> img(num_images);
		vector<UMat> images_warped_f(num_images);
		
		image1Updated.copyTo(img[0]);
		image2Updated.copyTo(img[1]);
		
		mask1.copyTo(masks_warped[0]);
		mask2.copyTo(masks_warped[1]);
		mask1.copyTo(masks_warped_f[0]);
		mask2.copyTo(masks_warped_f[1]);

		for (int i = 0; i < num_images; ++i)
		{
		    corners[i] = Point(0, 0);
		    img[i].copyTo(images_warped[i]);
			images_warped[i].convertTo(images_warped_f[i], CV_32F);
		}
		Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN);
		Mat mask_warped;
		vector<Mat> img_warped(num_images);
		compensator->feed(corners, images_warped, masks_warped);
		
		// seam finder
		cout << "seam finding...." << endl;
		Ptr<SeamFinder> seam_finder;
		seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
		seam_finder->find(images_warped_f, corners, masks_warped_f);
		
		cv::imwrite("/mnt/hs01/test/aaa/f1_test/09900/test/mask1.jpg", masks_warped_f[0]);
		cv::imwrite("/mnt/hs01/test/aaa/f1_test/09900/test/mask2.jpg", masks_warped_f[1]);
		
		masks_warped_f[0].copyTo(mask1);
		masks_warped_f[1].copyTo(mask2);
		masks_warped_f[0].release();
		masks_warped_f[1].release();
		
		images_warped.clear();
		images_warped_f.clear();
		
		for (int i = 0; i < num_images; i++) {
		    masks_warped[i].copyTo(mask_warped);
			img[i].copyTo(img_warped[i]);
		    compensator->apply(i, corners[i], img_warped[i], mask_warped);
		}
		
		cv::imwrite("/mnt/hs01/test/aaa/f1_test/09900/test/img1.jpg", img_warped[0]);
		cv::imwrite("/mnt/hs01/test/aaa/f1_test/09900/test/img2.jpg", img_warped[1]);
		
		cout << "blending...." << endl;
		//create blender
		cv::Ptr<cv::detail::Blender> blender = cv::detail::Blender::createDefault(cv::detail::Blender::MULTI_BAND, false);
		//prepare resulting size of image
		blender->prepare(Rect(0, 0, pano.size().width, pano.size().height));
		
		//feed images and the mask areas to blend
		blender->feed(img_warped[0], mask1, Point2f (0,0));
		cout << "blending1...." << endl;
		blender->feed(img_warped[1], mask2, Point2f (0,0));
		Mat result_s, result_mask;
		cout << "blending2...." << endl;
		//blend
		blender->blend(result_s, result_mask);
        //im2.copyTo(pano(cv::Rect(x_offset, y_offset, width, height)));
        //int rr = 0;
        //int cc = 0;
        //for (int r=y_offset; r < y_offset+height; r++){
        //    int cc = 0;
        //    for (int c = x_offset; c < x_offset+width; c++){
        //        if (pano.at<Vec3b>(r, c)[0] == 0 && pano.at<Vec3b>(r, c)[1] == 0 && pano.at<Vec3b>(r, c)[2] == 0){
        //            pano.at<Vec3b>(r,c) = im2.at<Vec3b>(rr,cc);
        //       }
        //        cc++;
        //    }
        //    rr++;
        //}
		cout << result_s.size() << endl;
        cv::imwrite("/mnt/hs01/test/aaa/f1_test/09900/test/pano.jpg", result_s);
    }
    
    cuda::resetDevice();
    return 0;
}

int main0(){
    json j;
    json o;
    o["name"] = "aaaaa";
    o["success"] = true;
    j["data"].push_back(o);
    cout<<j.dump()<<endl;
    return 1;
}
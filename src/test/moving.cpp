#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/calib3d.hpp"

#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"

#include <ctime>
#include <string>
#include <fstream>
#include <stdio.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h> 
#include <sys/types.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

using namespace std;
// using namespace cv::cuda;
using namespace cv;
using namespace cv::detail;

const double PIXEL = 0.1;

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

int findnInterpolated(double pixel, int shift){
    return int(pixel*shift);
}

int Move(string file, string view, bool GPU);

int main(int argc, char *argv[]){
    cuda::setDevice(0);

    if (argc <4 ) {
        cout<<"Lack of argument"<<endl;
    } else {
        bool GPU;
        string type = argv[1];
        if (type == "gpu") {
            GPU = true;
        }
        else if (type == "cpu") GPU = false;

        Move(argv[2], argv[3], GPU);
    }
    
    cuda::resetDevice();
    return 0;
}

Mat GetHomographyGPU(Mat im1Gray, Mat im2Gray, int he){
    Mat h;
    cv::cuda::SURF_CUDA surf(he);
    cv::cuda::GpuMat keypoints1GPU, keypoints2GPU;
    cv::cuda::GpuMat descriptors1GPU, descriptors2GPU;
    
    cv::cuda::GpuMat img1Gpu(im1Gray);
    cv::cuda::GpuMat img2Gpu(im2Gray);

    surf(img1Gpu, cv::cuda::GpuMat(), keypoints1GPU, descriptors1GPU);
    surf(img2Gpu, cv::cuda::GpuMat(), keypoints2GPU, descriptors2GPU);

    if (keypoints1GPU.cols > 0 && keypoints2GPU.cols > 0){
        // matching descriptors
        Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());

        cv::cuda::GpuMat knn_matchesGPU;
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
            //h = cv::estimateAffine2D(points1, points2, inliers);
            h = cv::findHomography(points1, points2, RANSAC, 1.5, inliers);

            return h;
        }
    }
}

Mat GetHomographyCPU(Mat im1Gray, Mat im2Gray, int he) {
    Mat h;

    Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(he);
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    surf->detectAndCompute( im1Gray, noArray(), keypoints1, descriptors1 );
    surf->detectAndCompute( im2Gray, noArray(), keypoints2, descriptors2 );

    if (keypoints1.size() > 0 && keypoints2.size() > 0){
        // matching descriptors
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);

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
        
        if (matches.size() > 0){
            std::vector<Point2f> points1, points2;

            for (size_t i=0; i<matches.size(); i++){
                points1.push_back( keypoints1[matches[i].queryIdx].pt);
                points2.push_back( keypoints2[matches[i].trainIdx].pt);
            }

            vector<uchar> inliers;

            h = cv::findHomography(points1, points2, RANSAC, 1.5, inliers);

            return h;
        }
    }
}


int Move(string file, string view, bool GPU){
    try {
        int64 start = cv::getTickCount();

        std::ifstream data_file(file);
        json jdata = json::parse(data_file);
        // string datapath = jdata["path"];
        string im1_path = jdata["image"];
        string im2_path = jdata["file"][jdata["file"].size()-1];
        string output = jdata["out"];

        // create json object (if view = json)
        json results;

        // make directory
        int check = mkdir((output).c_str(), 0777);

        Mat im1 = cv::imread(im1_path);
        Mat im2 = cv::imread(im2_path);
        
        Mat im1Gray, im2Gray, h;
        cv::cvtColor(im1, im1Gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(im2, im2Gray, cv::COLOR_BGR2GRAY);

        if (GPU) {
            h = GetHomographyGPU(im1Gray, im2Gray, 400);
        }
        else h = GetHomographyCPU(im1Gray, im2Gray, 400);

        int height = im2.rows;
        int width = im2.cols;
        std::vector<Point2f> cornersFirst,cornersWarped;
        cornersFirst.push_back(Point2f(0, 0));
        cornersFirst.push_back(Point2f(0, height - 1));
        cornersFirst.push_back(Point2f(width - 1, height -1));
        cornersFirst.push_back(Point2f(width - 1, 0));

        perspectiveTransform(cornersFirst, cornersWarped, h);

        Rect boudRect;
        boudRect = boundingRect(cornersWarped);

        int bx, by, bwidth, bheight;
        bx =  boudRect.tl().x;
        by =  boudRect.tl().y;
        bheight = boudRect.height;
        bwidth = boudRect.width;
        
        // calculate size of pano and x,y offset
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
        Mat pano;
        Size s(n_width, n_height);
        warpPerspective(im1, pano, h1, s, INTER_LINEAR, BORDER_REFLECT);
		
		// mask
		Mat mask1;
		mask1.create(im1.size(), CV_8U);
		mask1.setTo(Scalar::all(255));
		warpPerspective(mask1, mask1, h1, s);

        double process_time;
        double total_time;

        // create interpolated image
        int nInterpolated = jdata["file"].size();

        std::vector<double> xx[nInterpolated], yy[nInterpolated];
        for (int i=0; i<4; i++){
            xx[i] = linspace(cornersFirst[i].x, cornersWarped[i].x, nInterpolated);
            yy[i] = linspace(cornersFirst[i].y, cornersWarped[i].y, nInterpolated);
        }
        for (int k=0; k<nInterpolated; k++){
            try {
                int64 t_start = cv::getTickCount();

                string file_path = jdata["file"][k];
                int found = file_path.find_last_of("/");
                string name = file_path.substr(found+1);

                Mat imReference = cv::imread(jdata["file"][k]);
				
				//create panorama
				Mat image2Updated;
				warpPerspective(imReference, image2Updated, T, s, INTER_LINEAR, BORDER_REFLECT); 
				Mat image1Updated;
				pano.copyTo(image1Updated);
				
				Mat mask2_temp, mask1_temp;
				mask1.copyTo(mask1_temp);
				//mask2_temp.create(imReference.size(), CV_8U);
				//mask2_temp.setTo(Scalar::all(255));
				Mat gray;
				cv::cvtColor(imReference,gray, cv::COLOR_BGR2GRAY);
				cv::threshold(gray,mask2_temp, 0, 255, cv::THRESH_BINARY);
				warpPerspective(mask2_temp, mask2_temp, T, s); 
				
				
				int num_images = 2;
				vector<Point> corners(num_images);
				corners[0] = Point(0, 0);
				corners[1] = Point(0, 0);

				vector<UMat> images_warped(num_images);
				image1Updated.copyTo(images_warped[0]);
				image2Updated.copyTo(images_warped[1]);
				
				vector<UMat> masks_warped(num_images);
				mask1_temp.copyTo(masks_warped[0]);
				mask2_temp.copyTo(masks_warped[1]);
				
				vector<UMat> masks_warped_f(num_images);
				mask1_temp.copyTo(masks_warped_f[0]);
				mask2_temp.copyTo(masks_warped_f[1]);
				
				vector<UMat> images_warped_f(num_images);
				images_warped[0].convertTo(images_warped_f[0], CV_32F);
				images_warped[1].convertTo(images_warped_f[1], CV_32F);
				//exposure compensator
				Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN);
				Mat mask_warped;
				vector<Mat> img_warped(num_images);
				images_warped[0].copyTo(img_warped[0]);
				images_warped[1].copyTo(img_warped[1]);
				compensator->feed(corners, images_warped, masks_warped);
				
				//seam finder
				Ptr<SeamFinder> seam_finder;
				//cv::detail::GraphCutSeamFinder seam_finder;
				seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
				
				seam_finder->find(images_warped_f, corners, masks_warped_f);
				
				masks_warped_f[0].copyTo(mask1_temp);
				masks_warped_f[1].copyTo(mask2_temp);
				
				masks_warped_f.clear();
				images_warped.clear();
				images_warped_f.clear();
				
				for (int i = 0; i < num_images; i++) {
					masks_warped[i].copyTo(mask_warped);
					compensator->apply(i, corners[i], img_warped[i], mask_warped);
				}
				
				img_warped[0].copyTo(image1Updated);
				img_warped[1].copyTo(image2Updated);
				masks_warped.clear();
				img_warped.clear();
				//blend
				//create blender
				cv::Ptr<cv::detail::Blender> blender = cv::detail::Blender::createDefault(cv::detail::Blender::MULTI_BAND, false);
				//prepare resulting size of image
				blender->prepare(Rect(0, 0, pano.size().width, pano.size().height));
				//feed images and the mask areas to blend
				
				blender->feed(image1Updated, mask1_temp, Point2f (0,0));
				blender->feed(image2Updated, mask2_temp, Point2f (0,0));
				Mat pano_temp, result_mask;
				//blend
				blender->blend(pano_temp, result_mask);
				
                double u[4],v[4];
                for (int i=0; i< 4; i++){
                    u[i] = xx[i].at(k);
                    v[i] = yy[i].at(k);
                }
                
                std::vector<Point2f> pts1;
                for (int i=0; i<4; i++){
                    pts1.push_back(Point2f(u[i], v[i]));
                }

                Rect boudRectWarp;
                boudRectWarp = boundingRect(pts1);
                
                int bbx, bby, bbwidth, bbheight;
                bbx =  boudRectWarp.tl().x;
                bby =  boudRectWarp.tl().y;
                bbheight = boudRectWarp.height;
                bbwidth = boudRectWarp.width;

                Mat imtemp = pano_temp(Rect(bbx+x_offset, bby+y_offset, bbwidth, bbheight));
                // cv::imwrite("/opt/code/process/ccode/testview/b/"+name.at(k), imtemp);
                
                Mat M = findWarp(height, width, pts1);

                Mat result_image;
                
                cv::warpPerspective(imtemp, result_image, M, im1.size());
                cv::imwrite(output+"/"+name, result_image);    
                // cv::imwrite(datapath+name.at(k), view);

                int64 t_end = cv::getTickCount();
                process_time = (t_end - t_start)/cv::getTickFrequency();
                process_time = round(process_time*100)/100;

                if (view == "live") std::cout<< name << "   " << to_string(process_time) << std::endl;
                if (view == "json") {
                    json obj;
                    obj["name"] = name;
                    obj["time"] = process_time;
                    results["data"].push_back(obj);
                }        
            } catch( Exception e){
                // std::cout << e << std::endl;
            }
        }
        int64 end = cv::getTickCount();
        total_time = round((end-start)/cv::getTickFrequency());
        if (view == "live") std:: cout << "total_time: "<< to_string(total_time) << std::endl;
        if (view == "json") {
            results["total_time"] = total_time;
            std::cout<<results.dump() << std::endl;
        }
    } catch ( Exception e){
        // std::cout << e.what() << std::endl;
    }
}
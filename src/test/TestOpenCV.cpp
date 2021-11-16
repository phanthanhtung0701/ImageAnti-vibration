// TestOpenCV.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// OpenCV
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>
#include <nlohmann/json.hpp>

#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"

using namespace cv;
using namespace std;
using namespace cv::detail;

/**  @function main */
int main(int argc, char *argv[]) {
	int num_images = 2;
	vector<Point> corners(num_images);
	vector<UMat> masks_warped(num_images);
	vector<UMat> images_warped(num_images);
	vector<Mat> img(num_images);
	img[0] = cv::imread("/mnt/hs01/test/aaa/o/00000.jpg");
	img[1] = cv::imread("/mnt/hs01/test/aaa/o/00001.jpg");

	for (int i = 0; i < 2; ++i)
	{
		corners[i] = Point(0, 0);
		masks_warped[i].create(img[i].size(), CV_8U);
		masks_warped[i].setTo(Scalar::all(255));
		img[i].copyTo(images_warped[i]);
	}
	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN);
	Mat mask_warped;
	Mat img_warped;
	compensator->feed(corners, images_warped, masks_warped);
	for (int i = 0; i < 2; i++) {
		mask_warped.create(img[i].size(), CV_8U);
		img[i].copyTo(img_warped);
		mask_warped.setTo(Scalar::all(255));
		compensator->apply(i, corners[i], img_warped, mask_warped);
		cout << img_warped.size() << endl;
		cv::imwrite("/mnt/hs01/test/aaa/l/img_com" + to_string(i) + ".jpg", img_warped);
	}
}
// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

#include "homographysurf.h"

HomographySurf::HomographySurf(Mat img1, Mat img2){
    im1 = img1;
    im2 = img2;
    cv::cvtColor(im1, im1Gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(im2, im2Gray, cv::COLOR_BGR2GRAY);
    inliers_good = 0;
}

void HomographySurf::setReference(Mat img2){
    im2 = img2;
    cv::cvtColor(im2, im2Gray, cv::COLOR_BGR2GRAY);
    inliers_good = 0;
}

int HomographySurf::getNumberKp1(){
    return nkp1;
}

int HomographySurf::getNumberKp2(){
    return nkp2;
}

Mat HomographySurf::getRegistration(){
    return im1Reg;
}

Mat HomographySurf::getMatrix(){
    return h;
}

int HomographySurf::getNumberMatches(){
    return inliers_good;
}

bool HomographySurf::getMove(){
    return move;
}

void HomographySurf::setMove(bool nmove){
    move = nmove;
}

int findnInterpolated(double pixel, int shift){
    return int(pixel*shift);
}

bool checkMove(std::vector<Point2f> firstCorners, std::vector<Point2f> secondCorners, int height, int width){
    if ((abs(firstCorners[0].x - secondCorners[0].x) > width*DEVIATION_ANGLE
        || abs(firstCorners[1].x - secondCorners[1].x) > width*DEVIATION_ANGLE
        || abs(firstCorners[2].x - secondCorners[2].x) > width*DEVIATION_ANGLE
        || abs(firstCorners[3].x - secondCorners[3].x) > width*DEVIATION_ANGLE
        || abs(firstCorners[0].y - secondCorners[0].y) > height*DEVIATION_ANGLE
        || abs(firstCorners[1].y - secondCorners[1].y) > height*DEVIATION_ANGLE
        || abs(firstCorners[2].y - secondCorners[2].y) > height*DEVIATION_ANGLE
        || abs(firstCorners[3].y - secondCorners[3].y) > height*DEVIATION_ANGLE)
      
      || (secondCorners[0].x - firstCorners[0].x < 100 && secondCorners[0].y - firstCorners[0].y < 100       // check room out
      && secondCorners[1].x - firstCorners[1].x < 100 && secondCorners[1].y - firstCorners[1].y > 100
      && secondCorners[2].x - firstCorners[2].x > 100 && secondCorners[2].y - firstCorners[2].y > 100
      && secondCorners[3].x - firstCorners[3].x > 100 && secondCorners[3].y - firstCorners[3].y < 100)
    ){
        return true;    
    }
    else if (secondCorners[0].x - firstCorners[0].x > 100 && secondCorners[0].y - firstCorners[0].y > 100      // check room in
      && secondCorners[1].x - firstCorners[1].x > 100 && secondCorners[1].y - firstCorners[1].y < 100
      && secondCorners[2].x - firstCorners[2].x < 100 && secondCorners[2].y - firstCorners[2].y < 100
      && secondCorners[3].x - firstCorners[3].x < 100 && secondCorners[3].y - firstCorners[3].y > 100)
      {
          return true;
      }
    else return false;
}

void HomographySurf::getHomographyGPU(int he){

    cv::cuda::SURF_CUDA surf(he);
    cv::cuda::GpuMat keypoints1GPU, keypoints2GPU;
    cv::cuda::GpuMat descriptors1GPU, descriptors2GPU;
    
    cv::cuda::GpuMat img1Gpu(im1Gray);
    cv::cuda::GpuMat img2Gpu(im2Gray);

    surf(img1Gpu, cv::cuda::GpuMat(), keypoints1GPU, descriptors1GPU);
    surf(img2Gpu, cv::cuda::GpuMat(), keypoints2GPU, descriptors2GPU);
    
    img1Gpu.release();
    img2Gpu.release();
    if (keypoints1GPU.cols > 0 && keypoints2GPU.cols > 0){
        nkp1 = keypoints1GPU.cols;
        nkp2 = keypoints2GPU.cols;

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
	
        keypoints1GPU.release();
        keypoints2GPU.release();
        descriptors1GPU.release();
        descriptors2GPU.release();

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
            h = cv::findHomography(points1, points2, RANSAC, 0.5, inliers);
            inliers_good = count(inliers.begin(), inliers.end(), 1);
            //cuda::resetDevice();
        } 
        else inliers_good = 0;
    }
    else inliers_good = 0;
}

void HomographySurf::getHomographyCPU(int he) {
    Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(he);
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    surf->detectAndCompute( im1Gray, noArray(), keypoints1, descriptors1 );
    surf->detectAndCompute( im2Gray, noArray(), keypoints2, descriptors2 );

    if (keypoints1.size() > 0 && keypoints2.size() > 0){
        nkp1 = keypoints1.size();
        nkp2 = keypoints2.size();

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

            h = cv::findHomography(points1, points2, RANSAC, 0.5, inliers);

            inliers_good = count(inliers.begin(), inliers.end(), 1);
        }
        else inliers_good = 0;
    }
    else inliers_good = 0;
}

bool HomographySurf::warpImage(bool GPU){
    if (GPU) {
        getHomographyGPU(450);
    }
    else getHomographyCPU(450);
    int n = getNumberMatches();

    if (n >= 80){
        h = getMatrix();
        int height = im2.rows;
        int width = im2.cols;
        std::vector<Point2f> camera_corners,world_corners;
        camera_corners.push_back(Point2f(0, 0));
        camera_corners.push_back(Point2f(0, height - 1));
        camera_corners.push_back(Point2f(width - 1, height -1));
        camera_corners.push_back(Point2f(width - 1, 0));

        perspectiveTransform(camera_corners, world_corners, h);
        if(abs(world_corners[0].x - world_corners[2].x) < width*0.3
            || abs(world_corners[0].y - world_corners[2].y) < height*0.3
            || abs(world_corners[0].x - world_corners[2].x) > width*1.8
            || abs(world_corners[0].y - world_corners[2].y) > height*1.8) {
            return false;
        }

        bool check = checkMove(camera_corners, world_corners, height, width);
        // cout << check;
        // if (check) {
		// 	Size _s = Size(width, height);
		// 	Mat mask_pano;
		// 	mask_pano.create(_s, CV_8UC1);
		// 	mask_pano.setTo(Scalar::all(255));
		// 	warpPerspective(mask_pano, mask_pano, h, _s);
		// 	int white = cv::countNonZero(mask_pano);
		// 	int total_size = height*width;

		// 	double r = (double)white/(double)total_size;
		// 	if (r > 0.95) return false;
		// 	else move = true;
		// }
        if (check) move = true;
        warpPerspective(im1, im1Reg, h, im2.size());
        return true;
    }
    else return false;
}

Mat findWarp(int height, int width, std::vector<Point2f> corners){
    Mat h_temp ;
    Mat M_temp;

    std::vector<Point2f> preCorners;
    preCorners.push_back(Point2f(0, 0));
    preCorners.push_back(Point2f(0, height - 1));
    preCorners.push_back(Point2f(width - 1, height -1));
    preCorners.push_back(Point2f(width - 1, 0));

    M_temp = cv::getPerspectiveTransform(preCorners, corners);

    Rect boudRect;
    boudRect = boundingRect(corners);

    int bx, by, bwidth, bheight;
    bx =  boudRect.tl().x;
    by =  boudRect.tl().y;
    bheight = boudRect.height;
    bwidth = boudRect.width;

    cv::Mat_<double> T = (cv::Mat_<double>(3, 3) << 1, 0, -bx, 0, 1, -by, 0, 0, 1);
    Mat h1_temp = T*M_temp;
    h_temp = h1_temp.inv();
    return h_temp;
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

void list_dir(const char *path, std::vector<std::string> &files) {
   struct dirent *entry;
   DIR *dir = opendir(path);
   
   if (dir == NULL) {
      return;
   }
   while ((entry = readdir(dir)) != NULL) {
    if (entry->d_type == DT_REG){
        files.push_back(entry->d_name);
    }
   }
   closedir(dir);
}

int HomographySurf::getShift(){
    int height = im2.rows;
    int width = im2.cols;    
    h = getMatrix();
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

int HomographySurf::createMoving(string outputpath, string nameImage){ 
    std::vector<std::string> pre_image_reference;
    list_dir(outputpath.c_str(), pre_image_reference);
    std::sort(pre_image_reference.begin(), pre_image_reference.end());

    int k = nameImage.find('.');
    nameImage = nameImage.substr(0, k);
    int check = mkdir((outputpath+nameImage).c_str(), 0777);
    // if (check) std::cout<<"Unable to create directory"<<std::endl;

    int height = im2.rows;
    int width = im2.cols;    
    h = getMatrix();
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
    warpPerspective(im1, pano, h1, s);

    // std::cout << x_offset << " x " << y_offset << std::endl;
    // cv::imwrite("/opt/code/process/ccode/testview/pano.jpg", pano);

    // create interpolated image
    float sum;
    for (int i=0;i<4; i++){
        sum += abs(cornersWarped[i].x - cornersFirst[i].x)+abs(cornersWarped[i].y - cornersFirst[i].y);
    }
    sum /=8;
    int nInterpolated = findnInterpolated(PIXEL, int(sum));
    if (nInterpolated < 60) nInterpolated = 60;   // nInterpolated min = 60

    if (nInterpolated > pre_image_reference.size()) nInterpolated = pre_image_reference.size();
    std::vector<std::string> name(pre_image_reference.end()-nInterpolated, pre_image_reference.end());
    std::vector<double> xx[nInterpolated], yy[nInterpolated];
    for (int i=0; i<4; i++){
        xx[i] = linspace(cornersFirst[i].x, cornersWarped[i].x, nInterpolated);
        yy[i] = linspace(cornersFirst[i].y, cornersWarped[i].y, nInterpolated);
    }
    for (int k=0; k<nInterpolated; k++){
        Mat pano_temp = pano;
        Mat imReference = cv::imread(outputpath+name.at(k));

        std::cout << name.at(k)<< std::endl;
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

        // imReference.copyTo(pano_temp(cv::Rect(x_offset, y_offset, width, height)));
        int rr = 0;
        int cc = 0;
        for (int r=y_offset; r < y_offset+height; r++){
            int cc = 0;
            for (int c = x_offset; c < x_offset+width; c++){
                // if (pano_temp.at<Vec3b>(r, c)[0] == 0 && pano_temp.at<Vec3b>(r, c)[1] == 0 && pano_temp.at<Vec3b>(r, c)[2] == 0){
                //     pano_temp.at<Vec3b>(r,c) = imReference.at<Vec3b>(rr,cc);
                // }
                if (!(imReference.at<Vec3b>(rr,cc)[0] == 0 && imReference.at<Vec3b>(rr,cc)[1] == 0 && imReference.at<Vec3b>(rr,cc)[2] == 0)){
                    pano_temp.at<Vec3b>(r,c) = imReference.at<Vec3b>(rr,cc);
                }
                cc++;
            }
        rr++;
        }

        Mat imtemp = pano_temp(Rect(bbx+x_offset, bby+y_offset, bbwidth, bbheight));
        // cv::imwrite("/opt/code/process/ccode/testview/b/"+name.at(k), imtemp);
        
        Mat M = findWarp(height, width, pts1);

        Mat view;
        
        cv::warpPerspective(imtemp, view, M, im1.size());
        cv::imwrite(outputpath+nameImage+"/"+name.at(k), view);    
        // cv::imwrite(datapath+name.at(k), view);        
    }
    return int(sum);    
}

void HomographySurf::createMovingInter(string outputpath,string name){
    int k = name.find('.');
    name = name.substr(0, k);
    
    int check = mkdir((outputpath+name).c_str(), 0777);
    if (check) std::cout<<"Unable to create directory"<<std::endl;

    int height = im2.rows;
    int width = im2.cols;    
    h = getMatrix();

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
    warpPerspective(im1, pano, h1, s);

    //std::cout << x_offset << " x " << y_offset << std::endl;
    //create panorama
    //im2.copyTo(pano(cv::Rect(x_offset, y_offset, width, height)));
    int rr = 0;
    int cc = 0;
    for (int r=y_offset; r < y_offset+height; r++){
        int cc = 0;
        for (int c = x_offset; c < x_offset+width; c++){
            if (pano.at<Vec3b>(r, c)[0] == 0 && pano.at<Vec3b>(r, c)[1] == 0 && pano.at<Vec3b>(r, c)[2] == 0){
                pano.at<Vec3b>(r,c) = im2.at<Vec3b>(rr,cc);
            }
            cc++;
        }
        rr++;
    }
    // cv::imwrite(outputpath+name+"/pano.jpg", pano);
    
    // create interpolated image
    float sum;
    for (int i=0;i<4; i++){
        sum += abs(cornersWarped[i].x - cornersFirst[i].x)+abs(cornersWarped[i].y - cornersFirst[i].y);
    }
    sum /=8;
    int nInterpolated = findnInterpolated(PIXEL, int(sum));
    if (nInterpolated < 60) nInterpolated = 60;

    std::vector<double> xx[nInterpolated], yy[nInterpolated];
    for (int i=0; i<4; i++){
        xx[i] = linspace(cornersFirst[i].x, cornersWarped[i].x, nInterpolated);
        yy[i] = linspace(cornersFirst[i].y, cornersWarped[i].y, nInterpolated);
    }
    for (int k=0; k<nInterpolated; k++){
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

        Mat imtemp = pano(Rect(bbx+x_offset, bby+y_offset, bbwidth, bbheight));
        // cv::imwrite("/opt/code/process/ccode/testview/b/baseview"+to_string(k)+".jpg", imtemp);
        
        Mat M = findWarp(height, width, pts1);

        Mat view;
        
        cv::warpPerspective(imtemp, view, M, im1.size());
        cv::imwrite(outputpath+name+"/"+to_string(k)+".jpg", view);        
    }
}

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/cudaarithm.hpp"

#include <ctime>
#include <string>
#include <fstream>
#include <stdio.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h> 
#include <sys/types.h>
using namespace cv;
using namespace std;

int main(int argc, char *argv[]){
    std::cout << cuda::getCudaEnabledDeviceCount() << endl;

    for ( int i=0; i<cuda::getCudaEnabledDeviceCount(); i++){
        cout << i << endl;
        cv::cuda::printShortCudaDeviceInfo(i);
        cv::cuda::DeviceInfo dev_info(i);
        std::cout << dev_info.isCompatible() << std::endl;
    }
    cuda::setDevice(0);

    // cuda::resetDevice();
    return 0;
}

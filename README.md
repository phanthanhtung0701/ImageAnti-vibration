# py-process-media
Process Image &amp; Video

# build av cpu
#install tool

apt install gcc-7 g++-7 -y

apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libopencv-dev -y

cd /opt
mkdir process_media
cd process_media
git clone https://github.com/opencv/opencv.git
cd opencv 
cd ..
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
cd ..
cd opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D WITH_TBB=ON \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D CUDA_ARCH_BIN=7.5 \
-D BUILD_opencv_cudacodec=OFF \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_V4L=ON \
-D WITH_QT=OFF \
-D WITH_OPENGL=ON \
-D WITH_GSTREAMER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-D OPENCV_ENABLE_NONFREE=ON \
-D CMAKE_C_COMPILER=/usr/bin/gcc-7 \
-D CMAKE_CXX_COMPILER=g++-7 \
-D BUILD_EXAMPLES=ON ..

make -j4
make install
sudo ln -s /usr/local/lib64/pkgconfig/opencv.pc /usr/share/pkgconfig/
ldconfig
pkg-config --modversion opencv

apt-get install -y nlohmann-json-dev


#sua 
#/usr/include/nlohmann/json.hpp:6057:62:
#echo "sua /usr/include/nlohmann/json.hpp:6057:62: return *lhs.m_value.array < *rhs.m_value.array; thanh return (*lhs.m_value.array) < *rhs.m_value.array; "
# g++ -ggdb homographysurf.cpp process.cpp main.cpp `pkg-config --cflags --libs opencv` -o av_cpu


# build av gpu

# build moving
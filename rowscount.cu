#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

//Avilov Alexander Task 3, var 12

using namespace cv;
using namespace std;

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }

__global__ void calc(uchar *source, int *res, int rows, int cols)
{
    int regres = 0;
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= rows) return;
    for (int j = 0; j < cols; j++)
    {
      uchar3 w = *((uchar3*)&source[(i*cols + j)*3]);
      regres += w.x + w.y + w.z < 700;
    }
      res[i] = regres;
}

#define TILE_WIDTH 64

__global__ void sharedCalc(uchar3 *source, int *res, int rows, int cols)
{
    int regres = 0;
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int bx = blockIdx.x;
    int th = threadIdx.x;
    int blockSize = blockDim.x;
    if (i >= rows) return;

    __shared__ uchar3 cashe_tile[256][TILE_WIDTH];

    for(int p = 0; p < cols/TILE_WIDTH; ++p) //8
    {
      // загрузка подстрок в кеш __shared__
      for (int r = 0; r < 64; r++)
      {
        cashe_tile[th / 64 + r * 4][th % 64] = source[(th / 64 + r * 4 + +bx*blockSize)*cols + (th % 64 + p*64)];
      }
      __syncthreads();
      for (int j = 0; j < TILE_WIDTH; j++)
      {
        // вычисление кол-ва пикселей
        uchar3 w = cashe_tile[th][j];
        regres += w.x + w.y + w.z < 700;
      }
      __syncthreads();
    }
    res[i] = regres;
}

int main()
{
  cudaEvent_t startCUDA, stopCUDA;
  //clock_t startCPU;
  float elapsedTimeCUDA;
  //, elapsedTimeCPU;

    //picture variable
    Mat image;

    image = imread("fish1.png", CV_LOAD_IMAGE_COLOR);   // Read the file
    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    int * procres = new int[image.rows];

    for (int i = 0; i < image.rows; i++)
      procres[i] = 0;

    //===========================processor calculation===========================

    float elapsedTimeCPU;
    clock_t startCPU;
    startCPU = clock();

    for(int i = 0; i < image.rows; i++)
    {
        //pointer to 1st pixel in row
        Vec3b* p1 = image.ptr<Vec3b>(i);
        for (int j = 0; j < image.cols; j++)
              if (
                  p1[j][0] +
                  p1[j][1] +
                  p1[j][2] < 700)
              {
                procres[i]++;
              }
    }

    elapsedTimeCPU = (double)(clock()-startCPU)/CLOCKS_PER_SEC;
    //int stop = getTickCount();
    cout << "CPU sum time = " << elapsedTimeCPU*1000 << " ms\n";

    //=========================end processor calculation=========================

    //==============================GPU calculation==============================

    uchar *dev_img;
    uchar3 *dev3_img;
    int *res_arr;

    for (int i = 0; i < image.rows; i++)
      procres[i] = 0;
    //source image
    CHECK( cudaMalloc(&dev_img, 3*image.rows*image.cols));
    CHECK( cudaMemcpy(dev_img, image.data, 3*image.rows*image.cols,cudaMemcpyHostToDevice));

    CHECK( cudaMalloc(&dev3_img, 3*image.rows*image.cols));
    CHECK( cudaMemcpy(dev3_img, image.data, 3*image.rows*image.cols,cudaMemcpyHostToDevice));
    //res Array
    CHECK( cudaMalloc(&res_arr, image.rows*sizeof(int)));
    CHECK( cudaMemcpy(res_arr, procres, image.rows*sizeof(int),cudaMemcpyHostToDevice));

    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);
    cudaEventRecord(startCUDA,0);

    //calc<<< (image.rows+255)/256, 256>>>(dev_img, res_arr, image.rows, image.cols);
    sharedCalc<<< (image.rows+255)/256, 256>>>(dev3_img, res_arr, image.rows, image.cols);

    cudaEventRecord(stopCUDA,0);
    cudaEventSynchronize(stopCUDA);
    CHECK(cudaGetLastError());
    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);
    CHECK( cudaMemcpy(procres, res_arr, image.rows*sizeof(int), cudaMemcpyDeviceToHost));

    //============================end GPU calculation============================
    // for (int i = 0; i < image.rows; i++)
      // cout << procres[i] << endl;

    cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
    //cout << "CUDA memory throughput = " << 3*image.rows*image.cols*2/elapsedTimeCUDA/1024/1024/1.024 << " Gb/s\n";

    // check
    return 0;
}

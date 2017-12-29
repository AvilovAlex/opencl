#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <time.h>
#include <string.h>

using namespace cl;
using namespace std;
using namespace cv;

#define checkError(func) \
  if (errcode != CL_SUCCESS)\
  {\
    cout << "Error in " #func "\nError code = " << errcode << "\n";\
    exit(1);\
  }

#define checkErrorEx(command) \
  command; \
  checkError(command);

struct uchar3{
  uchar x,y,z;
};

int main()
{
  int device_index = 0;
  cl_int errcode;



      //picture variable
      cv::Mat image;

      image = cv::imread("fish1.png", CV_LOAD_IMAGE_COLOR);   // Read the file
      int rowCount = image.rows;
      int colCount = image.cols;
      if(! image.data )                              // Check for invalid input
      {
          cout <<  "Could not open or find the image" << std::endl ;
          return -1;
      }

      int * procresCPU = new int[image.rows];
      for (int i = 0; i < image.rows; i++)
        procresCPU[i] = 0;

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
                  procresCPU[i]++;
                }
      }

      elapsedTimeCPU = (double)(clock()-startCPU)/CLOCKS_PER_SEC;
      //int stop = getTickCount();
      cout << "CPU sum time = " << elapsedTimeCPU*1000 << " ms\n";

      //=========================end processor calculation=========================

      uchar *dev_img;
      uchar3 *dev3_img;
      int *res_arr;


      uchar3 * matrix = new uchar3[image.rows*image.cols];
      for (int i = 0; i < image.rows*image.cols; i++)
        matrix[i] = uchar3{image.data[i*3], image.data[i*3 + 1], image.data[i*3 + 2]};

    string sourceString =
      "struct uchr3{\n\
        uchar x,y,z;\n\
      };\n\
      __kernel void sharedCalc(__global struct uchr3 *source,__global int *res, int rows, int cols)\n\
      {\n\
          const int TILE_WIDTH = 32;\n\
          int regres = 0;\n\
          int i = get_global_id(0);\n\
          int bx = get_group_id(0);\n\
          int th = get_local_id(0);\n\
          int blockSize = get_local_size(0);\n\
          if (i >= rows) return;\n\
          __local struct uchr3 cashe_tile[256][32];\n\
          for(int p = 0; p < cols/TILE_WIDTH; ++p)\n\
          {\n\
            for (int r = 0; r < TILE_WIDTH; r++)\n\
            {\n\
              int ind = th / TILE_WIDTH + r * 256/TILE_WIDTH;\n\
              cashe_tile[ind][th % TILE_WIDTH] =\n\
              source[(ind + bx*blockSize)*cols + (th % TILE_WIDTH + p*TILE_WIDTH)];\n\
            }\n\
            barrier(CLK_LOCAL_MEM_FENCE);\n\
            for (int j = 0; j < TILE_WIDTH; j++)\n\
            {\n\
              struct uchr3 w = cashe_tile[th][j];\n\
              // uchr3 w = {0,0,0};\n\
              regres += w.x + w.y + w.z < 700;\n\
            }\n\
            barrier(CLK_LOCAL_MEM_FENCE);\n\
          }\n\
          res[i] = regres;\n\
    }";
  //получаем список доступных OpenCL-платформ (драйверов OpenCL)
  std::vector<Platform> platform;//массив в который будут записываться идентификаторы платформ
  checkErrorEx( errcode = Platform::get(&platform) );
  cout << "OpenCL platforms found: " << platform.size() << "\n";
  cout << "Platform[0] is : " << platform[0].getInfo<CL_PLATFORM_VENDOR>() << " ver. " << platform[0].getInfo<CL_PLATFORM_VERSION>() << "\n";
  //в полученном списке платформ находим устройство GPU (видеокарту)
  std::vector<Device> devices;
  platform[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
  cout << "GPGPU devices found: " << devices.size() << "\n";
  if (devices.size() == 0)
  {
      cout << "Warning: YOU DON'T HAVE GPGPU. Then CPU will be used instead.\n";
      checkErrorEx( errcode = platform[0].getDevices(CL_DEVICE_TYPE_CPU, &devices) );
      cout << "CPU devices found: " << devices.size() << "\n";
      if (devices.size() == 0) {cout << "Error: CPU devices not found\n"; exit(-1);}
  }
  cout << "Use device N " << device_index << ": " << devices[device_index].getInfo<CL_DEVICE_NAME>() << "\n";

  //создаем контекст на видеокарте
  checkErrorEx( Context context(devices, NULL, NULL, NULL, &errcode) );

  //создаем очередь задач для контекста
  checkErrorEx( CommandQueue queue(context, devices[device_index], CL_QUEUE_PROFILING_ENABLE, &errcode) );// третий параметр - свойства

  //создаем обьект-программу с заданным текстом программы
  checkErrorEx( Program program = Program(context, sourceString, false/*build*/, &errcode) );

  //компилируем и линкуем программу для видеокарты
  errcode = program.build(devices, "-cl-fast-relaxed-math -cl-no-signed-zeros -cl-mad-enable");
  if (errcode != CL_SUCCESS)
  {
      cout << "There were error during build kernel code. Please, check program code. Errcode = " << errcode << "\n";
      cout << "BUILD LOG: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[device_index]) << "\n";
      return 1;
  }
  int * procres = new int[image.rows];
  for (int i = 0; i < image.rows; i++)
    procres[i] = 0;

  //создаем буфферы в видеопамяти
  checkErrorEx( Buffer matrixCL = Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, image.rows*image.cols*sizeof(uchar3), matrix, &errcode ) );
  checkErrorEx( Buffer res = Buffer( context, CL_MEM_READ_WRITE, image.rows*sizeof(int), NULL , &errcode ) );

  //создаем объект - точку входа GPU-программы
  auto sharedCalc = KernelFunctor<Buffer, Buffer, int, int>(program, "sharedCalc");

  //создаем объект, соответствующий определенной конфигурации запуска kernel
  EnqueueArgs enqueueArgs(queue, rowCount/*globalSize*/, 256/*blockSize*/);

  //запускаем и ждем
  clock_t t0 = clock();
  Event event = sharedCalc(enqueueArgs, matrixCL, res, rowCount, colCount);
  checkErrorEx( errcode = event.wait() );
  clock_t t1 = clock();

  //считаем время
  cl_ulong time_start, time_end;
  errcode = event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &time_start);
  errcode = event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &time_end);
  double elapsedTimeGPU;
  if (errcode == CL_PROFILING_INFO_NOT_AVAILABLE)
    elapsedTimeGPU = (double)(t1-t0)/CLOCKS_PER_SEC;
  else
  {
    checkError(event.getEventProfilingInfo);
    elapsedTimeGPU = (double)(time_end - time_start)/1e9;
  }

  int* host_res = new int[rowCount];
  checkErrorEx( errcode = queue.enqueueReadBuffer(res, true, 0, sizeof(int)*rowCount, host_res, NULL, NULL) );
  // check
  for (int i = 0; i < rowCount; i++)
      if (::abs(host_res[i] - procresCPU[i]) > 1e-6)
      {
          cout << "Error in element " << i << ": procresGPU[i] = "
            << host_res[i] << " procresCPU[i] = " << procresCPU[i] << "\n";

          //exit(1);
      }
  int N = rowCount*colCount*3;
  cout << "CPU sum time = " << elapsedTimeCPU*1000 << " ms\n";
  cout << "CPU memory throughput = " << 2*(rowCount*sizeof(int)+N*sizeof(int))/elapsedTimeCPU/1024/1024/1024 << " Gb/s\n";
  cout << "GPU sum time = " << elapsedTimeGPU*1000 << " ms\n";
  cout << "GPU memory throughput = " << 2*(rowCount*sizeof(int)+N*sizeof(int))/elapsedTimeGPU/1024/1024/1024 << " Gb/s\n";
  return 0;
}

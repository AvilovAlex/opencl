#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>

#include <iostream>
#include <time.h>
#include <string.h>

using namespace cl;
using namespace std;

#define checkError(func) \
  if (errcode != CL_SUCCESS)\
  {\
    cout << "Error in " #func "\nError code = " << errcode << "\n";\
    exit(1);\
  }

#define checkErrorEx(command) \
  command; \
  checkError(command);

int main()
{
  int device_index = 0;
  cl_int errcode;

  const int rowCount = 25600;
  const int colCount = 1024;
  int * matrix = (int*) malloc(rowCount*colCount*sizeof(int));
  if (matrix==NULL) exit (1);

  //srand(0);

  //заполнение матрицы случайными числами
 for(int i = 0; i < rowCount; i++)
    for(int j = 0; j < colCount; j++)
      matrix[i*colCount + j] = colCount-j/*rand() % 100 + 1*/;

  int* countTranspArr = (int*) malloc(rowCount*sizeof(int));

  clock_t startCPU = clock();
  //#pragma omp parallel for
  for(int i = 0; i < rowCount; i++){
    int localCount = 0;
    for(int j = 0; j < colCount-1; j++){
      if(matrix[i*colCount + j] > matrix[i*colCount + j + 1])
        localCount++;
    }
    countTranspArr[i] = localCount;
  }

  int N = rowCount * colCount;
  double elapsedTimeCPU = (double)(clock()-startCPU)/CLOCKS_PER_SEC;

  //код kernel-функции
string sourceString = "\n\
__kernel void sharedCalc(__global uchar3 *source,__global int *res, int rows, int cols)\n\
{\n\
    int regres = 0;\n\
    int i = get_global_id(0);\n\
    int bx = get_group_id(0)s;\n\
    int th =  get_local_id(0);\n\
    int blockSize = get_local_size(0);\n\
    if (i >= rows) return;\n\

    __local uchar3 cashe_tile[256][64];\n\

    for(int p = 0; p < cols/64; ++p)\n\
    {\n\
      for (int r = 0; r < 64; r++)\n\
      {\n\
        cashe_tile[th / 64 + r * 4][th % 64] = source[(th / 64 + r * 4 + +bx*blockSize)*cols + (th % 64 + p*64)];\n\
      }\n\
      barrier(CLK_LOCAL_MEM_FENCE);\n\
      for (int j = 0; j < 64; j++)\n\
      {\n\
        uchar3 w = cashe_tile[th][j];\n\
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
  //создаем буфферы в видеопамяти
  checkErrorEx( Buffer matrixCL = Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int)*rowCount*colCount,  matrix, &errcode ) );
  checkErrorEx( Buffer res = Buffer( context, CL_MEM_READ_WRITE, sizeof(int)*rowCount,  NULL, &errcode ) );

  //создаем объект - точку входа GPU-программы
  auto countTransp2 = KernelFunctor<Buffer, Buffer, int, int>(program, "countTransp2");

  //создаем объект, соответствующий определенной конфигурации запуска kernel
  EnqueueArgs enqueueArgs(queue, (rowCount+255)/256*256/*globalSize*/, 256/*blockSize*/);

  //запускаем и ждем
  clock_t t0 = clock();
  Event event = countTransp2(enqueueArgs, matrixCL, res, rowCount, colCount);
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
      if (::abs(host_res[i] - countTranspArr[i]) > 1e-6)
      {
          cout << "Error in element N " << i << ": host_res[i] = "
            << host_res[i] << " countTranspArr[i] = " << countTranspArr[i] << "\n";
          exit(1);
      }
  cout << "CPU sum time = " << elapsedTimeCPU*1000 << " ms\n";
  cout << "CPU memory throughput = " << 2*(rowCount*sizeof(int)+N*sizeof(int))/elapsedTimeCPU/1024/1024/1024 << " Gb/s\n";
  cout << "GPU sum time = " << elapsedTimeGPU*1000 << " ms\n";
  cout << "GPU memory throughput = " << 2*(rowCount*sizeof(int)+N*sizeof(int))/elapsedTimeGPU/1024/1024/1024 << " Gb/s\n";
  return 0;
}

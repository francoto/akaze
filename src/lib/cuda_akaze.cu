#include <opencv2/features2d/features2d.hpp>
#include "cuda_akaze.h"
#include "cudautils.h"

#define CONVROW_W 160
#define CONVCOL_W 32
#define CONVCOL_H 40
#define CONVCOL_S 8

#define SCHARR_W 32
#define SCHARR_H 16

#define NLDSTEP_W 32
#define NLDSTEP_H 13

#define ORIENT_S (13 * 16)
#define EXTRACT_S 64

static const float atan2_p1 = 0.9997878412794807f * (float)(180 / CV_PI);
static const float atan2_p3 = -0.3258083974640975f * (float)(180 / CV_PI);
static const float atan2_p5 = 0.1555786518463281f * (float)(180 / CV_PI);
static const float atan2_p7 = -0.04432655554792128f * (float)(180 / CV_PI);

__device__ __constant__ float d_Kernel[21];
__device__ unsigned int d_PointCounter[1];

__device__ __constant__ int comp_idx_1[61 * 8];
__device__ __constant__ int comp_idx_2[61 * 8];

__device__ __constant__ float norm_factors[29];

__device__ float fastAtan2(float y, float x) {
  float ax = std::abs(x), ay = std::abs(y);
  float a, c, c2;
  if (ax >= ay) {
    c = ay / (ax + (float)DBL_EPSILON);
    c2 = c * c;
    a = (((atan2_p7 * c2 + atan2_p5) * c2 + atan2_p3) * c2 + atan2_p1) * c;
  } else {
    c = ax / (ay + (float)DBL_EPSILON);
    c2 = c * c;
    a = 90.f -
        (((atan2_p7 * c2 + atan2_p5) * c2 + atan2_p3) * c2 + atan2_p1) * c;
  }
  if (x < 0) a = 180.f - a;
  if (y < 0) a = 360.f - a;
  return a;
}

template <int RADIUS>
__global__ void ConvRowGPU(float *d_Result, float *d_Data, int width, int pitch,
                           int height) {
  __shared__ float data[CONVROW_W + 2 * RADIUS];
  const int tx = threadIdx.x;
  const int minx = blockIdx.x * CONVROW_W;
  const int maxx = min(minx + CONVROW_W, width);
  const int yptr = blockIdx.y * pitch;
  const int loadPos = minx + tx - RADIUS;
  const int writePos = minx + tx;

  if (loadPos < 0)
    data[tx] = d_Data[yptr];
  else if (loadPos >= width)
    data[tx] = d_Data[yptr + width - 1];
  else
    data[tx] = d_Data[yptr + loadPos];
  __syncthreads();
  if (writePos < maxx && tx < CONVROW_W) {
    float sum = 0.0f;
    for (int i = 0; i <= (2 * RADIUS); i++) sum += data[tx + i] * d_Kernel[i];
    d_Result[yptr + writePos] = sum;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Column convolution filter
///////////////////////////////////////////////////////////////////////////////
template <int RADIUS>
__global__ void ConvColGPU(float *d_Result, float *d_Data, int width, int pitch,
                           int height) {
  __shared__ float data[CONVCOL_W * (CONVCOL_H + 2 * RADIUS)];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int miny = blockIdx.y * CONVCOL_H;
  const int maxy = min(miny + CONVCOL_H, height) - 1;
  const int totStart = miny - RADIUS;
  const int totEnd = maxy + RADIUS;
  const int colStart = blockIdx.x * CONVCOL_W + tx;
  const int colEnd = colStart + (height - 1) * pitch;
  const int smemStep = CONVCOL_W * CONVCOL_S;
  const int gmemStep = pitch * CONVCOL_S;

  if (colStart < width) {
    int smemPos = ty * CONVCOL_W + tx;
    int gmemPos = colStart + (totStart + ty) * pitch;
    for (int y = totStart + ty; y <= totEnd; y += blockDim.y) {
      if (y < 0)
        data[smemPos] = d_Data[colStart];
      else if (y >= height)
        data[smemPos] = d_Data[colEnd];
      else
        data[smemPos] = d_Data[gmemPos];
      smemPos += smemStep;
      gmemPos += gmemStep;
    }
  }
  __syncthreads();
  if (colStart < width) {
    int smemPos = ty * CONVCOL_W + tx;
    int gmemPos = colStart + (miny + ty) * pitch;
    for (int y = miny + ty; y <= maxy; y += blockDim.y) {
      float sum = 0.0f;
      for (int i = 0; i <= 2 * RADIUS; i++)
        sum += data[smemPos + i * CONVCOL_W] * d_Kernel[i];
      d_Result[gmemPos] = sum;
      smemPos += smemStep;
      gmemPos += gmemStep;
    }
  }
}

template <int RADIUS>
double SeparableFilter(CudaImage &inimg, CudaImage &outimg, CudaImage &temp,
                       float *h_Kernel) {
  int width = inimg.width;
  int pitch = inimg.pitch;
  int height = inimg.height;
  float *d_DataA = inimg.d_data;
  float *d_DataB = outimg.d_data;
  float *d_Temp = temp.d_data;
  if (d_DataA == NULL || d_DataB == NULL || d_Temp == NULL) {
    printf("SeparableFilter: missing data\n");
    return 0.0;
  }
  // TimerGPU timer0(0);
  const unsigned int kernelSize = (2 * RADIUS + 1) * sizeof(float);
  safeCall(cudaMemcpyToSymbolAsync(d_Kernel, h_Kernel, kernelSize));

  dim3 blockGridRows(iDivUp(width, CONVROW_W), height);
  dim3 threadBlockRows(CONVROW_W + 2 * RADIUS);
  ConvRowGPU<RADIUS> << <blockGridRows, threadBlockRows>>>
      (d_Temp, d_DataA, width, pitch, height);
  // checkMsg("ConvRowGPU() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  dim3 blockGridColumns(iDivUp(width, CONVCOL_W), iDivUp(height, CONVCOL_H));
  dim3 threadBlockColumns(CONVCOL_W, CONVCOL_S);
  ConvColGPU<RADIUS> << <blockGridColumns, threadBlockColumns>>>
      (d_DataB, d_Temp, width, pitch, height);
  // checkMsg("ConvColGPU() execution failed\n");
  // safeCall(cudaThreadSynchronize());

  double gpuTime = 0;  // timer0.read();
#ifdef VERBOSE
  printf("SeparableFilter time =        %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

template <int RADIUS>
double LowPass(CudaImage &inimg, CudaImage &outimg, CudaImage &temp,
               double var) {
  float kernel[2 * RADIUS + 1];
  float kernelSum = 0.0f;
  for (int j = -RADIUS; j <= RADIUS; j++) {
    kernel[j + RADIUS] = (float)expf(-(double)j * j / 2.0 / var);
    kernelSum += kernel[j + RADIUS];
  }
  for (int j = -RADIUS; j <= RADIUS; j++) kernel[j + RADIUS] /= kernelSum;
  return SeparableFilter<RADIUS>(inimg, outimg, temp, kernel);
}

double LowPass(CudaImage &inimg, CudaImage &outimg, CudaImage &temp, double var,
               int kernsize) {
  if (kernsize <= 5)
    return LowPass<2>(inimg, outimg, temp, var);
  else if (kernsize <= 7)
    return LowPass<3>(inimg, outimg, temp, var);
  else if (kernsize <= 9)
    return LowPass<4>(inimg, outimg, temp, var);
  else {
    if (kernsize > 11)
      std::cerr << "Kernels larger than 11 not implemented" << std::endl;
    return LowPass<5>(inimg, outimg, temp, var);
  }
}

__global__ void Scharr(float *imgd, float *lxd, float *lyd, int width,
                       int pitch, int height) {
#define BW (SCHARR_W + 2)
  __shared__ float buffer[BW * (SCHARR_H + 2)];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * SCHARR_W + tx;
  int y = blockIdx.y * SCHARR_H + ty;
  int xp = (x == 0 ? 1 : (x > width ? width - 2 : x - 1));
  int yp = (y == 0 ? 1 : (y > height ? height - 2 : y - 1));
  buffer[ty * BW + tx] = imgd[yp * pitch + xp];
  __syncthreads();
  if (x < width && y < height && tx < SCHARR_W && ty < SCHARR_H) {
    float *b = buffer + (ty + 1) * BW + (tx + 1);
    float ul = b[-BW - 1];
    float ur = b[-BW + 1];
    float ll = b[+BW - 1];
    float lr = b[+BW + 1];
    lxd[y * pitch + x] = 3.0f * (lr - ll + ur - ul) + 10.0f * (b[+1] - b[-1]);
    lyd[y * pitch + x] = 3.0f * (lr + ll - ur - ul) + 10.0f * (b[BW] - b[-BW]);
  }
}

double Scharr(CudaImage &img, CudaImage &lx, CudaImage &ly) {
  // TimerGPU timer0(0);
  dim3 blocks(iDivUp(img.width, SCHARR_W), iDivUp(img.height, SCHARR_H));
  dim3 threads(SCHARR_W + 2, SCHARR_H + 2);
  Scharr << <blocks, threads>>>
      (img.d_data, lx.d_data, ly.d_data, img.width, img.pitch, img.height);
  // checkMsg("Scharr() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  double gpuTime = 0;  // timer0.read();
#ifdef VERBOSE
  printf("Scharr time          =        %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

__global__ void Flow(float *imgd, float *flowd, int width, int pitch,
                     int height, DIFFUSIVITY_TYPE type, float invk) {
#define BW (SCHARR_W + 2)
  __shared__ float buffer[BW * (SCHARR_H + 2)];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * SCHARR_W + tx;
  int y = blockIdx.y * SCHARR_H + ty;
  int xp = (x == 0 ? 1 : (x > width ? width - 2 : x - 1));
  int yp = (y == 0 ? 1 : (y > height ? height - 2 : y - 1));
  buffer[ty * BW + tx] = imgd[yp * pitch + xp];
  __syncthreads();
  if (x < width && y < height && tx < SCHARR_W && ty < SCHARR_H) {
    float *b = buffer + (ty + 1) * BW + (tx + 1);
    float ul = b[-BW - 1];
    float ur = b[-BW + 1];
    float ll = b[+BW - 1];
    float lr = b[+BW + 1];
    float lx = 3.0f * (lr - ll + ur - ul) + 10.0f * (b[+1] - b[-1]);
    float ly = 3.0f * (lr + ll - ur - ul) + 10.0f * (b[BW] - b[-BW]);
    float dif2 = invk * (lx * lx + ly * ly);
    if (type == PM_G1)
      flowd[y * pitch + x] = exp(-dif2);
    else if (type == PM_G2)
      flowd[y * pitch + x] = 1.0f / (1.0f + dif2);
    else if (type == WEICKERT)
      flowd[y * pitch + x] = 1.0f - exp(-3.315 / (dif2 * dif2 * dif2 * dif2));
    else
      flowd[y * pitch + x] = 1.0f / sqrt(1.0f + dif2);
  }
}

double Flow(CudaImage &img, CudaImage &flow, DIFFUSIVITY_TYPE type,
            float kcontrast) {
  // TimerGPU timer0(0);
  dim3 blocks(iDivUp(img.width, SCHARR_W), iDivUp(img.height, SCHARR_H));
  dim3 threads(SCHARR_W + 2, SCHARR_H + 2);
  Flow << <blocks, threads>>> (img.d_data, flow.d_data, img.width, img.pitch,
                               img.height, type,
                               1.0f / (kcontrast * kcontrast));
  //  //checkMsg("Flow() execution failed\n");
  //  //safeCall(cudaThreadSynchronize());
  double gpuTime = 0;  // timer0.read();
#ifdef VERBOSE
  printf("Flow time =                   %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

struct NLDStruct {
  float *imgd;
  float *flod;
  float *temd;
  int width;
  int pitch;
  int height;
  float stepsize;
};

__global__ void NLDStep(float *imgd, float *flod, float *temd, int width,
                        int pitch, int height, float stepsize) {
#undef BW
#define BW (NLDSTEP_W + 2)
  __shared__ float ibuff[BW * (NLDSTEP_H + 2)];
  __shared__ float fbuff[BW * (NLDSTEP_H + 2)];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * NLDSTEP_W + tx;
  int y = blockIdx.y * NLDSTEP_H + ty;
  int xp = (x == 0 ? 0 : (x > width ? width - 1 : x - 1));
  int yp = (y == 0 ? 0 : (y > height ? height - 1 : y - 1));
  ibuff[ty * BW + tx] = imgd[yp * pitch + xp];
  fbuff[ty * BW + tx] = flod[yp * pitch + xp];
  __syncthreads();
  if (tx < NLDSTEP_W && ty < NLDSTEP_H && x < width && y < height) {
    float *ib = ibuff + (ty + 1) * BW + (tx + 1);
    float *fb = fbuff + (ty + 1) * BW + (tx + 1);
    float ib0 = ib[0];
    float fb0 = fb[0];
    float xpos = (fb0 + fb[+1]) * (ib[+1] - ib0);
    float xneg = (fb0 + fb[-1]) * (ib0 - ib[-1]);
    float ypos = (fb0 + fb[+BW]) * (ib[+BW] - ib0);
    float yneg = (fb0 + fb[-BW]) * (ib0 - ib[-BW]);
    temd[y * pitch + x] = stepsize * (xpos - xneg + ypos - yneg);
  }
}

__global__ void NLDStep(NLDStruct s) {
#undef BW
#define BW (NLDSTEP_W + 2)
  __shared__ float ibuff[BW * (NLDSTEP_H + 2)];
  __shared__ float fbuff[BW * (NLDSTEP_H + 2)];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * NLDSTEP_W + tx;
  int y = blockIdx.y * NLDSTEP_H + ty;
  int xp = (x == 0 ? 0 : (x > s.width ? s.width - 1 : x - 1));
  int yp = (y == 0 ? 0 : (y > s.height ? s.height - 1 : y - 1));
  ibuff[ty * BW + tx] = s.imgd[yp * s.pitch + xp];
  fbuff[ty * BW + tx] = s.flod[yp * s.pitch + xp];
  __syncthreads();
  if (tx < NLDSTEP_W && ty < NLDSTEP_H && x < s.width && y < s.height) {
    float *ib = ibuff + (ty + 1) * BW + (tx + 1);
    float *fb = fbuff + (ty + 1) * BW + (tx + 1);
    float ib0 = ib[0];
    float fb0 = fb[0];
    float xpos = (fb0 + fb[+1]) * (ib[+1] - ib0);
    float xneg = (fb0 + fb[-1]) * (ib0 - ib[-1]);
    float ypos = (fb0 + fb[+BW]) * (ib[+BW] - ib0);
    float yneg = (fb0 + fb[-BW]) * (ib0 - ib[-BW]);
    s.imgd[y * s.pitch + x] =
        s.imgd[y * s.pitch + x] +
        s.stepsize * (xpos - xneg + ypos - yneg);  // temd[y*pitch + x];
  }
}

__global__ void NLDUpdate(float *imgd, float *temd, int width, int pitch,
                          int height) {
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  if (x < width && y < height) {
    int p = y * pitch + x;
    imgd[p] = imgd[p] + temd[p];
  }
}

double NLDStep(CudaImage &img, CudaImage &flow, CudaImage &temp,
               float stepsize) {
  // TimerGPU timer0(0);
  dim3 blocks0(iDivUp(img.width, NLDSTEP_W), iDivUp(img.height, NLDSTEP_H));
  dim3 threads0(NLDSTEP_W + 2, NLDSTEP_H + 2);
  NLDStruct s;
  s.imgd = img.d_data;
  s.flod = flow.d_data;
  s.temd = temp.d_data;
  s.width = img.width;
  s.pitch = img.pitch;
  s.height = img.height;
  s.stepsize = 0.5 * stepsize;
  NLDStep<<<blocks0, threads0>>>(img.d_data, flow.d_data, temp.d_data,
                                 img.width, img.pitch, img.height, 0.5f*stepsize);
  // checkMsg("NLDStep() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  dim3 blocks1(iDivUp(img.width, 32), iDivUp(img.height, 16));
  dim3 threads1(32, 16);
    NLDUpdate<<<blocks1, threads1>>>(img.d_data, temp.d_data, img.width,
                                     img.pitch, img.height);
  //  checkMsg("NLDUpdate() execution failed\n");
//    safeCall(cudaThreadSynchronize());
  double gpuTime = 0;  // timer0.read();
#ifdef VERBOSE
  printf("NLDStep time =                %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

__global__ void HalfSample(float *iimd, float *oimd, int iwidth, int iheight,
                           int ipitch, int owidth, int oheight, int opitch) {
  __shared__ float buffer[16 * 33];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * 16 + tx;
  int y = blockIdx.y * 16 + ty;
  if (x >= owidth || y >= oheight) return;
  float *ptri = iimd + (2 * y) * ipitch + (2 * x);
    if (2*owidth==iwidth) {
        buffer[ty*32 + tx] = owidth*(ptri[0] + ptri[1]);
        ptri += ipitch;
        buffer[ty*32 + tx + 16] = owidth*(ptri[0] + ptri[1]);
        if (ty==15) {
            ptri += ipitch;
            buffer[tx + 32*16] = owidth*(ptri[0] + ptri[1]);
        } else if (y*2+3 == iheight) {
            ptri += ipitch;
            buffer[tx + 32*(ty+1)] = owidth*(ptri[0] + ptri[1]);
        }
    } else {
        float f0 = owidth - x;
        float f2 = 1 + x;
        buffer[ty*32 + tx] = f0*ptri[0] + owidth*ptri[1] + f2*ptri[2];
        ptri += ipitch;
        buffer[ty*32 + tx + 16] = f0*ptri[0] + owidth*ptri[1] + f2*ptri[2];
        if (ty==15 && 2*oheight!=iheight) {
            ptri += ipitch;
            buffer[tx + 32*16] = f0*ptri[0] + owidth*ptri[1] + f2*ptri[1];
        } else if (y*2+3 == iheight && 2*oheight!=iheight) {
            ptri += ipitch;
            buffer[tx + 32*(ty+1)] = f0*ptri[0] + owidth*ptri[1] + f2*ptri[2];
        }
    }
/*  if (2 * owidth == iwidth) {
    buffer[ty * 32 + tx] = owidth * (ptri[0] + ptri[1]);
    ptri += ipitch;
    buffer[ty * 32 + tx + 16] = owidth * (ptri[0] + ptri[1]);
    if (ty == 15) {
      ptri += ipitch;
      buffer[tx + 32 * 16] = owidth * (ptri[0] + ptri[1]);
    }
  } else {
    float f0 = owidth - x;
    float f2 = 1 + x;
    buffer[ty * 32 + tx] = f0 * ptri[0] + owidth * ptri[1] + f2 * ptri[2];
    ptri += ipitch;
    buffer[ty * 32 + tx + 16] = f0 * ptri[0] + owidth * ptri[1] + f2 * ptri[2];
    if (ty == 15 && 2 * oheight != iheight) {
      ptri += ipitch;
      buffer[tx + 32 * 16] = f0 * ptri[0] + owidth * ptri[1] + f2 * ptri[1];
    }
  }*/
  __syncthreads();
  float *buff = buffer + 32 * ty + tx;
  if (2 * oheight == iheight)
    oimd[y * opitch + x] = oheight * (buff[0] + buff[16]) / (iwidth * iheight);
  else {
    float f0 = oheight - y;
    float f2 = 1 + y;
    oimd[y * opitch + x] = (f0 * buff[0] + oheight * buff[16] + f2 * buff[32]) /
                           (iwidth * iheight);
  }
}

__global__ void HalfSample2(float *iimd, float *oimd, int ipitch, int owidth,
                            int oheight, int opitch) {
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  if (x >= owidth || y >= oheight) return;
  float *ptr = iimd + (2 * y) * ipitch + (2 * x);
  oimd[y * opitch + x] =
      0.25f * (ptr[0] + ptr[1] + ptr[ipitch + 0] + ptr[ipitch + 1]);
}

double HalfSample(CudaImage &inimg, CudaImage &outimg) {
  // TimerGPU timer0(0);
  if (inimg.width == 2 * outimg.width && inimg.height == 2 * outimg.height) {
    dim3 blocks(iDivUp(outimg.width, 32), iDivUp(outimg.height, 16));
    dim3 threads(32, 16);
    HalfSample2 << <blocks, threads>>> (inimg.d_data, outimg.d_data,
                                        inimg.pitch, outimg.width,
                                        outimg.height, outimg.pitch);
  } else {
    dim3 blocks(iDivUp(outimg.width, 16), iDivUp(outimg.height, 16));
    dim3 threads(16, 16);
    HalfSample << <blocks, threads>>> (inimg.d_data, outimg.d_data, inimg.width,
                                       inimg.height, inimg.pitch, outimg.width,
                                       outimg.height, outimg.pitch);
  }
  // checkMsg("HalfSample() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  double gpuTime = 0;  // timer0.read();
#ifdef VERBOSE
  printf("HalfSample time =             %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

double Copy(CudaImage &inimg, CudaImage &outimg) {
  // TimerGPU timer0(0);
  double gpuTime = 0;  // timer0.read();
  safeCall(cudaMemcpy2DAsync(outimg.d_data, sizeof(float) * outimg.pitch,
                             inimg.d_data, sizeof(float) * outimg.pitch,
                             sizeof(float) * inimg.width, inimg.height,
                             cudaMemcpyDeviceToDevice));
#ifdef VERBOSE
  printf("Copy time =                   %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

float *AllocBuffers(int width, int height, int num, int omax, int maxpts,
                    std::vector<CudaImage> &buffers, cv::KeyPoint *&pts,
                    CudaImage *&ims) {
  buffers.resize(omax * num);
  int w = width;
  int h = height;
  int p = iAlignUp(w, 128);
  int size = 0;
  for (int i = 0; i < omax; i++) {
    for (int j = 0; j < num; j++) {
      CudaImage &buf = buffers[i * num + j];
      buf.width = w;
      buf.height = h;
      buf.pitch = p;
      buf.d_data = (float *)((long)size);
      size += h * p;
    }
    w /= 2;
    h /= 2;
    p = iAlignUp(w, 128);
  }
  int ptsstart = size;
  size += sizeof(cv::KeyPoint) * maxpts / sizeof(float);
  int imgstart = size;
  size += sizeof(CudaImage) * (num * omax + sizeof(float) - 1) / sizeof(float);
  float *memory = NULL;
  size_t pitch;
  safeCall(cudaMallocPitch((void **)&memory, &pitch, (size_t)4096,
                           (size + 4095) / 4096 * sizeof(float)));
  for (int i = 0; i < omax * num; i++) {
    CudaImage &buf = buffers[i];
    buf.d_data = memory + (long)buf.d_data;
  }
  pts = (cv::KeyPoint *)(memory + ptsstart);
  ims = (CudaImage *)(memory + imgstart);
  return memory;
}

void FreeBuffers(float *buffers) { safeCall(cudaFree(buffers)); }

__device__ unsigned int d_Maxval[1];
__device__ int d_Histogram[512];

#define CONTRAST_W 64
#define CONTRAST_H 7
#define HISTCONT_W 64
#define HISTCONT_H 8
#define HISTCONT_R 4

__global__ void MaxContrast(float *imgd, float *cond, int width, int pitch,
                            int height) {
#define WID (CONTRAST_W + 2)
  __shared__ float buffer[WID * (CONTRAST_H + 2)];
  __shared__ unsigned int maxval[32];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  if (tx < 32 && !ty) maxval[tx] = 0.0f;
  __syncthreads();
  int x = blockIdx.x * CONTRAST_W + tx;
  int y = blockIdx.y * CONTRAST_H + ty;
  if (x >= width || y >= height) return;
  float *b = buffer + ty * WID + tx;
  b[0] = imgd[y * pitch + x];
  __syncthreads();
  if (tx < CONTRAST_W && ty < CONTRAST_H && x < width - 2 && y < height - 2) {
    float dx = 3.0f * (b[0] - b[2] + b[2 * WID] - b[2 * WID + 2]) +
               10.0f * (b[WID] - b[WID + 2]);
    float dy = 3.0f * (b[0] + b[2] - b[2 * WID] - b[2 * WID + 2]) +
               10.0f * (b[1] - b[2 * WID + 1]);
    float grad = sqrt(dx * dx + dy * dy);
    cond[(y + 1) * pitch + (x + 1)] = grad;
    unsigned int *gradi = (unsigned int *)&grad;
    atomicMax(maxval + (tx & 31), *gradi);
  }
  __syncthreads();
  if (tx < 32 && !ty) atomicMax(d_Maxval, maxval[tx]);
}

__global__ void HistContrast(float *cond, int width, int pitch, int height,
                             float imaxval, int nbins) {
  __shared__ int hist[512];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int i = ty * HISTCONT_W + tx;
  if (i < nbins) hist[i] = 0;
  __syncthreads();
  int x = blockIdx.x * HISTCONT_W + tx;
  int y = blockIdx.y * HISTCONT_H * HISTCONT_R + ty;
  if (x > 0 && x < width - 1) {
    for (int i = 0; i < HISTCONT_R; i++) {
      if (y > 0 && y < height - 1) {
        int idx = min((int)(nbins * cond[y * pitch + x] * imaxval), nbins - 1);
        atomicAdd(hist + idx, 1);
      }
      y += HISTCONT_H;
    }
  }
  __syncthreads();
  if (i < nbins && hist[i] > 0) atomicAdd(d_Histogram + i, hist[i]);
}

double ContrastPercentile(CudaImage &img, CudaImage &temp, CudaImage &blur,
                          float perc, int nbins, float &contrast) {
  // TimerGPU timer0(0);
  LowPass(img, blur, temp, 1.0f, 5);

  float h_Maxval = 0.0f;
  safeCall(cudaMemcpyToSymbolAsync(d_Maxval, &h_Maxval, sizeof(float)));
  dim3 blocks1(iDivUp(img.width, CONTRAST_W), iDivUp(img.height, CONTRAST_H));
  dim3 threads1(CONTRAST_W + 2, CONTRAST_H + 2);
  MaxContrast << <blocks1, threads1>>>
      (blur.d_data, temp.d_data, blur.width, blur.pitch, blur.height);
  // checkMsg("MaxContrast() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  safeCall(cudaMemcpyFromSymbolAsync(&h_Maxval, d_Maxval, sizeof(float)));

  if (nbins > 512) {
    printf(
        "Warning: Largest number of possible bins in ContrastPercentile() is "
        "512\n");
    nbins = 512;
  }
  int h_Histogram[512];
  memset(h_Histogram, 0, nbins * sizeof(int));
  safeCall(
      cudaMemcpyToSymbolAsync(d_Histogram, h_Histogram, nbins * sizeof(int)));
  dim3 blocks2(iDivUp(temp.width, HISTCONT_W),
               iDivUp(temp.height, HISTCONT_H * HISTCONT_R));
  dim3 threads2(HISTCONT_W, HISTCONT_H);
  HistContrast << <blocks2, threads2>>> (temp.d_data, temp.width, temp.pitch,
                                         temp.height, 1.0f / h_Maxval, nbins);
  safeCall(
      cudaMemcpyFromSymbolAsync(h_Histogram, d_Histogram, nbins * sizeof(int)));

  int npoints = (temp.width - 2) * (temp.height - 2);
  int nthreshold = (int)(npoints * perc);
  int k = 0, nelements = 0;
  for (k = 0; nelements < nthreshold && k < nbins; k++)
    nelements += h_Histogram[k];
  contrast = (nelements < nthreshold ? 0.03f : h_Maxval * ((float)k / nbins));

  double gpuTime = 0;  // timer0.read();
#ifdef VERBOSE
  printf("ContrastPercentile time =     %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

__global__ void Derivate(float *imd, float *lxd, float *lyd, int width,
                         int pitch, int height, int step, float fac1,
                         float fac2) {
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  if (x >= width || y >= height) return;
  int xl = (x < step ? step - x : x - step);
  int xh = (x >= width - step ? 2 * width - x - step - 2 : x + step);
  int yl = (y < step ? step - y : y - step);
  int yh = (y >= height - step ? 2 * height - y - step - 2 : y + step);
  float ul = imd[yl * pitch + xl];
  float ur = imd[yl * pitch + xh];
  float ll = imd[yh * pitch + xl];
  float lr = imd[yh * pitch + xh];
  float cl = imd[y * pitch + xl];
  float cr = imd[y * pitch + xh];
  lxd[y * pitch + x] = fac1 * (ur + lr - ul - ll) + fac2 * (cr - cl);
  float uc = imd[yl * pitch + x];
  float lc = imd[yh * pitch + x];
  lyd[y * pitch + x] = fac1 * (lr + ll - ur - ul) + fac2 * (lc - uc);
}

__global__ void HessianDeterminant(float *lxd, float *lyd, float *detd,
                                   int width, int pitch, int height, int step,
                                   float fac1, float fac2) {
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  if (x >= width || y >= height) return;
  int xl = (x < step ? step - x : x - step);
  int xh = (x >= width - step ? 2 * width - x - step - 2 : x + step);
  int yl = (y < step ? step - y : y - step);
  int yh = (y >= height - step ? 2 * height - y - step - 2 : y + step);
  float ul = lxd[yl * pitch + xl];
  float ur = lxd[yl * pitch + xh];
  float ll = lxd[yh * pitch + xl];
  float lr = lxd[yh * pitch + xh];
  float cl = lxd[y * pitch + xl];
  float cr = lxd[y * pitch + xh];
  float lxx = fac1 * (ur + lr - ul - ll) + fac2 * (cr - cl);
  float uc = lxd[yl * pitch + x];
  float lc = lxd[yh * pitch + x];
  float lyx = fac1 * (lr + ll - ur - ul) + fac2 * (lc - uc);
  ul = lyd[yl * pitch + xl];
  ur = lyd[yl * pitch + xh];
  ll = lyd[yh * pitch + xl];
  lr = lyd[yh * pitch + xh];
  uc = lyd[yl * pitch + x];
  lc = lyd[yh * pitch + x];
  float lyy = fac1 * (lr + ll - ur - ul) + fac2 * (lc - uc);
  detd[y * pitch + x] = lxx * lyy - lyx * lyx;
}

double HessianDeterminant(CudaImage &img, CudaImage &lx, CudaImage &ly,
                          int step) {
  // TimerGPU timer0(0);
  float w = 10.0 / 3.0;
  float fac1 = 1.0 / (2.0 * (w + 2.0));
  float fac2 = w * fac1;
  dim3 blocks(iDivUp(img.width, 32), iDivUp(img.height, 16));
  dim3 threads(32, 16);
  Derivate << <blocks, threads>>> (img.d_data, lx.d_data, ly.d_data, img.width,
                                   img.pitch, img.height, step, fac1, fac2);
  // checkMsg("Derivate() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  HessianDeterminant << <blocks, threads>>> (lx.d_data, ly.d_data, img.d_data,
                                             img.width, img.pitch, img.height,
                                             step, fac1, fac2);
  // checkMsg("HessianDeterminant() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  double gpuTime = 0;  // timer0.read();
#ifdef VERBOSE
  printf("HessianDeterminant time =     %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

__global__ void FindExtrema(float *imd, float *imp, float *imn, int maxx,
                            int pitch, int maxy, int border, float dthreshold,
                            int scale, int octave, float size,
                            cv::KeyPoint *pts, int maxpts) {
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  if (x < border || x >= maxx || y < border || y >= maxy) return;
  int p = y * pitch + x;
  float v = imd[p];
  if (v > dthreshold && v > imd[p - pitch - 1] && v > imd[p + pitch + 1] &&
      v > imd[p - pitch + 1] && v > imd[p - pitch + 1] && v > imd[p - 1] &&
      v > imd[p + 1] && v > imd[p + pitch] && v > imd[p - pitch] &&
      v >= imn[p] && v >= imp[p]) {
    float dx = 0.5f * (imd[p + 1] - imd[p - 1]);
    float dy = 0.5f * (imd[p + pitch] - imd[p - pitch]);
    float dxx = imd[p + 1] + imd[p - 1] - 2.0f * v;
    float dyy = imd[p + pitch] + imd[p - pitch] - 2.0f * v;
    float dxy = 0.25f * (imd[p + pitch + 1] + imd[p - pitch - 1] -
                         imd[p + pitch - 1] - imd[p - pitch + 1]);
    float det = dxx * dyy - dxy * dxy;
    float idet = (det != 0.0f ? 1.0f / det : 0.0f);
    float dst0 = idet * (dxy * dy - dyy * dx);
    float dst1 = idet * (dxy * dx - dxx * dy);
    if (dst0 >= -1.0f && dst0 <= 1.0f && dst1 >= -1.0f && dst1 <= 1.0f) {
      unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
      if (idx < maxpts) {
        cv::KeyPoint &point = pts[idx];
        point.response = v;
        point.size = 2.0f * size;
        point.octave = octave;
        point.class_id = scale;
        int ratio = (1 << octave);
        point.pt.x = ratio * (x + dst0);
        point.pt.y = ratio * (y + dst1);
        point.angle = 0.0f;
        // printf("XXX %d %d %.2f %.2f XXX\n", x, y, dst0, dst1);
      }
    }
  }
}

double FindExtrema(CudaImage &img, CudaImage &imgp, CudaImage &imgn,
                   float border, float dthreshold, int scale, int octave,
                   float size, cv::KeyPoint *pts, int maxpts) {
  // TimerGPU timer0(0);
  dim3 blocks(iDivUp(img.width, 32), iDivUp(img.height, 16));
  dim3 threads(32, 16);
  int b = (int)border;
  FindExtrema << <blocks, threads>>>
      (img.d_data, imgp.d_data, imgn.d_data, img.width - b, img.pitch,
       img.height - b, b, dthreshold, scale, octave, size, pts, maxpts);
  // checkMsg("FindExtrema() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  double gpuTime = 0;  // timer0.read();
#ifdef VERBOSE
  printf("FindExtrema time =            %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

void ClearPoints() {
  int totPts = 0;
  safeCall(cudaMemcpyToSymbolAsync(d_PointCounter, &totPts, sizeof(int)));
}

int GetPoints(std::vector<cv::KeyPoint> &h_pts, cv::KeyPoint *d_pts) {
  int numPts = 0;
  safeCall(cudaMemcpyFromSymbolAsync(&numPts, d_PointCounter, sizeof(int)));
  h_pts.resize(numPts);
  safeCall(cudaMemcpy((float *)&h_pts[0], d_pts, sizeof(cv::KeyPoint) * numPts,
                      cudaMemcpyDeviceToHost));
  return numPts;
}

__global__ void ExtractDescriptors(cv::KeyPoint *d_pts, CudaImage *d_imgs,
                                   float *_vals, int size2, int size3,
                                   int size4) {
  __shared__ float acc_vals[3 * 30 * EXTRACT_S];

  float *acc_vals_im = &acc_vals[0];
  float *acc_vals_dx = &acc_vals[30 * EXTRACT_S];
  float *acc_vals_dy = &acc_vals[2 * 30 * EXTRACT_S];

  int p = blockIdx.x;

  float *vals = &_vals[p * 3 * 29];

  float iratio = 1.0f / (1 << d_pts[p].octave);
  int scale = (int)(0.5f * d_pts[p].size * iratio + 0.5f);
  float xf = d_pts[p].pt.x * iratio;
  float yf = d_pts[p].pt.y * iratio;
  float ang = d_pts[p].angle;
  float co = cos(ang);
  float si = sin(ang);
  int tx = threadIdx.x;
  int lev = d_pts[p].class_id;
  float *imd = d_imgs[4 * lev + 0].d_data;
  float *dxd = d_imgs[4 * lev + 2].d_data;
  float *dyd = d_imgs[4 * lev + 3].d_data;
  int pitch = d_imgs[4 * lev + 0].pitch;
  int winsize = max(3 * size3, 4 * size4);

  for (int i = 0; i < 30; ++i) {
    acc_vals_im[i * EXTRACT_S + tx] = 0.f;
    acc_vals_dx[i * EXTRACT_S + tx] = 0.f;
    acc_vals_dy[i * EXTRACT_S + tx] = 0.f;
  }

  __syncthreads();

  for (int i = tx; i < winsize * winsize; i += EXTRACT_S) {
    int y = i / winsize;
    int x = i - winsize * y;
    int m = max(x, y);
    if (m >= winsize) continue;
    int l = x - size2;
    int k = y - size2;
    int xp = (int)(xf + scale * (k * co - l * si) + 0.5f);
    int yp = (int)(yf + scale * (k * si + l * co) + 0.5f);
    int pos = yp * pitch + xp;
    float im = imd[pos];
    float dx = dxd[pos];
    float dy = dyd[pos];
    float rx = -dx * si + dy * co;
    float ry = dx * co + dy * si;

    if (m < 2 * size2) {
      int x2 = (x < size2 ? 0 : 1);
      int y2 = (y < size2 ? 0 : 1);
      // atomicAdd(norm2, (x < size2 && y < size2 ? 1 : 0));
      // Add 2x2
      acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tx] += im;
      acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tx + 1] += rx;
      acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tx + 2] += ry;
    }
    if (m < 3 * size3) {
      int x3 = (x < size3 ? 0 : (x < 2 * size3 ? 1 : 2));
      int y3 = (y < size3 ? 0 : (y < 2 * size3 ? 1 : 2));
      // atomicAdd(norm3, (x < size3 && y < size3 ? 1 : 0));
      // Add 3x3
      acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tx] += im;
      acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tx + 1] += rx;
      acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tx + 2] += ry;
    }
    if (m < 4 * size4) {
      int x4 = (x < 2 * size4 ? (x < size4 ? 0 : 1) : (x < 3 * size4 ? 2 : 3));
      int y4 = (y < 2 * size4 ? (y < size4 ? 0 : 1) : (y < 3 * size4 ? 2 : 3));
      // atomicAdd(norm4, (x < size4 && y < size4 ? 1 : 0));
      // Add 4x4
      acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tx] += im;
      acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tx + 1] += rx;
      acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tx + 2] += ry;
    }
  }

  __syncthreads();

// Reduce stuff
#pragma unroll
  float acc_reg;
  for (int i = 0; i < 15; ++i) {
    // 0..31 takes care of even accs, 32..63 takes care of odd accs
    int offset = 2 * i + (tx < 32 ? 0 : 1);
    int tx_d = tx < 32 ? tx : tx - 32;
    for (int d = 0; d < 90; d += 30) {
      if (tx_d < 32) {
        acc_reg = acc_vals[3 * 30 * tx_d + offset + d] +
                  acc_vals[3 * 30 * (tx_d + 32) + offset + d];
        acc_reg += __shfl_down(acc_reg, 1);
        acc_reg += __shfl_down(acc_reg, 2);
        acc_reg += __shfl_down(acc_reg, 4);
        acc_reg += __shfl_down(acc_reg, 8);
        acc_reg += __shfl_down(acc_reg, 16);
      }
      if (tx_d == 0) {
        acc_vals[offset + d] = acc_reg;
      }
    }
  }

  __syncthreads();

  // Have 29*3 values to store
  // They are in acc_vals[0..28,64*30..64*30+28,64*60..64*60+28]
  if (tx < 29) {
    vals[tx] = acc_vals[tx];
    vals[29 + tx] = acc_vals[29 + tx];
    vals[2 * 29 + tx] = acc_vals[2 * 29 + tx];
  }
}

__global__ void ExtractDescriptors_serial(cv::KeyPoint *d_pts,
                                          CudaImage *d_imgs, float *_vals,
                                          int size2, int size3, int size4) {
  __shared__ float acc_vals[30 * EXTRACT_S];
  __shared__ float final_vals[3 * 30];

  int p = blockIdx.x;

  float *vals = &_vals[p * 3 * 29];

  float iratio = 1.0f / (1 << d_pts[p].octave);
  int scale = (int)(0.5f * d_pts[p].size * iratio + 0.5f);
  float xf = d_pts[p].pt.x * iratio;
  float yf = d_pts[p].pt.y * iratio;
  float ang = d_pts[p].angle;
  float co = cos(ang);
  float si = sin(ang);
  int tx = threadIdx.x;
  int lev = d_pts[p].class_id;
  float *imd = d_imgs[4 * lev + 0].d_data;
  float *dxd = d_imgs[4 * lev + 2].d_data;
  float *dyd = d_imgs[4 * lev + 3].d_data;
  int pitch = d_imgs[4 * lev + 0].pitch;
  int winsize = max(3 * size3, 4 * size4);

  // IM
  for (int i = 0; i < 30; ++i) {
    acc_vals[i * EXTRACT_S + tx] = 0.f;
  }

  __syncthreads();

  for (int i = tx; i < winsize * winsize; i += EXTRACT_S) {
    int y = i / winsize;
    int x = i - winsize * y;
    int m = max(x, y);
    if (m >= winsize) continue;
    int l = x - size2;
    int k = y - size2;
    int xp = (int)(xf + scale * (k * co - l * si) + 0.5f);
    int yp = (int)(yf + scale * (k * si + l * co) + 0.5f);
    int pos = yp * pitch + xp;
    float im = imd[pos];
    if (m < 2 * size2) {
      int x2 = (x < size2 ? 0 : 1);
      int y2 = (y < size2 ? 0 : 1);
      // atomicAdd(norm2, (x < size2 && y < size2 ? 1 : 0));
      // Add 2x2
      acc_vals[(y2 * 2 + x2) + 30 * tx] += im;
    }
    if (m < 3 * size3) {
      int x3 = (x < size3 ? 0 : (x < 2 * size3 ? 1 : 2));
      int y3 = (y < size3 ? 0 : (y < 2 * size3 ? 1 : 2));
      // atomicAdd(norm3, (x < size3 && y < size3 ? 1 : 0));
      // Add 3x3
      acc_vals[(4 + y3 * 3 + x3) + 30 * tx] += im;
    }
    if (m < 4 * size4) {
      int x4 = (x < 2 * size4 ? (x < size4 ? 0 : 1) : (x < 3 * size4 ? 2 : 3));
      int y4 = (y < 2 * size4 ? (y < size4 ? 0 : 1) : (y < 3 * size4 ? 2 : 3));
      // atomicAdd(norm4, (x < size4 && y < size4 ? 1 : 0));
      // Add 4x4
      acc_vals[(4 + 9 + y4 * 4 + x4) + 30 * tx] += im;
    }
  }

  __syncthreads();

// Reduce stuff
#pragma unroll
  for (int i = 0; i < 15; ++i) {
    // 0..31 takes care of even accs, 32..63 takes care of odd accs
    int offset = 2 * i + (tx < 32 ? 0 : 1);
    int tx_d = tx < 32 ? tx : tx - 32;
    int acc_idx = 30 * tx_d + offset;
    if (tx_d < 32) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 32];
    }
    if (tx_d < 16) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 16];
    }
    if (tx_d < 8) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 8];
    }
    if (tx_d < 4) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 4];
    }
    if (tx_d < 2) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 2];
    }
    if (tx_d < 1) {
      final_vals[3 * offset] = acc_vals[acc_idx] + acc_vals[offset + 30];
    }
  }

  // DX
  for (int i = 0; i < 30; ++i) {
    acc_vals[i * EXTRACT_S + tx] = 0.f;
  }

  __syncthreads();

  for (int i = tx; i < winsize * winsize; i += EXTRACT_S) {
    int y = i / winsize;
    int x = i - winsize * y;
    int m = max(x, y);
    if (m >= winsize) continue;
    int l = x - size2;
    int k = y - size2;
    int xp = (int)(xf + scale * (k * co - l * si) + 0.5f);
    int yp = (int)(yf + scale * (k * si + l * co) + 0.5f);
    int pos = yp * pitch + xp;
    float dx = dxd[pos];
    float dy = dyd[pos];
    float rx = -dx * si + dy * co;
    if (m < 2 * size2) {
      int x2 = (x < size2 ? 0 : 1);
      int y2 = (y < size2 ? 0 : 1);
      // atomicAdd(norm2, (x < size2 && y < size2 ? 1 : 0));
      // Add 2x2
      acc_vals[(y2 * 2 + x2) + 30 * tx] += rx;
    }
    if (m < 3 * size3) {
      int x3 = (x < size3 ? 0 : (x < 2 * size3 ? 1 : 2));
      int y3 = (y < size3 ? 0 : (y < 2 * size3 ? 1 : 2));
      // atomicAdd(norm3, (x < size3 && y < size3 ? 1 : 0));
      // Add 3x3
      acc_vals[(4 + y3 * 3 + x3) + 30 * tx] += rx;
    }
    if (m < 4 * size4) {
      int x4 = (x < 2 * size4 ? (x < size4 ? 0 : 1) : (x < 3 * size4 ? 2 : 3));
      int y4 = (y < 2 * size4 ? (y < size4 ? 0 : 1) : (y < 3 * size4 ? 2 : 3));
      // atomicAdd(norm4, (x < size4 && y < size4 ? 1 : 0));
      // Add 4x4
      acc_vals[(4 + 9 + y4 * 4 + x4) + 30 * tx] += rx;
    }
  }

  __syncthreads();

// Reduce stuff
#pragma unroll
  for (int i = 0; i < 15; ++i) {
    // 0..31 takes care of even accs, 32..63 takes care of odd accs
    int offset = 2 * i + (tx < 32 ? 0 : 1);
    int tx_d = tx < 32 ? tx : tx - 32;
    int acc_idx = 30 * tx_d + offset;
    if (tx_d < 32) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 32];
    }
    if (tx_d < 16) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 16];
    }
    if (tx_d < 8) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 8];
    }
    if (tx_d < 4) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 4];
    }
    if (tx_d < 2) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 2];
    }
    if (tx_d < 1) {
      final_vals[3 * offset] = acc_vals[acc_idx] + acc_vals[offset + 30];
    }
  }

  // DY
  for (int i = 0; i < 30; ++i) {
    acc_vals[i * EXTRACT_S + tx] = 0.f;
  }

  __syncthreads();

  for (int i = tx; i < winsize * winsize; i += EXTRACT_S) {
    int y = i / winsize;
    int x = i - winsize * y;
    int m = max(x, y);
    if (m >= winsize) continue;
    int l = x - size2;
    int k = y - size2;
    int xp = (int)(xf + scale * (k * co - l * si) + 0.5f);
    int yp = (int)(yf + scale * (k * si + l * co) + 0.5f);
    int pos = yp * pitch + xp;
    float dx = dxd[pos];
    float dy = dyd[pos];
    float ry = dx * co + dy * si;
    if (m < 2 * size2) {
      int x2 = (x < size2 ? 0 : 1);
      int y2 = (y < size2 ? 0 : 1);
      // atomicAdd(norm2, (x < size2 && y < size2 ? 1 : 0));
      // Add 2x2
      acc_vals[(y2 * 2 + x2) + 30 * tx] += ry;
    }
    if (m < 3 * size3) {
      int x3 = (x < size3 ? 0 : (x < 2 * size3 ? 1 : 2));
      int y3 = (y < size3 ? 0 : (y < 2 * size3 ? 1 : 2));
      // atomicAdd(norm3, (x < size3 && y < size3 ? 1 : 0));
      // Add 3x3
      acc_vals[(4 + y3 * 3 + x3) + 30 * tx] += ry;
    }
    if (m < 4 * size4) {
      int x4 = (x < 2 * size4 ? (x < size4 ? 0 : 1) : (x < 3 * size4 ? 2 : 3));
      int y4 = (y < 2 * size4 ? (y < size4 ? 0 : 1) : (y < 3 * size4 ? 2 : 3));
      // atomicAdd(norm4, (x < size4 && y < size4 ? 1 : 0));
      // Add 4x4
      acc_vals[(4 + 9 + y4 * 4 + x4) + 30 * tx] += ry;
    }
  }

  __syncthreads();

// Reduce stuff
#pragma unroll
  for (int i = 0; i < 15; ++i) {
    // 0..31 takes care of even accs, 32..63 takes care of odd accs
    int offset = 2 * i + (tx < 32 ? 0 : 1);
    int tx_d = tx < 32 ? tx : tx - 32;
    int acc_idx = 30 * tx_d + offset;
    if (tx_d < 32) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 32];
    }
    if (tx_d < 16) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 16];
    }
    if (tx_d < 8) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 8];
    }
    if (tx_d < 4) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 4];
    }
    if (tx_d < 2) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 2];
    }
    if (tx_d < 1) {
      final_vals[3 * offset] = acc_vals[acc_idx] + acc_vals[offset + 30];
    }
  }

  __syncthreads();

  // Have 29*3 values to store
  // They are in acc_vals[0..28,64*30..64*30+28,64*60..64*60+28]
  if (tx < 29) {
    vals[tx] = final_vals[tx];
    vals[29 + tx] = final_vals[29 + tx];
    vals[2 * 29 + tx] = final_vals[2 * 29 + tx];
  }
}

__global__ void ExtractDescriptors_atomic_shared(cv::KeyPoint *d_pts,
                                                 CudaImage *d_imgs,
                                                 float *_vals, int size2,
                                                 int size3, int size4) {
  __shared__ float acc_vals[3 * 30 * EXTRACT_S / 2];

  float *acc_vals_im = &acc_vals[0];
  float *acc_vals_dx = &acc_vals[30 * EXTRACT_S / 2];
  float *acc_vals_dy = &acc_vals[2 * 30 * EXTRACT_S / 2];

  int p = blockIdx.x;

  float *vals = &_vals[p * 3 * 29];

  float iratio = 1.0f / (1 << d_pts[p].octave);
  int scale = (int)(0.5f * d_pts[p].size * iratio + 0.5f);
  float xf = d_pts[p].pt.x * iratio;
  float yf = d_pts[p].pt.y * iratio;
  float ang = d_pts[p].angle;
  float co = cos(ang);
  float si = sin(ang);
  int tx = threadIdx.x;
  int lev = d_pts[p].class_id;
  float *imd = d_imgs[4 * lev + 0].d_data;
  float *dxd = d_imgs[4 * lev + 2].d_data;
  float *dyd = d_imgs[4 * lev + 3].d_data;
  int pitch = d_imgs[4 * lev + 0].pitch;
  int winsize = max(3 * size3, 4 * size4);

  for (int i = 0; i < 15; ++i) {
    acc_vals_im[i * EXTRACT_S + tx] = 0.f;
    acc_vals_dx[i * EXTRACT_S + tx] = 0.f;
    acc_vals_dy[i * EXTRACT_S + tx] = 0.f;
  }

  __syncthreads();

  for (int i = tx; i < winsize * winsize; i += EXTRACT_S) {
    int y = i / winsize;
    int x = i - winsize * y;
    int m = max(x, y);
    if (m >= winsize) continue;
    int l = x - size2;
    int k = y - size2;
    int xp = (int)(xf + scale * (k * co - l * si) + 0.5f);
    int yp = (int)(yf + scale * (k * si + l * co) + 0.5f);
    int pos = yp * pitch + xp;
    float im = imd[pos];
    float dx = dxd[pos];
    float dy = dyd[pos];
    float rx = -dx * si + dy * co;
    float ry = dx * co + dy * si;

    int tx_d = tx & (EXTRACT_S >> 1 - 1);
    if (m < 2 * size2) {
      int x2 = (x < size2 ? 0 : 1);
      int y2 = (y < size2 ? 0 : 1);
      // atomicAdd(norm2, (x < size2 && y < size2 ? 1 : 0));
      // Add 2x2
      atomicAdd(&acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tx_d], im);
      atomicAdd(&acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tx_d + 1], rx);
      atomicAdd(&acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tx_d + 2], ry);
    }
    if (m < 3 * size3) {
      int x3 = (x < size3 ? 0 : (x < 2 * size3 ? 1 : 2));
      int y3 = (y < size3 ? 0 : (y < 2 * size3 ? 1 : 2));
      // atomicAdd(norm3, (x < size3 && y < size3 ? 1 : 0));
      // Add 3x3
      atomicAdd(&acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tx_d], im);
      atomicAdd(&acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tx_d + 1], rx);
      atomicAdd(&acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tx_d + 2], ry);
    }
    if (m < 4 * size4) {
      int x4 = (x < 2 * size4 ? (x < size4 ? 0 : 1) : (x < 3 * size4 ? 2 : 3));
      int y4 = (y < 2 * size4 ? (y < size4 ? 0 : 1) : (y < 3 * size4 ? 2 : 3));
      // atomicAdd(norm4, (x < size4 && y < size4 ? 1 : 0));
      // Add 4x4
      atomicAdd(&acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tx_d], im);
      atomicAdd(&acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tx_d + 1], rx);
      atomicAdd(&acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tx_d + 2], ry);
    }
  }

  __syncthreads();

// Reduce stuff
#pragma unroll
  for (int i = 0; i < 15; ++i) {
    // 0..31 takes care of even accs, 32..63 takes care of odd accs
    int offset = 2 * i + (tx < 32 ? 0 : 1);
    int tx_d = tx < 32 ? tx : tx - 32;
    /*if (tx_d < 32) {
      acc_vals[3*30 * tx_d/2 + offset] += acc_vals[3*30/2 * (tx_d + 32) +
    offset];
      acc_vals[3*30 * tx_d/2 + offset+30] += acc_vals[3*30/2 * (tx_d + 32) +
    offset+30];
      acc_vals[3*30 * tx_d/2 + offset+60] += acc_vals[3*30/2 * (tx_d + 32) +
    offset+60];
    }*/
    if (tx_d < 16) {
      acc_vals[3 * 30 * tx_d + offset] +=
          acc_vals[3 * 30 * (tx_d + 16) + offset];
      acc_vals[3 * 30 * tx_d + offset + 30] +=
          acc_vals[3 * 30 * (tx_d + 16) + offset + 30];
      acc_vals[3 * 30 * tx_d + offset + 60] +=
          acc_vals[3 * 30 * (tx_d + 16) + offset + 60];
    }
    if (tx_d < 8) {
      acc_vals[3 * 30 * tx_d + offset] +=
          acc_vals[3 * 30 * (tx_d + 8) + offset];
      acc_vals[3 * 30 * tx_d + offset + 30] +=
          acc_vals[3 * 30 * (tx_d + 8) + offset + 30];
      acc_vals[3 * 30 * tx_d + offset + 60] +=
          acc_vals[3 * 30 * (tx_d + 8) + offset + 60];
    }
    if (tx_d < 4) {
      acc_vals[3 * 30 * tx_d + offset] +=
          acc_vals[3 * 30 * (tx_d + 4) + offset];
      acc_vals[3 * 30 * tx_d + offset + 30] +=
          acc_vals[3 * 30 * (tx_d + 4) + offset + 30];
      acc_vals[3 * 30 * tx_d + offset + 60] +=
          acc_vals[3 * 30 * (tx_d + 4) + offset + 60];
    }
    if (tx_d < 2) {
      acc_vals[3 * 30 * tx_d + offset] +=
          acc_vals[3 * 30 * (tx_d + 2) + offset];
      acc_vals[3 * 30 * tx_d + offset + 30] +=
          acc_vals[3 * 30 * (tx_d + 2) + offset + 30];
      acc_vals[3 * 30 * tx_d + offset + 60] +=
          acc_vals[3 * 30 * (tx_d + 2) + offset + 60];
    }
    if (tx_d < 1) {
      acc_vals[offset] += acc_vals[3 * 30 + offset];
      acc_vals[offset + 30] += acc_vals[3 * 30 + offset + 30];
      acc_vals[offset + 60] += acc_vals[3 * 30 + offset + 60];
    }
  }

  __syncthreads();

  // Have 29*3 values to store
  // They are in acc_vals[0..28,64*30..64*30+28,64*60..64*60+28]
  if (tx < 29) {
    vals[tx] = acc_vals[tx];
    vals[29 + tx] = acc_vals[29 + tx];
    vals[2 * 29 + tx] = acc_vals[2 * 29 + tx];
  }
}

__global__ void ExtractDescriptors_atomic(cv::KeyPoint *d_pts,
                                          CudaImage *d_imgs, float *_vals,
                                          int size2, int size3, int size4) {
  __shared__ float acc_vals[3 * 29];

  float *acc_vals_im = &acc_vals[0];
  float *acc_vals_dx = &acc_vals[29];
  float *acc_vals_dy = &acc_vals[2 * 29];

  int p = blockIdx.x;

  float *vals = &_vals[p * 3 * 29];

  float iratio = 1.0f / (1 << d_pts[p].octave);
  int scale = (int)(0.5f * d_pts[p].size * iratio + 0.5f);
  float xf = d_pts[p].pt.x * iratio;
  float yf = d_pts[p].pt.y * iratio;
  float ang = d_pts[p].angle;
  float co = cos(ang);
  float si = sin(ang);
  int tx = threadIdx.x;
  int lev = d_pts[p].class_id;
  float *imd = d_imgs[4 * lev + 0].d_data;
  float *dxd = d_imgs[4 * lev + 2].d_data;
  float *dyd = d_imgs[4 * lev + 3].d_data;
  int pitch = d_imgs[4 * lev + 0].pitch;
  int winsize = max(3 * size3, 4 * size4);

  if (tx < 29) {
    acc_vals_im[tx] = 0.f;
    acc_vals_dx[tx] = 0.f;
    acc_vals_dy[tx] = 0.f;
  }

  __syncthreads();

  for (int i = tx; i < winsize * winsize; i += EXTRACT_S) {
    int y = i / winsize;
    int x = i - winsize * y;
    int m = max(x, y);
    if (m >= winsize) continue;
    int l = x - size2;
    int k = y - size2;
    int xp = (int)(xf + scale * (k * co - l * si) + 0.5f);
    int yp = (int)(yf + scale * (k * si + l * co) + 0.5f);
    int pos = yp * pitch + xp;
    float im = imd[pos];
    float dx = dxd[pos];
    float dy = dyd[pos];
    float rx = -dx * si + dy * co;
    float ry = dx * co + dy * si;

    if (m < 2 * size2) {
      int x2 = (x < size2 ? 0 : 1);
      int y2 = (y < size2 ? 0 : 1);
      // atomicAdd(norm2, (x < size2 && y < size2 ? 1 : 0));
      // Add 2x2
      atomicAdd(&acc_vals[3 * (y2 * 2 + x2)], im);
      atomicAdd(&acc_vals[3 * (y2 * 2 + x2) + 1], rx);
      atomicAdd(&acc_vals[3 * (y2 * 2 + x2) + 2], ry);
    }
    if (m < 3 * size3) {
      int x3 = (x < size3 ? 0 : (x < 2 * size3 ? 1 : 2));
      int y3 = (y < size3 ? 0 : (y < 2 * size3 ? 1 : 2));
      // atomicAdd(norm3, (x < size3 && y < size3 ? 1 : 0));
      // Add 3x3
      atomicAdd(&acc_vals[3 * (4 + y3 * 3 + x3)], im);
      atomicAdd(&acc_vals[3 * (4 + y3 * 3 + x3) + 1], rx);
      atomicAdd(&acc_vals[3 * (4 + y3 * 3 + x3) + 2], ry);
    }
    if (m < 4 * size4) {
      int x4 = (x < 2 * size4 ? (x < size4 ? 0 : 1) : (x < 3 * size4 ? 2 : 3));
      int y4 = (y < 2 * size4 ? (y < size4 ? 0 : 1) : (y < 3 * size4 ? 2 : 3));
      // atomicAdd(norm4, (x < size4 && y < size4 ? 1 : 0));
      // Add 4x4
      atomicAdd(&acc_vals[3 * (4 + 9 + y4 * 4 + x4)], im);
      atomicAdd(&acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 1], rx);
      atomicAdd(&acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 2], ry);
    }
  }

  __syncthreads();

  // Have 29*3 values to store
  // They are in acc_vals[0..28,64*30..64*30+28,64*60..64*60+28]
  if (tx < 29) {
    vals[tx] = acc_vals[tx];
    vals[29 + tx] = acc_vals[29 + tx];
    vals[2 * 29 + tx] = acc_vals[2 * 29 + tx];
  }
}

__global__ void BuildDescriptor(float *_valsim, unsigned char *_desc) {
  int p = blockIdx.x;

  size_t idx = threadIdx.x;

  if (idx < 61) {
    float *valsim = &_valsim[3 * 29 * p];

    unsigned char *desc = &_desc[61 * p];

    unsigned char desc_r = 0;

#pragma unroll
    for (int i = 0; i < (idx == 60 ? 6 : 8); ++i) {
      int idx1 = comp_idx_1[idx * 8 + i];
      int idx2 = comp_idx_2[idx * 8 + i];
      desc_r |= (valsim[idx1] > valsim[idx2] ? 1 : 0) << i;
    }

    desc[idx] = desc_r;
  }
}

#define NTHREADS_MATCH 32
__global__ void MatchDescriptors(unsigned char *d1, unsigned char *d2,
                                 int pitch, int nkpts_2, cv::DMatch *matches) {
  int p = blockIdx.x;

  int x = threadIdx.x;

  __shared__ int idxBest[NTHREADS_MATCH];
  __shared__ int idxSecondBest[NTHREADS_MATCH];
  __shared__ int scoreBest[NTHREADS_MATCH];
  __shared__ int scoreSecondBest[NTHREADS_MATCH];

  idxBest[x] = 0;
  idxSecondBest[x] = 0;
  scoreBest[x] = 512;
  scoreSecondBest[x] = 512;

  __syncthreads();

  // working single threaded version, the problem was popcll
  //  int idxBest = 1;
  //  int idxSecondBest = 0;
  //  int scoreBest = 512;
  //  int scoreSecondBest = 512;

  //  for (int i = 0; i < nkpts_2; ++i) {
  //    int score = 0;
  //    for (int j = 0; j < 64; ++j) {
  //      score += __popc(d1[pitch * p + j] ^ d2[pitch * i + j]);
  //    }

  //    if (score < scoreBest) {
  //      scoreSecondBest = scoreBest;
  //      scoreBest = score;
  //      idxSecondBest = idxBest;
  //      idxBest = i;

  //    } else if (score < scoreSecondBest) {
  //      scoreSecondBest = score;
  //      idxSecondBest = i;
  //    }
  //  }

  //    int idxBest = 1;
  //    int idxSecondBest = 0;
  //    int scoreBest = 512;
  //    int scoreSecondBest = 512;

  //    for (int i=0; i<nkpts_2; i+=32) {
  //        if( i+x < nkpts_2) {
  //            // Check d1[p] with d2[i]
  //            int score = 0;
  //            for(int j=0; j<16; ++j) {
  //                score += __popcll(d1[pitch*p+4*j] ^ d2[pitch*(i+x)+4*j]);
  //            }
  //            if( score < scoreBest ) {
  //                scoreSecondBest = scoreBest;
  //                scoreBest = score;
  //                idxSecondBest = idxBest;
  //                idxBest = i+x;
  //            } else if( score < scoreSecondBest ) {
  //                scoreSecondBest = score;
  //                idxSecondBest = i+x;
  //            }
  //        }
  //    }

  // curent version fixed with popc, still not convinced
  unsigned long long *d1i = (unsigned long long*)(d1 + pitch*p);

  for (int i = 0; i < nkpts_2; i += NTHREADS_MATCH) {
    unsigned long long *d2i = (unsigned long long*)(d2 + pitch*(x+i));
    if (i + x < nkpts_2) {
      // Check d1[p] with d2[i]
      int score = 0;
      for (int j = 0; j < 8; ++j) {
        score += __popcll(d1i[j] ^ d2i[j]);
      }
      if (score < scoreBest[x]) {
        scoreSecondBest[x] = scoreBest[x];
        scoreBest[x] = score;
        idxSecondBest[x] = idxBest[x];
        idxBest[x] = i + x;
      } else if (score < scoreSecondBest[x]) {
        scoreSecondBest[x] = score;
        idxSecondBest[x] = i + x;
      }
    }
  }

  //    for( int i=16; i>=1; i/=2) {
  //        int tBest = __shfl_down(scoreBest,i);
  //        int tIdx = __shfl_down(idxBest,i);
  //        if(tBest < scoreBest) {
  //            scoreSecondBest = scoreBest;
  //            idxSecondBest = idxBest;
  //            scoreBest = tBest;
  //            idxBest = tIdx;
  //        }
  //        tBest = __shfl_down(scoreSecondBest,i);
  //        tIdx = __shfl_down(idxSecondBest,i);
  //        if(tBest < scoreSecondBest) {
  //            scoreSecondBest = tBest;
  //            idxSecondBest = tIdx;
  //        }
  //    }

  __syncthreads();

  for (int i = NTHREADS_MATCH / 2; i >= 1; i /= 2) {
    if (x < i) {
      if (scoreBest[x + i] < scoreBest[x]) {
        scoreSecondBest[x] = scoreBest[x];
        scoreBest[x] = scoreBest[x + i];
        idxSecondBest[x] = idxBest[x];
        idxBest[x] = idxBest[x + i];
      } else if (scoreBest[x + i] < scoreSecondBest[x]) {
        scoreSecondBest[x] = scoreBest[x + i];
        idxSecondBest[x] = idxBest[x + i];
      }
      if (scoreSecondBest[x + i] < scoreSecondBest[x]) {
        scoreSecondBest[x] = scoreSecondBest[x + i];
        idxSecondBest[x] = idxSecondBest[x + i];
      }
    }
    }
    // if(i>16) __syncthreads();
    //        if(x<i) {
    //            if( scoreBest[x+i] < scoreSecondBest[x] ) {
    //                scoreSecondBest[x] = scoreBest[x+i];
    //                idxSecondBest[x] = idxBest[x+i];
    //            } else if (scoreSecondBest[x+i] < scoreSecondBest[x] ) {
    //                scoreSecondBest[x] = scoreSecondBest[x+i];
    //                idxSecondBest[x] = idxSecondBest[x+i];
    //            }
    //        }
    //        if(i>16) __syncthreads();
    //}

    /*for (int i = 1; i <= NTHREADS_MATCH; ++i) {
      if (scoreBest[i] < scoreBest[0]) {
        scoreSecondBest[0] = scoreBest[0];
        scoreBest[0] = scoreBest[i];
        idxSecondBest[0] = idxBest[0];
        idxBest[0] = idxBest[i];
      }  else if( scoreBest[i] < scoreSecondBest[0] ) {
           scoreSecondBest[0] = scoreBest[i];
           idxSecondBest[0] = idxBest[i];
       }
       if(scoreSecondBest[i] < scoreSecondBest[0]) {
           scoreSecondBest[0] = scoreSecondBest[i];
           idxSecondBest[0] = idxSecondBest[i];
       }
    }*/

    //    if(x==0) {
    //        matches[2*p].queryIdx = p;
    //        matches[2*p].trainIdx = idxBest;
    //        matches[2*p].distance = scoreBest;
    //        matches[2*p+1].queryIdx = p;
    //        matches[2*p+1].trainIdx = idxSecondBest;
    //        matches[2*p+1].distance = scoreSecondBest;
    //    }

    if (x == 0) {
      matches[2 * p].queryIdx = p;
      matches[2 * p].trainIdx = idxBest[x];
      matches[2 * p].distance = scoreBest[x];
      matches[2 * p + 1].queryIdx = p;
      matches[2 * p + 1].trainIdx = idxSecondBest[x];
      matches[2 * p + 1].distance = scoreSecondBest[x];
    }
  }

  void MatchDescriptors(cv::Mat & desc_query, cv::Mat & desc_train,
                        std::vector<std::vector<cv::DMatch> > & dmatches) {
    size_t pitch1, pitch2;
    unsigned char *descq_d;
    cudaMallocPitch(&descq_d, &pitch1, 64, desc_query.rows);
    cudaMemset2D(descq_d, pitch1, 0, 64, desc_query.rows);
    cudaMemcpy2D(descq_d, pitch1, desc_query.data, desc_query.cols,
                 desc_query.cols, desc_query.rows, cudaMemcpyHostToDevice);
    unsigned char *desct_d;
    cudaMallocPitch(&desct_d, &pitch2, 64, desc_train.rows);
    cudaMemset2D(desct_d, pitch2, 0, 64, desc_train.rows);
    cudaMemcpy2D(desct_d, pitch2, desc_train.data, desc_train.cols,
                 desc_train.cols, desc_train.rows, cudaMemcpyHostToDevice);

    dim3 block(desc_query.rows);

    cv::DMatch *dmatches_d;
    cudaMalloc(&dmatches_d, desc_query.rows * 2 * sizeof(cv::DMatch));

    MatchDescriptors << <block, NTHREADS_MATCH>>>
        (descq_d, desct_d, pitch1, desc_train.rows, dmatches_d);

    cv::DMatch *dmatches_h = new cv::DMatch[2 * desc_query.rows];
    cudaMemcpy(dmatches_h, dmatches_d, desc_query.rows * 2 * sizeof(cv::DMatch),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < desc_query.rows; ++i) {
      std::vector<cv::DMatch> tdmatch;
      tdmatch.push_back(dmatches_h[2 * i]);
      tdmatch.push_back(dmatches_h[2 * i + 1]);
      dmatches.push_back(tdmatch);
    }

    cudaFree(descq_d);
    cudaFree(desct_d);
    cudaFree(dmatches_d);

    delete[] dmatches_h;
  }

  double ExtractDescriptors(std::vector<cv::KeyPoint> & h_pts,
                            cv::KeyPoint * d_pts,
                            std::vector<CudaImage> & h_imgs, CudaImage * d_imgs,
                            unsigned char *desc_h, int patsize) {
    int size2 = patsize;
    int size3 = ceil(2.0f * patsize / 3.0f);
    int size4 = ceil(0.5f * patsize);
    int numPts = h_pts.size();

    // TimerGPU timer0(0);
    dim3 blocks(numPts);
    dim3 threads(EXTRACT_S);

    float *vals_h = new float[3 * 29 * numPts];
    float *vals_d;
    cudaMalloc(&vals_d, 3 * 29 * numPts * sizeof(float));

    ExtractDescriptors << <blocks, threads>>>
        (d_pts, d_imgs, vals_d, size2, size3, size4);

//#define PRINT_KPT

#ifdef PRINT_KPT
    cudaMemcpy(vals_h, vals_d, 3 * 29 * numPts * sizeof(float),
               cudaMemcpyDeviceToHost);

    //  int xkpt = 865, ykpt = 30;
    int xkpt = 840, ykpt = 45;

    int idx = -1;
    for (int i = 0; i < h_pts.size(); ++i) {
      if ((int)h_pts[i].pt.x == xkpt && (int)h_pts[i].pt.y == ykpt) idx = i;
    }
#endif

    static int comp_idx_1_h[61 * 8];
    static int comp_idx_2_h[61 * 8];

    int cntr = 0;
    for (int j = 0; j < 4; ++j) {
      for (int i = j + 1; i < 4; ++i) {
        comp_idx_1_h[cntr] = 3 * j;
        comp_idx_2_h[cntr] = 3 * i;
        cntr++;
      }
    }
    for (int j = 0; j < 3; ++j) {
      for (int i = j + 1; i < 4; ++i) {
        comp_idx_1_h[cntr] = 3 * j + 1;
        comp_idx_2_h[cntr] = 3 * i + 1;
        cntr++;
      }
    }
    for (int j = 0; j < 3; ++j) {
      for (int i = j + 1; i < 4; ++i) {
        comp_idx_1_h[cntr] = 3 * j + 2;
        comp_idx_2_h[cntr] = 3 * i + 2;
        cntr++;
      }
    }

    // 3x3
    for (int j = 4; j < 12; ++j) {
      for (int i = j + 1; i < 13; ++i) {
        comp_idx_1_h[cntr] = 3 * j;
        comp_idx_2_h[cntr] = 3 * i;
        cntr++;
      }
    }
    for (int j = 4; j < 12; ++j) {
      for (int i = j + 1; i < 13; ++i) {
        comp_idx_1_h[cntr] = 3 * j + 1;
        comp_idx_2_h[cntr] = 3 * i + 1;
        cntr++;
      }
    }
    for (int j = 4; j < 12; ++j) {
      for (int i = j + 1; i < 13; ++i) {
        comp_idx_1_h[cntr] = 3 * j + 2;
        comp_idx_2_h[cntr] = 3 * i + 2;
        cntr++;
      }
    }

    // 4x4
    for (int j = 13; j < 28; ++j) {
      for (int i = j + 1; i < 29; ++i) {
        comp_idx_1_h[cntr] = 3 * j;
        comp_idx_2_h[cntr] = 3 * i;
        cntr++;
      }
    }
    for (int j = 13; j < 28; ++j) {
      for (int i = j + 1; i < 29; ++i) {
        comp_idx_1_h[cntr] = 3 * j + 1;
        comp_idx_2_h[cntr] = 3 * i + 1;
        cntr++;
      }
    }
    for (int j = 13; j < 28; ++j) {
      for (int i = j + 1; i < 29; ++i) {
        comp_idx_1_h[cntr] = 3 * j + 2;
        comp_idx_2_h[cntr] = 3 * i + 2;
        cntr++;
      }
    }

    cudaMemcpyToSymbolAsync(comp_idx_1, comp_idx_1_h, 8 * 61 * sizeof(int));
    cudaMemcpyToSymbolAsync(comp_idx_2, comp_idx_2_h, 8 * 61 * sizeof(int));

    unsigned char *desc_d;
    cudaMalloc(&desc_d, numPts * 61);
    cudaMemsetAsync(desc_d, 0, numPts * 61);
    BuildDescriptor << <blocks, 64>>> (vals_d, desc_d);

    cudaMemcpy(desc_h, desc_d, 61 * numPts, cudaMemcpyDeviceToHost);
#ifdef PRINT_KPT
    if (idx == -1)
      std::cout << "GPU error:: keypoint not found! " << std::endl;
    else {
      float sum2 = 0, sum3 = 0, sum4 = 0;
      for (int i = 0; i < 4; ++i) {
        sum2 += vals_h[3 * 29 * idx + 3 * i];
      }
      for (int i = 0; i < 9; ++i) {
        sum3 += vals_h[3 * 29 * idx + 12 + 3 * i];
      }
      for (int i = 0; i < 16; ++i) {
        sum4 += vals_h[3 * 29 * idx + 39 + 3 * i];
      }

      std::cout << "sums: " << sum2 << " " << sum3 << " " << sum4 << std::endl;

      std::cout << "Keypoint idx: " << idx << std::endl;

      std::cout << "GPU output:\n";
      std::cout << "2x2\n";
      for (int i = 0; i < 12; ++i) {
        std::cout << vals_h[3 * 29 * idx + i] << " ";
      }
      std::cout << std::endl;
      std::cout << "3x3";
      std::cout << std::endl;
      for (int i = 12; i < 39; ++i) {
        std::cout << vals_h[3 * 29 * idx + i] << " ";
      }
      std::cout << std::endl;
      std::cout << "4x4";
      std::cout << std::endl;
      for (int i = 39; i < 3 * 29; ++i) {
        std::cout << vals_h[3 * 29 * idx + i] << " ";
      }
      std::cout << std::endl;
      std::cout << "descriptor\n";
      std::cout << std::endl;
      for (int i = 0; i < 61; ++i) {
        std::cout << (unsigned int)desc_h[idx * 61 + i] << " ";
      }
      std::cout << "\n";
    }
#endif

    ////checkMsg("ExtractDescriptors() execution failed\n");
    // safeCall(cudaThreadSynchronize());
    double gpuTime = 0;  // timer0.read();
#ifdef VERBOSE
    printf("ExtractDescriptors time =     %.2f ms\n", gpuTime);
#endif

    cudaFree(vals_d);
    cudaFree(desc_d);

    delete[] vals_h;

    return gpuTime;
  }

  __global__ void FindOrientation(cv::KeyPoint * d_pts, CudaImage * d_imgs) {
    __shared__ float resx[42], resy[42];
    __shared__ float re8x[42], re8y[42];
    int p = blockIdx.x;
    int tx = threadIdx.x;
    if (tx < 48) resx[tx] = resy[tx] = 0.0f;
    __syncthreads();
    int lev = d_pts[p].class_id;
    float *dxd = d_imgs[4 * lev + 2].d_data;
    float *dyd = d_imgs[4 * lev + 3].d_data;
    int pitch = d_imgs[4 * lev + 0].pitch;
    int octave = d_pts[p].octave;
    int step = (int)(0.5f * d_pts[p].size + 0.5f) >> octave;
    int x = (int)(d_pts[p].pt.x + 0.5f) >> octave;
    int y = (int)(d_pts[p].pt.y + 0.5f) >> octave;
    int i = (tx & 15) - 6;
    int j = (tx / 16) - 6;
    int r2 = i * i + j * j;
    if (r2 < 36) {
      float gweight = exp(-r2 / (2.5f * 2.5f * 2.0f));
      int pos = (y + step * j) * pitch + (x + step * i);
      float dx = gweight * dxd[pos];
      float dy = gweight * dyd[pos];
      float angle = atan2(dy, dx);
      int a = max(min((int)(angle * (21 / CV_PI)) + 21, 41), 0);
      atomicAdd(resx + a, dx);
      atomicAdd(resy + a, dy);
    }
    __syncthreads();
    if (tx < 42) {
      re8x[tx] = resx[tx];
      re8y[tx] = resy[tx];
      for (int k = tx + 1; k < tx + 7; k++) {
        re8x[tx] += resx[k < 42 ? k : k - 42];
        re8y[tx] += resy[k < 42 ? k : k - 42];
      }
    }
    __syncthreads();
    if (tx == 0) {
      float maxr = 0.0f;
      int maxk = 0;
      for (int k = 0; k < 42; k++) {
        float r = re8x[k] * re8x[k] + re8y[k] * re8y[k];
        if (r > maxr) {
          maxr = r;
          maxk = k;
        }
      }
      float angle = atan2(re8y[maxk], re8x[maxk]);
      d_pts[p].angle = (angle < 0.0f ? angle + 2.0f * CV_PI : angle);
      // printf("XXX %.2f %.2f %.2f\n", d_pts[p].pt.x, d_pts[p].pt.y,
      // d_pts[p].angle/CV_PI*180.0f);
    }
  }

  double FindOrientation(std::vector<cv::KeyPoint> & h_pts,
                         cv::KeyPoint * d_pts, std::vector<CudaImage> & h_imgs,
                         CudaImage * d_imgs) {
    safeCall(cudaMemcpyAsync(d_imgs, (float *)&h_imgs[0],
                             sizeof(CudaImage) * h_imgs.size(),
                             cudaMemcpyHostToDevice));
    int numPts = h_pts.size();
    // TimerGPU timer0(0);
    dim3 blocks(numPts);
    dim3 threads(ORIENT_S);
    FindOrientation << <blocks, threads>>> (d_pts, d_imgs);
    // checkMsg("FindOrientation() execution failed\n");
    // safeCall(cudaThreadSynchronize());
    double gpuTime = 0;  // timer0.read();
#ifdef VERBOSE
    printf("FindOrientation time =        %.2f ms\n", gpuTime);
#endif
    return gpuTime;
  }

#include "cuda_akaze.h"
#include "cudautils.h"

#define CONVROW_W     160
#define CONVCOL_W      32
#define CONVCOL_H      40
#define CONVCOL_S       8

#define SCHARR_W       32
#define SCHARR_H       16

#define NLDSTEP_W      32
#define NLDSTEP_H      13

__device__ __constant__ float d_Kernel[21];

template<int RADIUS>
__global__ void ConvRowGPU(float *d_Result, float *d_Data, int width, int pitch, int height)
{
  __shared__ float data[CONVROW_W + 2*RADIUS];
  const int tx = threadIdx.x;
  const int minx = blockIdx.x*CONVROW_W;
  const int maxx = min(minx + CONVROW_W, width);
  const int yptr = blockIdx.y*pitch;
  const int loadPos = minx + tx - RADIUS; 
  const int writePos = minx + tx;

  if (loadPos<0) 
    data[tx] = d_Data[yptr];
  else if (loadPos>=width) 
    data[tx] = d_Data[yptr + width-1];
  else
    data[tx] = d_Data[yptr + loadPos];
  __syncthreads();
  if (writePos<maxx && tx<CONVROW_W) {
    float sum = 0.0f;
    for (int i=0;i<=(2*RADIUS);i++) 
      sum += data[tx + i]*d_Kernel[i];
    d_Result[yptr + writePos] = sum;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Column convolution filter
///////////////////////////////////////////////////////////////////////////////
template<int RADIUS>
__global__ void ConvColGPU(float *d_Result, float *d_Data, int width, int pitch, int height)
{
  __shared__ float data[CONVCOL_W*(CONVCOL_H + 2*RADIUS)];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int miny = blockIdx.y*CONVCOL_H;
  const int maxy = min(miny + CONVCOL_H, height) - 1;
  const int totStart = miny - RADIUS;
  const int totEnd = maxy + RADIUS;
  const int colStart = blockIdx.x*CONVCOL_W + tx;
  const int colEnd = colStart + (height-1)*pitch;
  const int smemStep = CONVCOL_W*CONVCOL_S;
  const int gmemStep = pitch*CONVCOL_S;
 
  if (colStart<width) {
    int smemPos = ty*CONVCOL_W + tx;
    int gmemPos = colStart + (totStart + ty)*pitch;
    for (int y = totStart+ty;y<=totEnd;y+=blockDim.y){
      if (y<0) 
        data[smemPos] = d_Data[colStart];
      else if (y>=height) 
        data[smemPos] = d_Data[colEnd];
      else 
        data[smemPos] = d_Data[gmemPos];  
      smemPos += smemStep;
      gmemPos += gmemStep;
    }
  }
  __syncthreads();
  if (colStart<width) {
    int smemPos = ty*CONVCOL_W + tx;
    int gmemPos = colStart + (miny + ty)*pitch;
    for (int y=miny+ty;y<=maxy;y+=blockDim.y) {
      float sum = 0.0f;
      for (int i=0;i<=2*RADIUS;i++)
        sum += data[smemPos + i*CONVCOL_W]*d_Kernel[i];
      d_Result[gmemPos] = sum;
      smemPos += smemStep;
      gmemPos += gmemStep;
    }
  }
}


template<int RADIUS>
double SeparableFilter(CudaImage &inimg, CudaImage &outimg, CudaImage &temp, float *h_Kernel) 
{ 
  int width = inimg.width;
  int pitch = inimg.pitch;
  int height = inimg.height;
  float *d_DataA = inimg.d_data;
  float *d_DataB = outimg.d_data;
  float *d_Temp = temp.d_data;
  if (d_DataA==NULL || d_DataB==NULL || d_Temp==NULL) {
    printf("SeparableFilter: missing data\n");
    return 0.0;
  } 
  TimerGPU timer0(0);
  const unsigned int kernelSize = (2*RADIUS+1)*sizeof(float);
  safeCall(cudaMemcpyToSymbol(d_Kernel, h_Kernel, kernelSize));
        
  dim3 blockGridRows(iDivUp(width, CONVROW_W), height);
  dim3 threadBlockRows(CONVROW_W + 2*RADIUS); 
  ConvRowGPU<RADIUS><<<blockGridRows, threadBlockRows>>>(d_Temp, d_DataA, width, pitch, height);
  checkMsg("ConvRowGPU() execution failed\n");
  safeCall(cudaThreadSynchronize());
  dim3 blockGridColumns(iDivUp(width, CONVCOL_W), iDivUp(height, CONVCOL_H));
  dim3 threadBlockColumns(CONVCOL_W, CONVCOL_S);
  ConvColGPU<RADIUS><<<blockGridColumns, threadBlockColumns>>>(d_DataB, d_Temp, width, pitch, height); 
  checkMsg("ConvColGPU() execution failed\n");
  safeCall(cudaThreadSynchronize());

  double gpuTime = timer0.read();
#ifdef VERBOSE
  printf("SeparableFilter time =        %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

template<int RADIUS>
double LowPass(CudaImage &inimg, CudaImage &outimg, CudaImage &temp, double var)
{
  float kernel[2*RADIUS+1];
  float kernelSum = 0.0f;
  for (int j=-RADIUS;j<=RADIUS;j++) {
    kernel[j+RADIUS] = (float)expf(-(double)j*j/2.0/var);      
    kernelSum += kernel[j+RADIUS];
  }
  for (int j=-RADIUS;j<=RADIUS;j++) 
    kernel[j+RADIUS] /= kernelSum;  
  return SeparableFilter<RADIUS>(inimg, outimg, temp, kernel); 
}

double LowPass(CudaImage &inimg, CudaImage &outimg, CudaImage &temp, double var, int kernsize)
{
  if (kernsize<=5)
    return LowPass<2>(inimg, outimg, temp, var);
  else if (kernsize<=7)
    return LowPass<3>(inimg, outimg, temp, var);
  else if (kernsize<=9)
    return LowPass<4>(inimg, outimg, temp, var);
  else {
    if (kernsize>11)
      std::cerr << "Kernels larger than 11 not implemented" << std::endl;
    return LowPass<5>(inimg, outimg, temp, var);
  }
}

__global__ void Scharr(float *imgd, float *lxd, float *lyd, int width, int pitch, int height)
{
  #define BW (SCHARR_W+2)
  __shared__ float buffer[BW*(SCHARR_H+2)];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x*SCHARR_W + tx;
  int y = blockIdx.y*SCHARR_H + ty;
  int xp = (x==0 ? 1 : (x>width ? width-2 : x-1));
  int yp = (y==0 ? 1 : (y>height ? height-2 : y-1));
  buffer[ty*BW + tx] = imgd[yp*pitch + xp];
  __syncthreads();
  if (x<width && y<height && tx<SCHARR_W && ty<SCHARR_H) {
    float *b = buffer + (ty+1)*BW + (tx+1);
    float ul = b[-BW-1];
    float ur = b[-BW+1];
    float ll = b[+BW-1];
    float lr = b[+BW+1];
    lxd[y*pitch+x] = 3.0f*(lr - ll + ur - ul) + 10.0f*(b[+1] - b[-1]); 
    lyd[y*pitch+x] = 3.0f*(lr + ll - ur - ul) + 10.0f*(b[BW] - b[-BW]);
  }
}

double Scharr(CudaImage &img, CudaImage &lx, CudaImage &ly)
{
  TimerGPU timer0(0);
  dim3 blocks(iDivUp(img.width, SCHARR_W), iDivUp(img.height, SCHARR_H));
  dim3 threads(SCHARR_W+2, SCHARR_H+2); 
  Scharr<<<blocks, threads>>>(img.d_data, lx.d_data, ly.d_data, img.width, img.pitch, img.height);
  checkMsg("Scharr() execution failed\n");
  safeCall(cudaThreadSynchronize());
  double gpuTime = timer0.read();
#ifdef VERBOSE
  printf("Scharr time          =        %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

__global__ void Flow(float *imgd, float *flowd, int width, int pitch, int height, DIFFUSIVITY_TYPE type, float invk)
{
  #define BW (SCHARR_W+2)
  __shared__ float buffer[BW*(SCHARR_H+2)];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x*SCHARR_W + tx;
  int y = blockIdx.y*SCHARR_H + ty;
  int xp = (x==0 ? 1 : (x>width ? width-2 : x-1));
  int yp = (y==0 ? 1 : (y>height ? height-2 : y-1));
  buffer[ty*BW + tx] = imgd[yp*pitch + xp];
  __syncthreads();
  if (x<width && y<height && tx<SCHARR_W && ty<SCHARR_H) {
    float *b = buffer + (ty+1)*BW + (tx+1);
    float ul = b[-BW-1];
    float ur = b[-BW+1];
    float ll = b[+BW-1];
    float lr = b[+BW+1];
    float lx = 3.0f*(lr - ll + ur - ul) + 10.0f*(b[+1] - b[-1]); 
    float ly = 3.0f*(lr + ll - ur - ul) + 10.0f*(b[BW] - b[-BW]);
    float dif2 = invk*(lx*lx + ly*ly);
    if (type==PM_G1)
      flowd[y*pitch+x] = exp(-dif2);
    else if (type==PM_G2)
      flowd[y*pitch+x] = 1.0f/(1.0f + dif2);
    else if (type==WEICKERT) 
      flowd[y*pitch+x] = 1.0f - exp(-3.315/(dif2*dif2*dif2*dif2));
    else 
      flowd[y*pitch+x] = 1.0f/sqrt(1.0f + dif2);
  }
}

double Flow(CudaImage &img, CudaImage &flow, DIFFUSIVITY_TYPE type, float kcontrast)
{
  TimerGPU timer0(0);
  dim3 blocks(iDivUp(img.width, SCHARR_W), iDivUp(img.height, SCHARR_H));
  dim3 threads(SCHARR_W+2, SCHARR_H+2); 
  Flow<<<blocks, threads>>>(img.d_data, flow.d_data, img.width, img.pitch, img.height, type, 1.0f/(kcontrast*kcontrast));
  checkMsg("Flow() execution failed\n");
  safeCall(cudaThreadSynchronize());
  double gpuTime = timer0.read();
#ifdef VERBOSE
  printf("Flow time =                   %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

__global__ void NLDStep(float *imgd, float *flod, float *temd, int width, int pitch, int height, float stepsize)
{ 
  #undef BW
  #define BW (NLDSTEP_W+2)
  __shared__ float ibuff[BW*(NLDSTEP_H+2)];
  __shared__ float fbuff[BW*(NLDSTEP_H+2)];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x*NLDSTEP_W + tx;
  int y = blockIdx.y*NLDSTEP_H + ty;
  int xp = (x==0 ? 0 : (x>width ? width-1 : x-1));
  int yp = (y==0 ? 0 : (y>height ? height-1 : y-1));
  ibuff[ty*BW + tx] = imgd[yp*pitch + xp];
  fbuff[ty*BW + tx] = flod[yp*pitch + xp];
  __syncthreads();
  if (tx<NLDSTEP_W && ty<NLDSTEP_H && x<width && y<height) {
    float *ib = ibuff + (ty+1)*BW + (tx+1);
    float *fb = fbuff + (ty+1)*BW + (tx+1);
    float ib0 = ib[0];
    float fb0 = fb[0];
    float xpos = (fb0 + fb[ +1])*(ib[ +1] - ib0);
    float xneg = (fb0 + fb[ -1])*(ib0 - ib[ -1]);
    float ypos = (fb0 + fb[+BW])*(ib[+BW] - ib0);
    float yneg = (fb0 + fb[-BW])*(ib0 - ib[-BW]);
    temd[y*pitch + x] = stepsize*(xpos-xneg + ypos-yneg);
  }
}

__global__ void NLDUpdate(float *imgd, float *temd, int width, int pitch, int height)
{
  int x = blockIdx.x*32 + threadIdx.x;
  int y = blockIdx.y*16 + threadIdx.y;
  if (x<width && y<height) {
    int p = y*pitch + x;
    imgd[p] = imgd[p] + temd[p];
  }
}

double NLDStep(CudaImage &img,CudaImage &flow, CudaImage &temp, float stepsize)
{
  TimerGPU timer0(0);
  dim3 blocks0(iDivUp(img.width, NLDSTEP_W), iDivUp(img.height, NLDSTEP_H));
  dim3 threads0(NLDSTEP_W+2, NLDSTEP_H+2); 
  NLDStep<<<blocks0, threads0>>>(img.d_data, flow.d_data, temp.d_data, img.width, img.pitch, img.height, 0.5f*stepsize);
  checkMsg("NLDStep() execution failed\n");
  safeCall(cudaThreadSynchronize());
  dim3 blocks1(iDivUp(img.width, 32), iDivUp(img.height, 16));
  dim3 threads1(32, 16); 
  NLDUpdate<<<blocks1, threads1>>>(img.d_data, temp.d_data, img.width, img.pitch, img.height);
  checkMsg("NLDUpdate() execution failed\n");
  safeCall(cudaThreadSynchronize());
  double gpuTime = timer0.read();
#ifdef VERBOSE
  printf("NLDStep time =                %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

__global__ void HalfSample(float *iimd, float *oimd, int iwidth, int iheight, int ipitch, int owidth, int oheight, int opitch) 
{
  __shared__ float buffer[16*33];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x*16 + tx;
  int y = blockIdx.y*16 + ty;
  if (x>=owidth || y>=oheight)
    return;
  float *ptri = iimd + (2*y)*ipitch + (2*x);
  if (2*owidth==iwidth) {
    buffer[ty*32 + tx] = owidth*(ptri[0] + ptri[1]);
    ptri += ipitch;
    buffer[ty*32 + tx + 16] = owidth*(ptri[0] + ptri[1]);
    if (ty==15) {
      ptri += ipitch;
      buffer[tx + 32*16] = owidth*(ptri[0] + ptri[1]);
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
    }
  }
  __syncthreads();
  float *buff = buffer + 32*ty + tx;
  if (2*oheight==iheight) 
    oimd[y*opitch + x] = oheight*(buff[0] + buff[16])/(iwidth*iheight); 
  else {
    float f0 = oheight - y;
    float f2 = 1 + y;
    oimd[y*opitch + x] = (f0*buff[0] + oheight*buff[16] + f2*buff[32])/(iwidth*iheight); 
  }
}

__global__ void HalfSample2(float *iimd, float *oimd, int ipitch, int owidth, int oheight, int opitch) 
{
  int x = blockIdx.x*32 + threadIdx.x;
  int y = blockIdx.y*16 + threadIdx.y;
  if (x>=owidth || y>=oheight)
    return;
  float *ptr = iimd + (2*y)*ipitch + (2*x);
  oimd[y*opitch + x] = 0.25f*(ptr[0] + ptr[1] + ptr[ipitch+0] + ptr[ipitch+1]);
}

double HalfSample(CudaImage &inimg, CudaImage &outimg)
{
  TimerGPU timer0(0);
  if (inimg.width==2*outimg.width && inimg.height==2*outimg.height) {
    dim3 blocks(iDivUp(outimg.width, 32), iDivUp(outimg.height, 16));
    dim3 threads(32, 16); 
    HalfSample2<<<blocks, threads>>>(inimg.d_data, outimg.d_data, inimg.pitch, outimg.width, outimg.height, outimg.pitch);
  } else {
    dim3 blocks(iDivUp(outimg.width, 16), iDivUp(outimg.height, 16));
    dim3 threads(16, 16); 
    HalfSample<<<blocks, threads>>>(inimg.d_data, outimg.d_data, inimg.width, inimg.height, inimg.pitch, outimg.width, outimg.height, outimg.pitch);
  }
  checkMsg("HalfSample() execution failed\n");
  safeCall(cudaThreadSynchronize());
  double gpuTime = timer0.read();
#ifdef VERBOSE
  printf("HalfSample time =             %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

double Copy(CudaImage &inimg, CudaImage &outimg)
{
  TimerGPU timer0(0);
  double gpuTime = timer0.read();
  safeCall(cudaMemcpy2D(outimg.d_data, sizeof(float)*outimg.pitch, inimg.d_data, sizeof(float)*outimg.pitch, 
			sizeof(float)*inimg.width, inimg.height, cudaMemcpyDeviceToDevice));
#ifdef VERBOSE
  printf("Copy time =                   %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

void *AllocBuffers(int width, int height, int num, int omax, std::vector<float*> &buffers) 
{
  int w = width;
  int h = height;
  int p = iAlignUp(w, 128);
  int size = 0;        
  std::vector<int> starts(omax*num);
  for (int i=0;i<omax;i++) {
    for (int j=0;j<num;j++) {
      starts[i*num + j] = size;
      size += h*p;
    }
    w /= 2;
    h /= 2;
    p = iAlignUp(w, 128);
  }
  float *memory = NULL;
  size_t pitch;
  safeCall(cudaMallocPitch((void **)&memory, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float)));
  buffers.resize(omax*num);
  for (int i=0;i<omax*num;i++)
    buffers[i] = memory + starts[i];
  return memory;
}

void FreeBuffers(float *buffers)
{
  safeCall(cudaFree(buffers));
}

__device__ unsigned int d_Maxval[1];
__device__ int d_Histogram[512];

#define CONTRAST_W 64
#define CONTRAST_H 7
#define HISTCONT_W 64
#define HISTCONT_H 8
#define HISTCONT_R 4

__global__ void MaxContrast(float *imgd, float *cond, int width, int pitch, int height)
{
  #define WID (CONTRAST_W + 2)
  __shared__ float buffer[WID*(CONTRAST_H + 2)];
  __shared__ unsigned int maxval[32];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  if (tx<32 && !ty)
    maxval[tx] = 0.0f;
  __syncthreads();
  int x = blockIdx.x*CONTRAST_W + tx;
  int y = blockIdx.y*CONTRAST_H + ty;
  if (x>=width || y>=height)
    return;
  float *b = buffer + ty*WID + tx;
  b[0] = imgd[y*pitch + x];
  __syncthreads();
  if (tx<CONTRAST_W && ty<CONTRAST_H && x<width-2 && y<height-2) {
    float dx = 3.0f*(b[0] - b[2] + b[2*WID] - b[2*WID+2]) + 10.0f*(b[WID] - b[WID+2]);
    float dy = 3.0f*(b[0] + b[2] - b[2*WID] - b[2*WID+2]) + 10.0f*(b[1] - b[2*WID+1]);
    float grad = sqrt(dx*dx + dy*dy);
    cond[(y+1)*pitch + (x+1)] = grad;
    unsigned int *gradi = (unsigned int *)&grad;
    atomicMax(maxval + (tx&31), *gradi);
  }
  __syncthreads();
  if (tx<32 && !ty) 
    atomicMax(d_Maxval, maxval[tx]);
}

 __global__ void HistContrast(float *cond, int width, int pitch, int height, float imaxval, int nbins)
{
  __shared__ int hist[512];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int i = ty*HISTCONT_W + tx;
  if (i<nbins)
    hist[i] = 0;
  __syncthreads();
  int x = blockIdx.x*HISTCONT_W + tx;
  int y = blockIdx.y*HISTCONT_H*HISTCONT_R + ty;
  if (x>0 && x<width-1) {
    for (int i=0;i<HISTCONT_R;i++) {
      if (y>0 && y<height-1) {
        int idx = min((int)(nbins*cond[y*pitch + x]*imaxval), nbins-1);
        atomicAdd(hist + idx, 1);
      }
      y += HISTCONT_H;
    }
  }
  __syncthreads();
  if (i<nbins && hist[i]>0)
    atomicAdd(d_Histogram + i, hist[i]);
}

double ContrastPercentile(CudaImage &img, CudaImage &temp, CudaImage &blur, float perc, int nbins, float &contrast)
{
  TimerGPU timer0(0);
  LowPass(img, blur, temp, 1.0f, 5);

  float h_Maxval = 0.0f;
  safeCall(cudaMemcpyToSymbol(d_Maxval, &h_Maxval, sizeof(float)));
  dim3 blocks1(iDivUp(img.width, CONTRAST_W), iDivUp(img.height, CONTRAST_H));
  dim3 threads1(CONTRAST_W + 2, CONTRAST_H + 2);
  MaxContrast<<<blocks1,threads1>>>(blur.d_data, temp.d_data, blur.width, blur.pitch, blur.height);
  checkMsg("MaxContrast() execution failed\n");
  safeCall(cudaThreadSynchronize());
  safeCall(cudaMemcpyFromSymbol(&h_Maxval, d_Maxval, sizeof(float)));

  if (nbins>512) {
    printf("Warning: Largest number of possible bins in ContrastPercentile() is 512\n");
    nbins = 512;
  }
  int h_Histogram[512];
  memset(h_Histogram, 0, nbins*sizeof(int));
  safeCall(cudaMemcpyToSymbol(d_Histogram, h_Histogram, nbins*sizeof(int)));
  dim3 blocks2(iDivUp(temp.width, HISTCONT_W), iDivUp(temp.height, HISTCONT_H*HISTCONT_R));
  dim3 threads2(HISTCONT_W, HISTCONT_H);
  HistContrast<<<blocks2,threads2>>>(temp.d_data, temp.width, temp.pitch, temp.height, 1.0f/h_Maxval, nbins);
  safeCall(cudaMemcpyFromSymbol(h_Histogram, d_Histogram, nbins*sizeof(int)));

  int npoints = (temp.width-2)*(temp.height-2);
  int nthreshold = (int)(npoints*perc);
  int k = 0, nelements = 0;
  for (k=0;nelements<nthreshold && k<nbins;k++)
    nelements += h_Histogram[k];
  contrast = (nelements<nthreshold ? 0.03f : h_Maxval*((float)k/nbins));

  double gpuTime = timer0.read();
#ifdef VERBOSE
  printf("ContrastPercentile time =     %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

__global__ void Derivate(float *imd, float *lxd, float *lyd, int width, int pitch, int height, int step, float fac1, float fac2)
{
  int x = blockIdx.x*32 + threadIdx.x;
  int y = blockIdx.y*16 + threadIdx.y;
  if (x>=width || y>=height)
    return;
  int xl = (x<step ? step-x : x-step);
  int xh = (x>=width-step ? 2*width-x-step-2 : x+step);
  int yl = (y<step ? step-y : y-step);
  int yh = (y>=height-step ? 2*height-y-step-2 : y+step);
  float ul = imd[yl*pitch + xl];
  float ur = imd[yl*pitch + xh];
  float ll = imd[yh*pitch + xl];
  float lr = imd[yh*pitch + xh];
  float cl = imd[y*pitch + xl];
  float cr = imd[y*pitch + xh];
  lxd[y*pitch + x] = fac1*(ur + lr - ul - ll) + fac2*(cr - cl);
  float uc = imd[yl*pitch + x];
  float lc = imd[yh*pitch + x];
  lyd[y*pitch + x] = fac1*(lr + ll - ur - ul) + fac2*(lc - uc);
}

__global__ void HessianDeterminant(float *lxd, float *lyd, float *detd, int width, int pitch, int height, int step, float fac1, float fac2)
{
  int x = blockIdx.x*32 + threadIdx.x;
  int y = blockIdx.y*16 + threadIdx.y;
  if (x>=width || y>=height)
    return;
  int xl = (x<step ? step-x : x-step);
  int xh = (x>=width-step ? 2*width-x-step-2 : x+step);
  int yl = (y<step ? step-y : y-step);
  int yh = (y>=height-step ? 2*height-y-step-2 : y+step);
  float ul = lxd[yl*pitch + xl];
  float ur = lxd[yl*pitch + xh];
  float ll = lxd[yh*pitch + xl];
  float lr = lxd[yh*pitch + xh];
  float cl = lxd[y*pitch + xl];
  float cr = lxd[y*pitch + xh];
  float lxx = fac1*(ur + lr - ul - ll) + fac2*(cr - cl);
  float uc = lxd[yl*pitch + x];
  float lc = lxd[yh*pitch + x];
  float lyx = fac1*(lr + ll - ur - ul) + fac2*(lc - uc);
  ul = lyd[yl*pitch + xl];
  ur = lyd[yl*pitch + xh];
  ll = lyd[yh*pitch + xl];
  lr = lyd[yh*pitch + xh];
  uc = lyd[yl*pitch + x];
  lc = lyd[yh*pitch + x];
  float lyy = fac1*(lr + ll - ur - ul) + fac2*(lc - uc);
  detd[y*pitch + x] = lxx*lyy - lyx*lyx;
}

double HessianDeterminant(CudaImage &img, CudaImage &lx, CudaImage &ly, int step)
{
  TimerGPU timer0(0);
  float w = 10.0/3.0;
  float fac1 = 1.0/(2.0*(w + 2.0));
  float fac2 = w*fac1;
  dim3 blocks(iDivUp(img.width, 32), iDivUp(img.height, 16));
  dim3 threads(32, 16);
  Derivate<<<blocks, threads>>>(img.d_data, lx.d_data, ly.d_data, img.width, img.pitch, img.height, step, fac1, fac2);
  checkMsg("Derivate() execution failed\n");
  safeCall(cudaThreadSynchronize());
  HessianDeterminant<<<blocks, threads>>>(lx.d_data, ly.d_data, img.d_data, img.width, img.pitch, img.height, step, fac1, fac2);
  checkMsg("HessianDeterminant() execution failed\n");
  safeCall(cudaThreadSynchronize());
  double gpuTime = timer0.read();
#ifdef VERBOSE
  printf("HessianDeterminant time =     %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

 __global__ void FindExtrema(float *imd, float *imp, float *imn, int maxx, int pitch, int maxy, int border, float dthreshold, int scale)
{
  int x = blockIdx.x*32 + threadIdx.x;
  int y = blockIdx.y*16 + threadIdx.y;
  if (x<border || x>=maxx || y<border || y>=maxy) 
    return;
  int p = y*pitch + x;
  float v = imd[p];
  if (v>dthreshold && v>imd[p-pitch-1] && v>imd[p+pitch+1] && v>imd[p-pitch+1] && 
      v>imd[p-pitch+1] &&  v>imd[p-1] && v>imd[p+1] && v>imd[p+pitch] && 
      v>imd[p-pitch] && v>=imn[p] && v>=imp[p]) {
    int s = 1<<(scale/4);
    //printf("XXX %02d %03d %03d\n", scale, x*s, y*s);
    printf("XXX %02d %03d %03d %f\n", scale, x*s, y*s, v);
  }
}

double FindExtrema(CudaImage &img, CudaImage &imgp, CudaImage &imgn, float border, float dthreshold, int scale)
{
  TimerGPU timer0(0);
  dim3 blocks(iDivUp(img.width, 32), iDivUp(img.height, 16));
  dim3 threads(32, 16);
  int b = (int)border;
  std::cout << img.d_data << " " << imgp.d_data << " " << imgn.d_data << std::endl;
  std::cout << img.width << " " << imgp.width << " " << imgn.width << std::endl;
  FindExtrema<<<blocks, threads>>>(img.d_data, imgp.d_data, imgn.d_data, img.width-b, img.pitch, img.height-b, b, dthreshold, scale);
  checkMsg("FindExtrema() execution failed\n");
  safeCall(cudaThreadSynchronize());
  double gpuTime = timer0.read();
#ifdef VERBOSE
  printf("FindExtrema time =            %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

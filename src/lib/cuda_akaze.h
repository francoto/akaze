#include "AKAZEConfig.h"
#include "cudaImage.h"

void *AllocBuffers(int width, int height, int num, int omax, std::vector<float*> &buffers);
void FreeBuffers(float *buffers);
double LowPass(CudaImage &inimg, CudaImage &outimg, CudaImage &temp, double var, int kernsize);
double Scharr(CudaImage &img, CudaImage &lx, CudaImage &ly);
double Flow(CudaImage &img, CudaImage &flow, DIFFUSIVITY_TYPE type, float kcontrast);
double NLDStep(CudaImage &img,CudaImage &flow, CudaImage &temp, float stepsize);
double HalfSample(CudaImage &inimg, CudaImage &outimg);
double Copy(CudaImage &inimg, CudaImage &outimg);
double ContrastPercentile(CudaImage &img, CudaImage &temp, CudaImage &blur, float perc, int nbins, float &contrast);
double HessianDeterminant(CudaImage &img, CudaImage &lx, CudaImage &ly, int step);
double FindExtrema(CudaImage &img, CudaImage &imgp, CudaImage &imgn, float border, float dthreshold, int scale);

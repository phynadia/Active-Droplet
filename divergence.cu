#include "global.h"

__global__ void divergence(complex *cux, complex *cuy, complex *cwz, double *kx, double *ky)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = IDXC(index_x, index_y);

    double cwz_re, cwz_im;

    if(index_x < Nx && index_y < Nyh)
    {
      double ksqr = kx[index_x] * kx[index_x] + ky[index_y] * ky[index_y];
      if(ksqr > nalias)
      {
        cux[index].x = 0.0; cux[index].y = 0.0;
        cuy[index].x = 0.0; cuy[index].y = 0.0;
        cwz[index].x = 0.0; cwz[index].y = 0.0;
      }

      cwz_re =  (-1.0) * (kx[index_x] * cux[index].y 
                               +ky[index_y] * cuy[index].y);

      cwz_im =  (kx[index_x] * cux[index].x 
                     + ky[index_y] * cuy[index].x);
      
      cwz[index].x = cwz_re;
      cwz[index].y = cwz_im;
    }
}
/******************************************************************************************/

__global__ void curl(complex *cux, complex *cuy, complex *cwz, double *kx, double *ky)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = IDXC(index_x, index_y);

    if(index_x < Nx && index_y < Nyh)
    {
      double ux_re, ux_im, uy_re, uy_im;
      double ksqr = kx[index_x] * kx[index_x] + ky[index_y] * ky[index_y];
      if(ksqr > nalias)
      {
        cux[index].x = 0.0; cux[index].y = 0.0;
        cuy[index].x = 0.0; cuy[index].y = 0.0;
        cwz[index].x = 0.0; cwz[index].y = 0.0;
      }

      ux_re = cux[index].x;
      ux_im = cux[index].y;
      uy_re = cuy[index].x;
      uy_im = cuy[index].y;
       
      cwz[index].x = -kx[index_x] * uy_im + ky[index_y] * ux_im;
      cwz[index].y =  kx[index_x] * uy_re - ky[index_y] * ux_re;
    }
}

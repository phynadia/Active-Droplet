#include "global.h"
__global__ void velocity(complex *cwz, complex *cux, complex *cuy, double *kx, double *ky)
{
    double psi_re = 0.0, psi_im = 0.0;
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = IDXC(index_x, index_y);

    if(index_x < Nx && index_y < Nyh)
    {
      double ksqr = kx[index_x]*kx[index_x] + ky[index_y]*ky[index_y];
      if(ksqr > 1e-10)
      {
        psi_re = cwz[index].x / ksqr;
        psi_im = cwz[index].y / ksqr;
      }
      if(ksqr > nalias)
      {

        cwz[index].x = 0.0;
        cwz[index].y = 0.0;
        cux[index].x = 0.0;
        cux[index].y = 0.0;
        cuy[index].x = 0.0;
        cuy[index].y = 0.0;

      }else{
             cux[index].x = -ky[index_y] * psi_im;
             cux[index].y =  ky[index_y] * psi_re;
             cuy[index].x =  kx[index_x] * psi_im;
             cuy[index].y = -kx[index_x] * psi_re;
           }
    }
}//End of the function.......


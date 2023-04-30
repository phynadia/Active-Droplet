#include "global.h"

__global__ void chemical_potential(complex *cphi, complex *cmu, complex *cgphi1, complex *cgphi2, complex *cpsi, complex *cmu_psi, complex *cgpsi1, complex *cgpsi2, double *kx, double *ky)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = IDXC(index_x, index_y);
    double ksqr;

    if(index_x < Nx && index_y < Nyh)
    {
      ksqr = kx[index_x] * kx[index_x] + ky[index_y] * ky[index_y];

      if(ksqr > nalias)
      {
        cmu[index].x = 0.0; cmu[index].y = 0.0;
        cphi[index].x = 0.0; cphi[index].y = 0.0;
        cgphi1[index].x = 0.0; cgphi1[index].y = 0.0;
        cgphi2[index].x = 0.0; cgphi2[index].y = 0.0;
      }else
       {
        cmu[index].x = 1.5 * sigma1 * epsilon * ksqr * cphi[index].x;
        cmu[index].y = 1.5 * sigma1 * epsilon * ksqr * cphi[index].y;

        cgphi1[index].x = -kx[index_x] * cphi[index].y;
        cgphi1[index].y =  kx[index_x] * cphi[index].x;
        cgphi2[index].x = -ky[index_y] * cphi[index].y;
        cgphi2[index].y =  ky[index_y] * cphi[index].x;
       }

      if(ksqr > nalias)
      {
        cmu_psi[index].x = 0.0; cmu_psi[index].y = 0.0;
        cpsi[index].x = 0.0; cpsi[index].y = 0.0;
        cgpsi1[index].x = 0.0; cgpsi1[index].y = 0.0;
        cgpsi2[index].x = 0.0; cgpsi2[index].y = 0.0;
      }else
       {
        cmu_psi[index].x = -1.5 * activity * epsilon * ksqr * cpsi[index].x;
        cmu_psi[index].y = -1.5 * activity * epsilon * ksqr * cpsi[index].y;

        cgpsi1[index].x = -kx[index_x] * cpsi[index].y;
        cgpsi1[index].y =  kx[index_x] * cpsi[index].x;
        cgpsi2[index].x = -ky[index_y] * cpsi[index].y;
        cgpsi2[index].y =  ky[index_y] * cpsi[index].x;
       }
    }
}

/************************************************************************************/

__global__ void surface_stress(double *phi, double *gphi1, double *gphi2, double *psi, double *gpsi1, double *gpsi2
)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = IDXR(index_x, index_y);

    if(index_x < Nx && index_y < Ny)
    {
      gphi1[index] *= phi[index];
      gphi2[index] *= phi[index];

      gpsi1[index] *= psi[index];
      gpsi2[index] *= psi[index];

      gpsi1[index] += gphi1[index];
      gpsi2[index] += gphi2[index];              
    }

}

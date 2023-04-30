#include "global.h"

__global__ void nonlin(double *w, double *ux, double *uy)
{

    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_y = threadIdx.y + blockIdx.y * blockDim.y;

    int index = IDXR(index_x, index_y);

      if(index_x < Nx && index_y < Ny)
      {
        ux[index] *= w[index];
        uy[index] *= w[index];
      }
}

__global__ void nonlin_phi_psi(double *tphi, double *nphi, double *nlphi, double *ux, double *uy, double *tpsi, double *npsi, double *nlpsi
)
{

    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_y = threadIdx.y + blockIdx.y * blockDim.y;

    int index = IDXR(index_x, index_y);

      if(index_x < Nx && index_y < Ny)
      {
        double phi, psi;
        phi = tphi[index];
        psi = tpsi[index];

        nlphi[index] = -phi + phi * phi * phi - beta * psi;

        tphi[index] = phi * ux[index];
        nphi[index] = phi * uy[index];
        

        nlpsi[index] = -psi + psi * psi * psi - beta * phi;

        tpsi[index] = psi * ux[index];
        npsi[index] = psi * uy[index];
      }

}

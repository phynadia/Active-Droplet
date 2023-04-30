#include "global.h"

__global__ void init(double *wz
#ifdef FORCING
, double *fwz
#endif
)
{
    double x, y;

    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int index;

    for(; index_x < Nx; index_x += gridDim.x*blockDim.x)
    {
        x = dx * index_x;
        for(; index_y < Ny; index_y += gridDim.y*blockDim.y)
        {
            y = dy * index_y;

            index = IDXR(index_x, index_y);

              wz[index] = uamp * sin(2*x) * sin(2*y);

            #ifdef FORCING
              fwz[index] = famp * kf * cos(kf * y);
            #endif

        }
    }


}
/*****************************************************************************/
void init_vorticity(double *wz
#ifdef FORCING
, double *fwz
#endif
)
{
    double x, y;
    int index;

    #ifdef INIT_READ_DATA
      char filename[100];
      sprintf(filename, "init_vort/wz.in");
      read_array_real(wz, filename);
    #endif

    for(int i = 0; i < Nx; i++)
    {
        x = dx * i;
        for(int j = 0; j < Ny; j++)
        {
            y = dy * j;

            index = IDXR(i, j);

            #ifndef INIT_READ_DATA
              wz[index] = uamp * sin(2*x) * sin(2*y);
            #endif

            #ifdef FORCING
              fwz[index] = famp * kf * cos(kf * y);
            #endif

        }
    }
}
/*****************************************************************************/
/*
__global__ void init_phi(double *phi)
{
    double x, y;

    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int index;

    for(; index_x < Nx; index_x += gridDim.x*blockDim.x)
    {
        x = dx * index_x;
        for(; index_y < Ny; index_y += gridDim.y*blockDim.y)
        {
            y = dy * index_y;

            index = IDXR(index_x, index_y);

            phi[index] = -tanh((sqrt((x - lx/2.0) * (x - lx/2.0)/(pi*pi/9.0)
                       +(y-ly/2.0) * (y-ly/2.0)/(pi*pi/16.0)) - 1.0) / (epsilon));

        }
    }
}

__global__ void init_psi(double *psi)
{
    double x, y;

    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int index;

    for(; index_x < Nx; index_x += gridDim.x*blockDim.x)
    {
        x = dx * index_x;
        for(; index_y < Ny; index_y += gridDim.y*blockDim.y)
        {
            y = dy * index_y;

            index = IDXR(index_x, index_y);

            psi[index] = -tanh((sqrt((x - lx/2.0) * (x - lx/2.0)
                                    +(y-ly/2.0) * (y-ly/2.0)) - (pi/5.0)) / (epsilon));

            //psi[index] *= 0.5;
        }
    }
}
*/


///////////////////////////////////////////////////////////////

void host_init_phi_psi(double *phi, double *psi)
{
    srand((unsigned)time(NULL));
    double rn1, rn2;

    for(int i = 0; i < Nx; i++)
    {
      double x = i * dx;
      for(int j = 0; j < Ny; j++)
      {
        double y = j * dy;

        int index = IDXR(i, j);

        rn1 = (double) rand() / (double)(RAND_MAX);
        rn2 = (double) rand() / (double)(RAND_MAX);

      //  theta = 2.0 * pi * rn2;
 
 //        xp[ip] = pi + (pi/5.5) * rn1 * cos(theta);
 //       yp[ip] = pi + (pi/5.5) * rn2 * sin(theta);

         phi[index] = -tanh((sqrt((x - lx/2.0) * (x - lx/2.0)/(pi*pi/4.0)
                      + (y-ly/2.0) * (y-ly/2.0)/(pi*pi/4.0)) - 1.0) / (epsilon));

       //  psi[index] = -tanh((sqrt((x - lx/2.0) * (x - lx/2.0)/(pi*pi/16.0)
       //               + (y-ly/2.0) * (y-ly/2.0)/(pi*pi/16.0)) - 1.0) / (epsilon));

//         phi[index] = 0.0; psi[index] = 0.0;

         if(phi[index] >= 0.0)
           psi[index] = 0.0 + 0.1 * (1.0 - 2*rn1);
         else
           psi[index] = -1;

        

      }
    }
}


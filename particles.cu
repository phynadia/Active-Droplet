#include "global.h"

void cpu_particles_evolution(double *x, double *y, double *ux, double *uy)
{
    double uxp, uyp, xp , yp;
    double xxp, yyp, xd, yd;
    double g1, g2, g3, g4;
    int ix, iy, ixp, iyp;
    int Idx_11, Idx_12, Idx_21, Idx_22;

    for(int ip = 0; ip < nparticles; ip++)
    {
      xp = x[ip];
      yp = y[ip];

      xxp = fmod(xp, lx);
      yyp = fmod(yp, ly);

      if(xp < 0.0)
      {
        xxp = lx + xxp;
      }
      if(yp < 0.0)
      {
        yyp = ly + yyp;
      }

      ix = floor(xxp / dx);
      iy = floor(yyp / dy);

      ixp = ix + 1;
      iyp = iy + 1;
      ixp = (1 - ixp / Nx) * ixp;
      iyp = (1 - iyp / Ny) * iyp;

      xd = fmod(xxp, dx) / dx;
      yd = fmod(yyp, dy) / dy;

      Idx_11 = IDXR(ix, iy);
      Idx_12 = IDXR(ix, iyp);
      Idx_21 = IDXR(ixp, iy);
      Idx_22 = IDXR(ixp, iyp);

      g1 = ux[Idx_11];
      g2 = ux[Idx_21];
      g3 = ux[Idx_12];
      g4 = ux[Idx_22];

      uxp = g1 + xd * (g2 - g1) + yd * (g3 - g1)
          + xd * yd * (g1 + g4 - g2 - g3);

      g1 = uy[Idx_11];
      g2 = uy[Idx_21];
      g3 = uy[Idx_12];
      g4 = uy[Idx_22];

      uyp = g1 + xd * (g2 - g1) + yd * (g3 - g1)
          + xd * yd * (g1 + g4 - g2 - g3);


      xp = xxp + dt * uxp;
      yp = yyp + dt * uyp;

      xp = fmod(xp, lx);
      yp = fmod(yp, ly);

      if(xp < 0.0)
      {
        xp = lx + xp;
      }

      if(yp < 0.0)
      {
        yp = ly + yp;
      }

      x[ip] = xp;
      y[ip] = yp;
   }      
}
/****************************************************************/


__global__ void particles_evolution(double *x, double *y, double *ux, double *uy)
{

    double uxp, uyp, xp , yp;
    double xxp, yyp, xd, yd;
    double g1, g2, g3, g4;
    int ix, iy, ixp, iyp;
    int Idx_11, Idx_12, Idx_21, Idx_22;

    int ip = threadIdx.x + blockIdx.x * blockDim.x;

    if(ip < nparticles)
    {
      xp = x[ip];
      yp = y[ip];

      xxp = fmod(xp, lx);
      yyp = fmod(yp, ly);

      if(xp < 0.0)
      {
        xxp = lx + xxp;
      }

      if(yp < 0.0)
      {
        yyp = ly + yyp; 
      }

      ix = floor(xxp / dx);
      iy = floor(yyp / dy);

      ixp = ix + 1;
      iyp = iy + 1;
      ixp = (1 - ixp / Nx) * ixp;
      iyp = (1 - iyp / Ny) * iyp;

      xd = fmod(xxp, dx) / dx;
      yd = fmod(yyp, dy) / dy; 

      Idx_11 = IDXR(ix, iy);
      Idx_12 = IDXR(ix, iyp);
      Idx_21 = IDXR(ixp, iy);
      Idx_22 = IDXR(ixp, iyp);

      g1 = ux[Idx_11];
      g2 = ux[Idx_21];
      g3 = ux[Idx_12];
      g4 = ux[Idx_22];

      uxp = g1 + xd * (g2 - g1) + yd * (g3 - g1)
          + xd * yd * (g1 + g4 - g2 - g3);
    
      g1 = uy[Idx_11];
      g2 = uy[Idx_21];
      g3 = uy[Idx_12];
      g4 = uy[Idx_22];

      uyp = g1 + xd * (g2 - g1) + yd * (g3 - g1)
          + xd * yd * (g1 + g4 - g2 - g3);
 

      xp = xxp + dt * uxp;
      yp = yyp + dt * uyp;

      xp = fmod(xp, lx);
      yp = fmod(yp, ly);

      if(xp < 0.0)
      {
        xp = lx + xp;
      }

      if(yp < 0.0)
      {
        yp = ly + yp;
      }

      x[ip] = xp;
      y[ip] = yp;

  }
}//End of the function.....................

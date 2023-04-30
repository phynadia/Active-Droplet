#include "global.h"


__global__ void normalize(double *w)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;

    int index;
    double scale;

    scale = 1.0 / (Nx * Ny);

    for(; index_x < Nx; index_x += gridDim.x*blockDim.x)
    {
        for(; index_y < Ny; index_y += gridDim.y*blockDim.y)
        {
            index = IDXR(index_x, index_y);

            w[index] *= scale;
        }
    }
}

/**************************************************************************************/
__global__ void dealiasing(complex *w, double *kx, double *ky)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;

    int index;

    for(; index_x < Nx; index_x += gridDim.x*blockDim.x)
    {
        for(; index_y < Nyh; index_y += gridDim.y*blockDim.y)
        {
            index = IDXC(index_x, index_y);

            double ksqr = kx[index_x] * kx[index_x] + ky[index_y] * ky[index_y];

            if(ksqr > nalias)
            {
              w[index].x = 0.0;
              w[index].y = 0.0;
            }
        }
    }
}
/***************************************************************************************/
__global__ void init_kxky(double *kx, double *ky)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(; index < Nx; index += gridDim.x*blockDim.x)
    {
        if(index < Nx/2)
            kx[index] = double(index) * (2.0*pi/lx);
        else
            kx[index] = double(index - Nx) * (2.0*pi/lx);
    }

    index = blockIdx.x * blockDim.x + threadIdx.x;

    for(; index < Nyh; index += gridDim.x*blockDim.x)
    {
        ky[index] = double(index) * (2.0*pi/ly);
    }
}
/***************************************************************************************/
void energy(double *ux, double *uy, double *eu)
{
    double scale;
    *eu = 0.0;
    scale = 1.0 / (Nx * Ny);

    for(int i = 0; i < Nx; i++)
    {
      for(int j = 0; j < Ny; j++)
      {
        int index = IDXR(i, j);//j + i * (Ny+2);

        *eu += 0.5 * scale * (ux[index] * ux[index]
                            + uy[index] * uy[index]);

      }
    }
   // printf("energy = %lf\n", *eu / (Nx*Ny));
}
/***************************************************************************************/

void write_array_real(double *u, char filename[100])
{
    FILE *f;
    int i, j, index;

    f = fopen(filename, "wb");

    for(i = 0; i < Nx; i++)
    {
      for(j = 0; j < Ny; j++)
      {

       index = IDXR(i, j);
       fwrite(&u[index], sizeof(double), 1, f);

      }
    }

    fclose(f);
}

/***************************************************************************************/

void read_array_real(double *u, char filename[100])
{
    FILE *f;
    int i, j, index;

    f = fopen(filename, "rb");

    for(i = 0; i < Nx; i++)
    {
      for(j = 0; j < Ny; j++)
      {

       index = IDXR(i, j);
       fread(&u[index], sizeof(double), 1, f);

      }
    }

    fclose(f);
}

/****************************************************************************************/

void write_particle_positions(double *x, double *y, char filename[100])
{
    FILE *f;
    int pindex;

    f = fopen(filename, "wb");

    for(pindex = 0; pindex < nparticles; pindex++)
    {
      fwrite(&x[pindex], sizeof(double), 1, f);
      fwrite(&y[pindex], sizeof(double), 1, f); 
    }

    fclose(f);
}

/****************************************************************************************/
void deformation(double *phi, double *ux, double *uy, FILE *fp1, FILE *fp2, FILE *fp3, double *t)
{
    double sph,en_drop, St, S0, gamma;
    double u1, u2, ph, Vx, Vy, V, x_cm, y_cm;
    int i, j, index, idx1, idx2, i1, j1;

    sph = 0.0; en_drop = 0.0; St = 0.0;

    for(i = 0; i < Nx; i++)
    {
      for(j = 0; j < Ny; j++)
      {
       
       index = IDXR(i, j);
       ph = phi[index];

       if(ph > 0.0)
       {
	  ph = 1.0;

	  sph += ph;
	  phi[index] = 1.0;

       }else{
              phi[index] = -1.0;
            }

      }
    }

    for(i = 0; i < Nx; i++)
    {
      for(j = 0; j < Ny; j++)
      {

       index = IDXR(i, j);
       i1 = i + 1;
       j1 = j + 1;

       if(i1 >= Nx) i1 = 0;
       if(j1 >= Ny) j1 = 0;

       idx1 = IDXR(i1, j);
       idx2 = IDXR(i, j1);

       u1 = ux[index];
       u2 = uy[index];
       ph = phi[index];

       if(ph > 0.0)
       {      

	 en_drop += (u1 * u1 + u2 * u2);

         Vx += u1 * ph;
	 Vy += u2 * ph; 

         x_cm += i * ph;
         y_cm += j * ph;
         
       }

       if((phi[index]*phi[idx1]) < 0.0 || (phi[index]*phi[idx2]) < 0.0)
       {
         St += 1.0; 
       }

      }
    }

    Vx = Vx / sph;
    Vy = Vy / sph;
    V = sqrt(Vx * Vx + Vy * Vy);

    en_drop = en_drop / sph;

    x_cm = (x_cm * dx) / sph;
    y_cm = (y_cm * dx) / sph;

    St = St * dx;
    S0 = 2.0 * sqrt(pi * sph * dx * dy);
    gamma = (St / S0) - 1.0;

    fprintf(fp1, "%lf \t %lf \t %lf \n", *t, x_cm, y_cm);
    fprintf(fp2, "%lf \t %lf \t %lf \t %lf \t %lf\n", *t, Vx, Vy, V, en_drop);
    fprintf(fp3, "%lf \t %lf \t %lf \t %lf\n", *t, S0, St, gamma);

}//End of the function.............................




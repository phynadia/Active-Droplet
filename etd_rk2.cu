#include "global.h"

__global__ void ETD2RK_step1(complex *cwz, complex *cnln,
#ifdef FORCING
  complex *cfwz,
#endif
#ifdef PHI_PSI
  complex *csf, 
#endif
double *kx, double *ky)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = IDXC(index_x, index_y);
    double ksqr, exp_w, etd_c;

    if(index_x < Nx && index_y < Nyh)
    {

      ksqr = kx[index_x] * kx[index_x] + ky[index_y] * ky[index_y];

      etd_c = -(nu * ksqr + alpha);
      exp_w = exp(etd_c * dt);

      if(etd_c != 0.0)
      {
        etd_c = (exp_w - 1.0) / etd_c;
      }else{
             etd_c = 0.0;
           }

      #ifdef FORCING
        cnln[index].x = cfwz[index].x - cnln[index].x;
        cnln[index].y = cfwz[index].y - cnln[index].y;
      #else
        cnln[index].x *= (-1.0);
        cnln[index].y *= (-1.0);
      #endif

      #ifdef PHI_PSI
        cnln[index].x += (csf[index].x);
        cnln[index].y += (csf[index].y);
      #endif
 
        cwz[index].x = exp_w * cwz[index].x + cnln[index].x * etd_c;
        cwz[index].y = exp_w * cwz[index].y + cnln[index].y * etd_c;
    }
}

/*****************************************************************************************/

__global__ void ETD2RK_step2(complex *cwz, complex *cnln, complex *cnlp,
#ifdef FORCING
  complex *cfwz,
#endif
#ifdef PHI_PSI
  complex *csf,
#endif
double *kx, double *ky)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = IDXC(index_x, index_y);
    double ksqr, exp_w, etd_c;

    if(index_x < Nx && index_y < Nyh)
    {

      ksqr = kx[index_x] * kx[index_x] + ky[index_y] * ky[index_y];

      etd_c = -(nu * ksqr + alpha);
      exp_w = exp(etd_c * dt);
    
      if(etd_c != 0.0)
      {
        etd_c = (exp_w - 1.0 - etd_c * dt) / (etd_c * etd_c * dt);
      }else{
             etd_c = 0.0;
           }

      #ifdef FORCING
        cnln[index].x = cfwz[index].x - cnln[index].x;
        cnln[index].y = cfwz[index].y - cnln[index].y;
      #else
        cnln[index].x *= (-1.0);
        cnln[index].y *= (-1.0);
      #endif

      #ifdef PHI_PSI
        cnln[index].x += (csf[index].x);
        cnln[index].y += (csf[index].y);
      #endif

        cwz[index].x = cwz[index].x + (cnln[index].x - cnlp[index].x) * etd_c;
        cwz[index].y = cwz[index].y + (cnln[index].y - cnlp[index].y) * etd_c;  
    }

}

/******************************************************************************************/
__global__ void etdrk2_step1(complex *cphi, complex *cnphi, complex *cnlphi, 
complex *cpsi, complex *cnpsi, complex *cnlpsi, double *kx, double *ky)
{

    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = IDXC(index_x, index_y);
    double ksqr, exp_p, etd_c;
    double ph1, ph2;

    ph1 = 1.5 * lambda1 * sigma1 * epsilon;
    ph2 = 0.75 * lambda1 * (sigma1 / epsilon);

    double ps1, ps2, exp_s, etd_s;
    ps1 = 1.5 * lambda2 * sigma2 * epsilon;
    ps2 = 0.75 * lambda2 * (sigma2 / epsilon);  

    if(index_x < Nx && index_y < Nyh)
    {

      ksqr = kx[index_x] * kx[index_x] + ky[index_y] * ky[index_y];

      etd_c = (-1.0) * ph1 * ksqr * ksqr;
      exp_p = exp(etd_c * dt);

      if(etd_c != 0.0)
      {
        etd_c = (exp_p - 1.0) / etd_c;
      }else{
             etd_c = 0.0;
           }
      double nln_re;
      double nln_im;
       
      if(ksqr > nalias)
      {
        cnphi[index].x = 0.0; cnphi[index].y = 0.0;
        cnlphi[index].x = 0.0; cnlphi[index].y = 0.0;
      }

      nln_re = (-1.0) * (cnphi[index].x + ph2 * ksqr * cnlphi[index].x);
      nln_im = (-1.0) * (cnphi[index].y + ph2 * ksqr * cnlphi[index].y);

      cphi[index].x = exp_p * cphi[index].x + nln_re * etd_c;
      cphi[index].y = exp_p * cphi[index].y + nln_im * etd_c;
  
      cnlphi[index].x = nln_re;
      cnlphi[index].y = nln_im;                

        etd_s = (-1.0) * ps1 * ksqr * ksqr;
        exp_s = exp(etd_s * dt);

        if(etd_s != 0.0)
        {
          etd_s = (exp_s - 1.0) / etd_s;
        }else{
               etd_s = 0.0;
             }

        if(ksqr > nalias)
        {
          cnpsi[index].x = 0.0; cnpsi[index].y = 0.0;
          cnlpsi[index].x = 0.0; cnlpsi[index].y = 0.0;
        }

        nln_re = (-1.0) * (cnpsi[index].x + ps2 * ksqr * cnlpsi[index].x);
        nln_im = (-1.0) * (cnpsi[index].y + ps2 * ksqr * cnlpsi[index].y);

        cpsi[index].x = exp_s * cpsi[index].x + nln_re * etd_s;
        cpsi[index].y = exp_s * cpsi[index].y + nln_im * etd_s;

        cnlpsi[index].x = nln_re;
        cnlpsi[index].y = nln_im;                
    }

}

/************************************************************************************/
__global__ void etdrk2_step2(complex *cphi, complex *cnphi, complex *cnlphi, complex *cnlrk, complex *cpsi, complex *cnpsi, complex *cnlpsi, complex *cnlrk_psi, double *kx, double *ky)
{

    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = IDXC(index_x, index_y);
    double ksqr, exp_p, etd_c;
    double ph1, ph2;

    ph1 = 1.5 * lambda1 * sigma1 * epsilon;
    ph2 = 0.75 * lambda1 * (sigma1 / epsilon);

      double ps1, ps2, exp_s, etd_s;
      ps1 = 1.5 * lambda2 * sigma2 * epsilon;
      ps2 = 0.75 * lambda2 * (sigma2 / epsilon);

    if(index_x < Nx && index_y < Nyh)
    {

      ksqr = kx[index_x] * kx[index_x] + ky[index_y] * ky[index_y];

      etd_c = (-1.0) * ph1 * ksqr * ksqr;
      exp_p = exp(etd_c * dt);

      if(etd_c != 0.0)
      {
        etd_c = (exp_p - 1.0 - etd_c * dt) / (etd_c * etd_c * dt);
      }else{
             etd_c = 0.0;
           }

      double nln_re;
      double nln_im;

      if(ksqr > nalias)
      {
        cnphi[index].x = 0.0; cnphi[index].y = 0.0;
        cnlphi[index].x = 0.0; cnlphi[index].y = 0.0;
      }

      nln_re = (-1.0) * (cnphi[index].x + ph2 * ksqr * cnlphi[index].x);
      nln_im = (-1.0) * (cnphi[index].y + ph2 * ksqr * cnlphi[index].y);

      cphi[index].x = cphi[index].x + etd_c * (nln_re - cnlrk[index].x);
      cphi[index].y = cphi[index].y + etd_c * (nln_im - cnlrk[index].y);

      
        etd_s = (-1.0) * ps1 * ksqr * ksqr;
        exp_s = exp(etd_s * dt);

        if(etd_s != 0.0)
        {
          etd_s = (exp_s - 1.0 - etd_s * dt) / (etd_s * etd_s * dt);
        }else{
               etd_s = 0.0;
             }

        if(ksqr > nalias)
        {
          cnpsi[index].x = 0.0; cnpsi[index].y = 0.0;
          cnlpsi[index].x = 0.0; cnlpsi[index].y = 0.0;
        }

        nln_re = (-1.0) * (cnpsi[index].x + ps2 * ksqr * cnlpsi[index].x);
        nln_im = (-1.0) * (cnpsi[index].y + ps2 * ksqr * cnlpsi[index].y);

        cpsi[index].x = cpsi[index].x + etd_s * (nln_re - cnlrk_psi[index].x);
        cpsi[index].y = cpsi[index].y + etd_s * (nln_im - cnlrk_psi[index].y);
      
    }

}
/*****************************************************************************************/

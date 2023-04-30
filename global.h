#ifndef _GLOBAL_H
#define _GLOBAL_H
/**************************/
//#define FORCING
#define PHI_PSI
//#define TRACER
//#define INIT_READ_DATA
/**************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cufft.h>
#include <cuda_runtime.h>

#define Nx (256)
#define Ny (256)
#define Nyh (Ny/2+1)
#define pi (M_PI)
#define lx (2.0 * pi)
#define ly (2.0 * pi)
#define dx (lx / Nx)
#define dy (ly / Ny)

#ifdef PHI_PSI
  #define nalias (Nx * Nx / 16.0)
#else
  #define nalias (Nx * Nx / 9.0)
#endif
/***********************************/
#define nu (1.5)
#define dt (2e-3)
#define maxiter (4e4)
#define nfile (100)
#define uamp (0.0)
#define alpha (0.1)

#ifdef FORCING
  #define famp (0.0)
  #define kf (4)
#endif
#ifdef TRACER
  #define pnfile (15000)
  #define p_TPB (256)
  #define nparticles (1024*128)
#endif
#ifdef PHI_PSI
  #define epsilon (3.0*dx)

  #define sigma1 (0.0)
  #define lambda1 (0.0)

  #define sigma2 (1.0)
  #define lambda2 (5e-3)

  #define activity (0.0)
  #define beta (0.0)
#endif

/***********************************/
#define IDXR(i, j) (j + (Ny+2) * i)
#define IDXC(i, j) (j + Nyh * i)

#define NRANK (2)
#define BATCH (1)
#define TPBX (16)
#define TPBY (16)

typedef double2 complex;

#ifdef FORCING
__global__ void init(double *wz, double *fwz);
void init_vorticity(double *wz, double *fwz);
#else
__global__ void init(double *wz);
void init_vorticity(double *wz);
#endif

__global__ void ETD2RK_step2(complex *cwz, complex *cnln, complex *cnlp,
#ifdef FORCING
  complex *cfwz,
#endif
#ifdef PHI_PSI
  complex *csf,
#endif
double *kx, double *ky);

__global__ void ETD2RK_step1(complex *cwz, complex *cnln, 
#ifdef FORCING
  complex *cfwz,
#endif
#ifdef PHI_PSI
  complex *csf,
#endif
double *kx, double *ky);

__global__ void nonlin(double *w, double *ux, double *uy);
__global__ void divergence(complex *cux, complex *cuy, complex *cwz, double *kx, double *ky);
__global__ void velocity(complex *cwz, complex *cux, complex *cuy, double *kx, double *ky);
__global__ void particles_evolution(double *x, double *y, double *ux, double *uy);
__global__ void dealiasing(complex *w, double *kx, double *ky);
__global__ void init_kxky(double *kx, double *ky);
__global__ void normalize(double *w);

void cpu_particles_evolution(double *x, double *y, double *ux, double *uy);
void energy(double *ux, double *uy, double *eu);
void write_array_real(double *u, char filename[100]);
void read_array_real(double *u, char filename[100]);
void write_particle_positions(double *x, double *y, char filename[100]);

#ifdef PHI_PSI
//FILE *fp1, *fp2, *fp3;

void deformation(double *phi, double *ux, double *uy, FILE *fp1, FILE *fp2, FILE *fp3, double *t);

void host_init_phi_psi(double *phi, double *psi);

__global__ void etdrk2_step1(complex *cphi, complex *cnphi, complex *cnlphi,
complex *cpsi, complex *cnpsi, complex *cnlpsi, double *kx, double *ky);

__global__ void etdrk2_step2(complex *cphi, complex *cnphi, complex *cnlphi, complex *cnlrk, complex *cpsi, complex *cnpsi, complex *cnlpsi, complex *cnlrk_psi, double *kx, double *ky);

__global__ void nonlin_phi_psi(double *tphi, double *nphi, double *nlphi, double *ux, double *uy, double *tpsi, double *npsi, double *nlpsi);

__global__ void chemical_potential(complex *cphi, complex *cmu, complex *cgphi1, complex *cgphi2, complex *cpsi, complex *cmu_psi, complex *cgpsi1, complex *cgpsi2, double *kx, double *ky);

__global__ void surface_stress(double *phi, double *gphi1, double *gphi2
,double *psi, double *gpsi1, double *gpsi2
);

__global__ void curl(complex *cux, complex *cuy, complex *cwz, double *kx, double *ky);


#endif

#define cudaCheck(expr) {\
                            cudaError_t __cuda_error = expr;\
                            if((__cuda_error) != cudaSuccess)\
                            {\
                                printf("CUDA error on or before line number %d in file: %s. Error code: %d. Description: %s\n",\
                                        __LINE__, __FILE__, __cuda_error, cudaGetErrorString(__cuda_error));\
                                printf("Terminating execution\n");\
                                cudaDeviceReset();\
                                exit(0);\
                            }\
                        }

#ifdef _CUFFT_H_
#define cufftCheck(expr) {\
                            cufftResult __cufft_error = expr;\
                            if(__cufft_error != CUFFT_SUCCESS)\
                            {\
                                printf("cuFFT error on or before line number %d in file:%s. Error code: %d\n",\
                                        __LINE__, __FILE__, __cufft_error);\
                                printf("Terminating execution\n");\
                                cudaDeviceReset();\
                                exit(0);\
                            }\
                       }
#endif

#endif

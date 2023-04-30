#include "global.h"

    cufftHandle pfor, pinv;

void dfft(double *w, complex *cw)
{
    //gpuTimer.start();
    cufftCheck(cufftExecD2Z(pfor, w, cw));
    cudaDeviceSynchronize();
    //gpuTimer.stop();
    //gpuFFTTime += gpuTimer.elapsed();
}

void ifft(complex *cw, double *w)
{
    //puTimer.start();
    cufftCheck(cufftExecZ2D(pinv, cw, w));
    cudaDeviceSynchronize();
    //gpuTimer.stop();
    //gpuFFTTime += gpuTimer.elapsed();
}

__global__ void Add_matrix(complex *A, complex *B)
{
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_y = threadIdx.y + blockIdx.y * blockDim.y;

    int index = IDXC(index_x, index_y);

    if(index_x < Nx && index_y < Nyh)
    {
      A[index].x += B[index].x;
      A[index].y += B[index].y;
    }
}

//................................................................................

int main()
{
        system("mkdir -p vort");
        system("mkdir -p data");
        system("mkdir -p init_vort");
        #ifdef PHI_PSI
          system("mkdir -p phase_field");
        #endif

        #ifdef TRACER
          system("mkdir -p positions");
        #endif

        cudaSetDevice(0);
        double *wz, *ux, *uy, eu = 0.0;
        
        //int i, j, Idx;
//................................................
	double *dev_wz, *dev_twz, *dev_ux, *dev_uy, *kx, *ky;
        complex *dev_cwz, *dev_cux, *dev_cuy, *dev_tcwz, *dev_cnl;
        double *dev_eu;
        #ifdef FORCING
          double *dev_fwz, *fwz;
          complex *dev_cfwz;
        #endif

        #ifdef PHI_PSI
	  FILE *fp1, *fp2, *fp3;
          double *phi;
          double *dev_phi, *dev_tphi, *dev_nlphi, *dev_nphi;
          complex *dev_cphi, *dev_tcphi, *dev_cnlphi, *dev_cnphi;
          complex *dev_cnlrk, *dev_cwork;
      
            double *psi;
            double *dev_psi, *dev_tpsi, *dev_nlpsi, *dev_npsi;
            complex *dev_cpsi, *dev_tcpsi, *dev_cnlpsi, *dev_cnpsi;
            complex *dev_cnlrk_psi, *dev_cwork1;
        #endif

        int n[NRANK] = {Nx, Ny};

        cufftPlanMany(&pfor, NRANK, n, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z, BATCH);
        cufftPlanMany(&pinv, NRANK, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D, BATCH);
//................................................
        FILE *f1;

        f1 = fopen("data/ken.dat", "w");

	int size = Nx * Nyh * sizeof(complex);
//................................................

        cudaCheck(cudaMalloc((void **) &dev_cwz, size));
        cudaCheck(cudaMalloc((void **) &dev_tcwz, size));
        cudaCheck(cudaMalloc((void **) &dev_cux, size));
        cudaCheck(cudaMalloc((void **) &dev_cuy, size));
        cudaCheck(cudaMalloc((void **) &dev_cnl, size));
        #ifdef FORCING
          fwz = (double *) malloc(size);
          cudaCheck(cudaMalloc((void **) &dev_cfwz, size));
          dev_fwz = (double *) dev_cfwz;
        #endif

        #ifdef PHI_PSI

	  fp1 = fopen("data/cm_trajectory.dat", "w");
	  fp2 = fopen("data/cm_velocity.dat", "w");
	  fp3 = fopen("data/deform.dat", "w");

          phi = (double *) malloc(size);
          cudaCheck(cudaMalloc((void **) &dev_cphi, size));
          cudaCheck(cudaMalloc((void **) &dev_tcphi, size));
          cudaCheck(cudaMalloc((void **) &dev_cnlphi, size));
          cudaCheck(cudaMalloc((void **) &dev_cnphi, size));
          cudaCheck(cudaMalloc((void **) &dev_cnlrk, size));
          cudaCheck(cudaMalloc((void **) &dev_cwork, size));
          dev_phi = (double *) dev_cphi;
          dev_tphi = (double *) dev_tcphi;
          dev_nlphi = (double *) dev_cnlphi;
          dev_nphi = (double *) dev_cnphi;
          
            psi = (double *) malloc(size);
            cudaCheck(cudaMalloc((void **) &dev_cpsi, size));
            cudaCheck(cudaMalloc((void **) &dev_tcpsi, size));
            cudaCheck(cudaMalloc((void **) &dev_cnlpsi, size));
            cudaCheck(cudaMalloc((void **) &dev_cnpsi, size));
            cudaCheck(cudaMalloc((void **) &dev_cnlrk_psi, size));
            cudaCheck(cudaMalloc((void **) &dev_cwork1, size));
            dev_psi = (double *) dev_cpsi;
            dev_tpsi = (double *) dev_tcpsi;
            dev_nlpsi = (double *) dev_cnlpsi;
            dev_npsi = (double *) dev_cnpsi;
        #endif
        cudaCheck(cudaMalloc((void **) &dev_eu,  sizeof(double)));
        cudaCheck(cudaMalloc((void **) &kx, Nx * sizeof(double)));
        cudaCheck(cudaMalloc((void **) &ky, Nyh * sizeof(double)));
        dev_wz = (double *) dev_cwz;
        dev_ux = (double *) dev_cux;
        dev_uy = (double *) dev_cuy;
        dev_twz = (double *) dev_tcwz;
//................................................
	wz = (double *) malloc(size);
        ux = (double *) malloc(size);
        uy = (double *) malloc(size);
        #ifdef TRACER
          double *xp, *yp, *dev_xp, *dev_yp;
	  xp = (double *) malloc(nparticles * sizeof(double));
          yp = (double *) malloc(nparticles * sizeof(double));
          cudaCheck(cudaMalloc((void **) &dev_xp, nparticles * sizeof(double)));
          cudaCheck(cudaMalloc((void **) &dev_yp, nparticles * sizeof(double)));
        #endif
        dim3 dimBlock(TPBX, TPBY);
        dim3 dimGrid((int) ceil(Nx/dimBlock.x), (int) ceil(Ny/dimBlock.y));

        init_kxky<<<dimGrid, dimBlock>>>(kx, ky);

        #ifdef FORCING
	 // init<<<dimGrid, dimBlock>>>(dev_wz, dev_fwz);
         // dfft(dev_fwz, dev_cfwz);
         init_vorticity(wz, fwz);
         cudaCheck(cudaMemcpy(dev_wz, wz, size, cudaMemcpyHostToDevice));
         cudaCheck(cudaMemcpy(dev_fwz, fwz, size, cudaMemcpyHostToDevice));
         dfft(dev_fwz, dev_cfwz);
        #else
         init_vorticity(wz);
         cudaCheck(cudaMemcpy(dev_wz, wz, size, cudaMemcpyHostToDevice));
         // init<<<dimGrid, dimBlock>>>(dev_wz);
        #endif
 
        dfft(dev_wz, dev_cwz);
        dealiasing<<<dimGrid, dimBlock>>>(dev_cwz, kx, ky);

        #ifdef PHI_PSI
            host_init_phi_psi(phi, psi);
            cudaCheck(cudaMemcpy(dev_phi, phi, size, cudaMemcpyHostToDevice));
            cudaCheck(cudaMemcpy(dev_psi, psi, size, cudaMemcpyHostToDevice));
            dfft(dev_phi, dev_cphi);
            dfft(dev_psi, dev_cpsi);
            dealiasing<<<dimGrid, dimBlock>>>(dev_cphi, kx, ky);
            dealiasing<<<dimGrid, dimBlock>>>(dev_cpsi, kx, ky);
        #endif
//.............................................................................
    #ifdef TRACER
      int p_nblocks = (int) ceil(nparticles / p_TPB);

      srand((unsigned)time(NULL));
      double rn, rn1, rn2, theta;
      int seed = 1;
      srand(seed);

      for (int ip = 0; ip < nparticles; ip++)
      {

        rn = (double) rand() / (double)(RAND_MAX);
        rn1 = (double) rand() / (double)(RAND_MAX);
        rn2 = (double) rand() / (double)(RAND_MAX);

        theta = 2.0 * pi * rn;
        
        xp[ip] = pi + (pi/5.5) * rn1 * cos(theta);
        yp[ip] = pi + (pi/5.5) * rn2 * sin(theta);
        
      }
      cudaCheck(cudaMemcpy(dev_xp, xp, nparticles * sizeof(double), cudaMemcpyHostToDevice));
      cudaCheck(cudaMemcpy(dev_yp, yp, nparticles * sizeof(double), cudaMemcpyHostToDevice));
    #endif
/******************************Time Stepping**********************************/
    char filename[100];
    double t = 0.0;
    int findex = 0;// fpindex = 0;
    int navg = (maxiter/nfile);
    //int pnavg = (maxiter/pnfile);


    for(int k = 0; k < (maxiter+1); k++)
    {
        velocity<<<dimGrid, dimBlock>>>(dev_cwz, dev_cux, dev_cuy, kx, ky);
        cudaCheck(cudaMemcpy(dev_tcwz, dev_cwz, size, cudaMemcpyDeviceToDevice));
/*****************************************************************************/
        ifft(dev_tcwz, dev_twz);
        ifft(dev_cux, dev_ux);
        ifft(dev_cuy, dev_uy);
        normalize<<<dimGrid, dimBlock>>>(dev_twz);
        normalize<<<dimGrid, dimBlock>>>(dev_ux);
        normalize<<<dimGrid, dimBlock>>>(dev_uy);

        #ifdef TRACER
          if(k % pnavg == 0)
          {
            cudaCheck(cudaMemcpy(xp, dev_xp, nparticles * sizeof(double), cudaMemcpyDeviceToHost));
            cudaCheck(cudaMemcpy(yp, dev_yp, nparticles * sizeof(double), cudaMemcpyDeviceToHost));

            sprintf(filename, "positions/xy.%d", fpindex);
            write_particle_positions(xp, yp, filename);

            fpindex++;
          }
        /*
          for(int ip = 0; ip < nparticles; ip++)
          {
            printf("%f \t %lf \t %lf\n", t, xp[ip], yp[ip]);
          }
        */
          //cpu_particles_evolution(xp, yp, ux, uy);
          particles_evolution<<<p_nblocks, p_TPB>>>(dev_xp, dev_yp, dev_ux, dev_uy);
        #endif

        #ifndef PHI_PSI
          if(k % navg == 0)
          {
            cudaCheck(cudaMemcpy(wz, dev_twz, size, cudaMemcpyDeviceToHost));
            sprintf(filename, "vort/wz.%d", findex);
            write_array_real(wz, filename);

            findex++;
          }          
        #endif
/*.....................Writing energy at each time step.......................*/
        eu = 0.0;
        cudaCheck(cudaMemcpy(ux, dev_ux, size, cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(uy, dev_uy, size, cudaMemcpyDeviceToHost));
        energy(ux, uy, &eu);

        if(eu > 100) abort();

        fprintf(f1, "%lf \t %.12lf\n", t, eu);
/*............................................................................*/
        #ifdef PHI_PSI
          dealiasing<<<dimGrid, dimBlock>>>(dev_cphi, kx, ky);
          cudaCheck(cudaMemcpy(dev_tcphi, dev_cphi, size, cudaMemcpyDeviceToDevice));
          cudaCheck(cudaMemcpy(dev_cwork, dev_cphi, size, cudaMemcpyDeviceToDevice));
          ifft(dev_tcphi, dev_tphi);
          normalize<<<dimGrid, dimBlock>>>(dev_tphi);

/******************************************************************************/

	  if(k % navg == 0)
          {
            cudaCheck(cudaMemcpy(phi, dev_tphi, size, cudaMemcpyDeviceToHost));
            sprintf(filename, "phase_field/phi.%d", findex);
            write_array_real(phi, filename);
            //deformation(phi, ux, uy, fp1, fp2, fp3, &t);	  
	  }


/******************************************************************************/	  

       	    dealiasing<<<dimGrid, dimBlock>>>(dev_cpsi, kx, ky);
            cudaCheck(cudaMemcpy(dev_tcpsi, dev_cpsi, size, cudaMemcpyDeviceToDevice));
            cudaCheck(cudaMemcpy(dev_cwork1, dev_cpsi, size, cudaMemcpyDeviceToDevice));
            ifft(dev_tcpsi, dev_tpsi);
            normalize<<<dimGrid, dimBlock>>>(dev_tpsi);

          if(k % navg == 0)
          {
            //cudaCheck(cudaMemcpy(phi, dev_tphi, size, cudaMemcpyDeviceToHost));
            //sprintf(filename, "phase_field/phi.%d", findex);
            //write_array_real(phi, filename);

            cudaCheck(cudaMemcpy(psi, dev_tpsi, size, cudaMemcpyDeviceToHost));
            sprintf(filename, "phase_field/psi.%d", findex);
            write_array_real(psi, filename);

            cudaCheck(cudaMemcpy(wz, dev_twz, size, cudaMemcpyDeviceToHost));
            sprintf(filename, "vort/wz.%d", findex);
            write_array_real(wz, filename);

            findex++;
          }
          
          nonlin_phi_psi<<<dimGrid, dimBlock>>>(dev_tphi, dev_nphi, dev_nlphi, dev_ux, 
                                                dev_uy,dev_tpsi, dev_npsi, dev_nlpsi);

          dfft(dev_tphi, dev_tcphi);
          dfft(dev_nphi, dev_cnphi);
          divergence<<<dimGrid, dimBlock>>>(dev_tcphi, dev_cnphi, dev_cnphi, kx, ky);
          dfft(dev_nlphi, dev_cnlphi);

          dfft(dev_tpsi, dev_tcpsi);
          dfft(dev_npsi, dev_cnpsi);
          divergence<<<dimGrid, dimBlock>>>(dev_tcpsi, dev_cnpsi, dev_cnpsi, kx, ky);
          dfft(dev_nlpsi, dev_cnlpsi);

          etdrk2_step1<<<dimGrid, dimBlock>>>(dev_cphi, dev_cnphi, dev_cnlphi, 
                                              dev_cpsi, dev_cnpsi, dev_cnlpsi, kx, ky);
          cudaCheck(cudaMemcpy(dev_cnlrk, dev_cnlphi, size, cudaMemcpyDeviceToDevice));
          cudaCheck(cudaMemcpy(dev_cnlrk_psi, dev_cnlpsi, size, cudaMemcpyDeviceToDevice));

          chemical_potential<<<dimGrid, dimBlock>>>(dev_cwork, dev_tcphi, dev_cnphi, 
                    dev_cnlphi, dev_cwork1, dev_tcpsi, dev_cnpsi, dev_cnlpsi, kx, ky);

          ifft(dev_tcphi, dev_tphi);
          ifft(dev_cnphi, dev_nphi);
          ifft(dev_cnlphi, dev_nlphi);
          normalize<<<dimGrid, dimBlock>>>(dev_tphi);
          normalize<<<dimGrid, dimBlock>>>(dev_nphi);
          normalize<<<dimGrid, dimBlock>>>(dev_nlphi);
/************************************************************
          tphi contain chemical potential, mu.
          nphi = del_x(phi); nlphi = del_y(phi);
*************************************************************/
          
            ifft(dev_tcpsi, dev_tpsi);
            ifft(dev_cnpsi, dev_npsi);
            ifft(dev_cnlpsi, dev_nlpsi);
            normalize<<<dimGrid, dimBlock>>>(dev_tpsi);
            normalize<<<dimGrid, dimBlock>>>(dev_npsi);
            normalize<<<dimGrid, dimBlock>>>(dev_nlpsi);

            surface_stress<<<dimGrid, dimBlock>>>(dev_tphi, dev_nphi, dev_nlphi,
                                                  dev_tpsi, dev_npsi, dev_nlpsi);

            dfft(dev_npsi, dev_cnpsi);
            dfft(dev_nlpsi, dev_cnlpsi);
            curl<<<dimGrid, dimBlock>>>(dev_cnpsi, dev_cnlpsi, dev_cnlpsi, kx, ky);
            
/*************************************************************
          cnlphi = curl (mu \grad phi)
**************************************************************/
        #endif//...PHI_PSI......

/*********************************************************************************/
        nonlin<<<dimGrid, dimBlock>>>(dev_twz, dev_ux, dev_uy);
        dfft(dev_ux, dev_cux);
        dfft(dev_uy, dev_cuy);
        divergence<<<dimGrid, dimBlock>>>(dev_cux, dev_cuy, dev_tcwz, kx, ky);
//............................................................................
        #ifdef PHI_PSI
          #ifdef FORCING
            ETD2RK_step1<<<dimGrid, dimBlock>>>(dev_cwz, dev_tcwz, dev_cfwz, dev_cnlpsi, kx, ky);
          #else
            ETD2RK_step1<<<dimGrid, dimBlock>>>(dev_cwz, dev_tcwz, dev_cnlpsi, kx, ky);
          #endif
        #else
          #ifdef FORCING
            ETD2RK_step1<<<dimGrid, dimBlock>>>(dev_cwz, dev_tcwz, dev_cfwz, kx, ky);
          #else
            ETD2RK_step1<<<dimGrid, dimBlock>>>(dev_cwz, dev_tcwz, kx, ky);
          #endif
        #endif

        cudaCheck(cudaMemcpy(dev_cnl, dev_tcwz, size, cudaMemcpyDeviceToDevice));

///***************************Step-2 of RK-2************************************/
///***************************Step-2 of RK-2************************************/
///***************************Step-2 of RK-2************************************/
        velocity<<<dimGrid, dimBlock>>>(dev_cwz, dev_cux, dev_cuy, kx, ky);
        cudaCheck(cudaMemcpy(dev_tcwz, dev_cwz, size, cudaMemcpyDeviceToDevice));
///*****************************************************************************/
        ifft(dev_tcwz, dev_twz);
        ifft(dev_cux, dev_ux);
        ifft(dev_cuy, dev_uy);
        normalize<<<dimGrid, dimBlock>>>(dev_twz);
        normalize<<<dimGrid, dimBlock>>>(dev_ux);
        normalize<<<dimGrid, dimBlock>>>(dev_uy);

        #ifdef PHI_PSI
          dealiasing<<<dimGrid, dimBlock>>>(dev_cphi, kx, ky);
          cudaCheck(cudaMemcpy(dev_tcphi, dev_cphi, size, cudaMemcpyDeviceToDevice));
          cudaCheck(cudaMemcpy(dev_cwork, dev_tcphi, size, cudaMemcpyDeviceToDevice));
          ifft(dev_tcphi, dev_tphi);
          normalize<<<dimGrid, dimBlock>>>(dev_tphi);

          dealiasing<<<dimGrid, dimBlock>>>(dev_cpsi, kx, ky);
          cudaCheck(cudaMemcpy(dev_tcpsi, dev_cpsi, size, cudaMemcpyDeviceToDevice));
          cudaCheck(cudaMemcpy(dev_cwork1, dev_tcpsi, size, cudaMemcpyDeviceToDevice));
          ifft(dev_tcpsi, dev_tpsi);
          normalize<<<dimGrid, dimBlock>>>(dev_tpsi);

          nonlin_phi_psi<<<dimGrid, dimBlock>>>(dev_tphi, dev_nphi, dev_nlphi, dev_ux, 
                                            dev_uy,dev_tpsi, dev_npsi, dev_nlpsi);


          dfft(dev_tphi, dev_tcphi);
          dfft(dev_nphi, dev_cnphi);
          divergence<<<dimGrid, dimBlock>>>(dev_tcphi, dev_cnphi, dev_cnphi, kx, ky);
          dfft(dev_nlphi, dev_cnlphi);
          cudaCheck(cudaMemcpy(dev_tcphi, dev_cnlphi, size, cudaMemcpyDeviceToDevice));

            dfft(dev_tpsi, dev_tcpsi);
            dfft(dev_npsi, dev_cnpsi);
            divergence<<<dimGrid, dimBlock>>>(dev_tcpsi, dev_cnpsi, dev_cnpsi, kx, ky);
            dfft(dev_nlpsi, dev_cnlpsi);
            cudaCheck(cudaMemcpy(dev_tcpsi, dev_cnlpsi, size, cudaMemcpyDeviceToDevice));

            etdrk2_step2<<<dimGrid, dimBlock>>>(dev_cphi, dev_cnphi, dev_cnlphi, dev_cnlrk,
                               dev_cpsi, dev_cnpsi, dev_cnlpsi, dev_cnlrk_psi, kx, ky);
            chemical_potential<<<dimGrid, dimBlock>>>(dev_cwork, dev_tcphi, dev_cnphi, 
                        dev_cnlphi, dev_cwork1, dev_tcpsi, dev_cnpsi, dev_cnlpsi, kx, ky);

          ifft(dev_tcphi, dev_tphi);
          ifft(dev_cnphi, dev_nphi);
          ifft(dev_cnlphi, dev_nlphi);
          normalize<<<dimGrid, dimBlock>>>(dev_tphi);
          normalize<<<dimGrid, dimBlock>>>(dev_nphi);
          normalize<<<dimGrid, dimBlock>>>(dev_nlphi);

            ifft(dev_tcpsi, dev_tpsi);
            ifft(dev_cnpsi, dev_npsi);
            ifft(dev_cnlpsi, dev_nlpsi);
            normalize<<<dimGrid, dimBlock>>>(dev_tpsi);
            normalize<<<dimGrid, dimBlock>>>(dev_npsi);
            normalize<<<dimGrid, dimBlock>>>(dev_nlpsi);

            surface_stress<<<dimGrid, dimBlock>>>(dev_tphi, dev_nphi, dev_nlphi,
                                                  dev_tpsi, dev_npsi, dev_nlpsi);

            dfft(dev_npsi, dev_cnpsi);
            dfft(dev_nlpsi, dev_cnlpsi);

            curl<<<dimGrid, dimBlock>>>(dev_cnpsi, dev_cnlpsi, dev_cnlpsi, kx, ky);

        #endif //...PHI_PSI.......
        
///*****************************************************************************/
        nonlin<<<dimGrid, dimBlock>>>(dev_twz, dev_ux, dev_uy);
        dfft(dev_ux, dev_cux);
        dfft(dev_uy, dev_cuy);
        divergence<<<dimGrid, dimBlock>>>(dev_cux, dev_cuy, dev_tcwz, kx, ky);
///*****************************************************************************/
        #ifdef PHI_PSI
          #ifdef FORCING
            ETD2RK_step2<<<dimGrid, dimBlock>>>(dev_cwz, dev_tcwz, dev_cnl, 
                                                dev_cfwz, dev_cnlpsi, kx, ky);
          #else
            ETD2RK_step2<<<dimGrid, dimBlock>>>(dev_cwz, dev_tcwz, dev_cnl, 
                                                dev_cnlpsi, kx, ky);
          #endif
        #else
          #ifdef FORCING
            ETD2RK_step2<<<dimGrid, dimBlock>>>(dev_cwz, dev_tcwz, dev_cnl, 
                                                dev_cfwz, kx, ky);
          #else
            ETD2RK_step2<<<dimGrid, dimBlock>>>(dev_cwz, dev_tcwz, dev_cnl, kx, ky);
          #endif     
        #endif
///*****************************************************************************/
        t = t + dt;
    }
/**********************End of the time step***********************************/
/*************************************************************************************/
        cufftDestroy(pfor); cufftDestroy(pinv);

	cudaFree(dev_cwz); cudaFree(kx); cudaFree(ky);
        cudaFree(dev_tcwz); cudaFree(dev_cux); 
        cudaFree(dev_cuy); cudaFree(dev_cnl);
        #ifdef FORCING
          cudaFree(dev_cfwz); free(fwz);
        #endif

        #ifdef PHI_PSI

	  fclose(fp1); fclose(fp2); fclose(fp3);

          free(phi);
          cudaFree(dev_cphi); cudaFree(dev_tcphi);
          cudaFree(dev_cnlphi); cudaFree(dev_cnphi);
          cudaFree(dev_cnlrk); cudaFree(dev_cwork);
          //#ifdef ADD_SURFACTANTS
            free(psi);
            cudaFree(dev_cpsi); cudaFree(dev_tcphi);
            cudaFree(dev_cnlpsi); cudaFree(dev_cnphi);
            cudaFree(dev_cnlrk_psi); cudaFree(dev_cwork1);
          //#endif
        #endif
        free(wz); free(ux); free(uy);
        #ifdef TRACER
          free(xp); free(yp);
          cudaFree(dev_xp); cudaFree(dev_yp);
        #endif
        fclose(f1);
/*************************************************************************************/
return 0;
}

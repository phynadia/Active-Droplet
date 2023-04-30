CC = nvcc -Wno-deprecated-gpu-targets #-arch=sm_60
#CFLAGS = -O3 -arch sm_20 -gencode arch=compute_20,code=sm_20 -lineinfo
LIBS= -lm -lcufft -lineinfo #-lfftw3 -L=/home/nadia/software/fftw3/lib

all: chns2d.exe

%.o : %.cu
	$(info --- Building '$@' from '$<' using pattern rule for .cu -> .out)
	$(CC) $(CFLAGS) -c $@ $<

chns2d.exe: main.o init.o nonlin.o divergence.o etd_rk2.o velocity.o particles.o surface_tension.o functions.o
	$(CC) -o $@ $+ $(LIBS)


clean:
	$(RM) *.o *.exe a.out

scrub: clean
	$(RM) *.dat


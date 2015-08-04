#include<stdio.h>
#include<malloc.h>
#include<math.h>

// CUDA Kernel //

void __global__ eval(double* a,double b,double* c,int size) {
  for(int i=0;i<size;i++) {
    c[i]=a[i]*b;
  }
}


// CPU code //
int main() {

  srand48(time(NULL));

  // define vector size //
  //const int vector_size = 2560000;
  const int vector_size = 10000000;

  // allocate memory on cpu //
  double* vector_a_cpu;
  double* vector_c_cpu;
  cudaMallocHost((void**)&vector_a_cpu,
                 vector_size*sizeof(double));
  cudaMallocHost((void**)&vector_c_cpu,
                 vector_size*sizeof(double));

  double b=drand48();
  // initialize cpu input vector //
  for(int i=0;i<vector_size;i++) {
    vector_a_cpu[i]=drand48();
  }
  // initialize cpu output vector to 0 //
  for(int i=0;i<vector_size;i++) {
    vector_c_cpu[i]=0;
  }

  // allocate memory on gpu //
  double* vector_a_gpu;
  double* vector_c_gpu;
  cudaMalloc((void**)&vector_a_gpu,vector_size*sizeof(double));
  cudaMalloc((void**)&vector_c_gpu,(vector_size)*sizeof(float));

  // copy input vector to gpu //
  cudaMemcpy(vector_a_gpu,vector_a_cpu,
             vector_size*sizeof(double),cudaMemcpyHostToDevice);
  // initialize gpu output vector to 0 //
  cudaMemcpy(vector_c_gpu,vector_c_cpu,
             (vector_size)*sizeof(double),cudaMemcpyHostToDevice);

  // cuda kernel call //
  eval<<<1,1>>>(vector_a_gpu,b,vector_c_gpu,vector_size);

  // copy output vector from gpu to cpu //
  cudaMemcpy(vector_c_cpu,vector_c_gpu,
             (vector_size)*sizeof(double),cudaMemcpyDeviceToHost);

  // verify results //
  bool success=true;
  for(int i=0;i<vector_size;i++) {
    if (fabsf(vector_a_cpu[i]*b-vector_c_cpu[i])>1e-5*b) {
      printf("computation result is wrong: index=%i expected result=%g "
             "cuda result=%g\n", i, vector_a_cpu[i]*b, vector_c_cpu[i]);
      success=false;
      break;
    }
  }
  if (success) {
    printf("computation result is correct.\n");
  }

  // free gpu memory //
  cudaFree(vector_a_gpu);
  cudaFree(vector_c_gpu);

  // free cpu memory //
  cudaFreeHost(vector_a_cpu);
  cudaFreeHost(vector_c_cpu);

  // exit //
  return 0;
}

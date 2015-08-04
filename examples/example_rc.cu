#include <cuda.h>
#include <stdio.h>

__global__ void kernel(float *a) {
   a[0] = a[0]*a[0];
}

int main() {
   float *data[2];
   float *deviceData[2];
   cudaStream_t stream[2];

   srand(time(NULL));

   // Allocate memory on host and device, set host data and initialize streams
   for ( int i = 0 ; i < 2 ; i++ ) {
      cudaMallocHost(&(data[i]), sizeof(float));
      cudaMalloc(&(deviceData[i]), sizeof(float));
      *(data[i]) = (rand() % 10) + (float)i+3;
      cudaStreamCreate(&stream[i]);
   }

   // Print initial data
   for ( int i = 0 ; i < 2 ; i++ ) {
      printf("%f ", *(data[i]));
   }
   printf("\n");

   // Asynchronous copy of data, kernel execution and copying back of data
   for ( int i = 0 ; i < 2 ; i++ ) {
      cudaMemcpyAsync(deviceData[0], data[i], sizeof(float), cudaMemcpyHostToDevice, stream[i]);
      kernel<<<1, 1, 0, stream[i]>>>(deviceData[i]);
      cudaMemcpyAsync(data[i], deviceData[i], sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
   }

   // Sync streams and print results
   for ( int i = 0 ; i < 2 ; i++ ) {
      cudaStreamSynchronize(stream[i]);
      printf("%f ", *(data[i]));
   }
   printf("\n");
   
   return 0;
}




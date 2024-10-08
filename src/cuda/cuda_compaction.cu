#include "moana/cuda/cuda_compaction.hpp"
#include <string>

namespace moana {

#define SCAN_BLOCKSIZE 1024

__global__ void preKernel(int size, int worldID, int *d_tempScanBuffer, int *d_scanInBuffer, WavefrontPathData *d_pathDataBuffer) {

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    int threadIndex = x + (y * gridDim.x * blockDim.x);

    if(threadIndex >= size) return;
    d_tempScanBuffer[threadIndex] = 0;

    if(d_pathDataBuffer[threadIndex].isValid) {
        d_tempScanBuffer[threadIndex] = d_pathDataBuffer[threadIndex].targetNode == worldID ? 1 : 0;
        d_scanInBuffer[threadIndex] = d_tempScanBuffer[threadIndex];
    }
}

__global__ void postKernel(int size, int worldID, int *d_transferOffset,int *d_scanInBuffer, int *d_scanResultBuffer, WavefrontPathData *d_transferPathDataBuffer, WavefrontPathData *d_pathDataBuffer) {

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    int threadIndex = x + (y * gridDim.x * blockDim.x);

    if(threadIndex == 0) d_transferOffset[worldID + 1] = d_scanResultBuffer[size - 1] + d_transferOffset[worldID];

    if(threadIndex < size && d_scanInBuffer[threadIndex] == 1)
        d_transferPathDataBuffer[d_transferOffset[worldID] + d_scanResultBuffer[threadIndex] - 1] = d_pathDataBuffer[threadIndex];
}

__global__ void Hillis_Steele_Scan_Kernel(int size, int *d_out, int *d_in, int space, int step, int steps) {

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    // 2D Kernel Launch parameters

    int threadIndex = x + (y * gridDim.x * blockDim.x);

    if(threadIndex >= size) return;

    if (threadIndex >= space) d_out[threadIndex] += d_in[threadIndex - space];
}


void Hillis_Steele_Scan(Renderer::Params &params, int width, int height, int worldSize) {

    int N = params.pathSize;

    // std::vector<int> scanResult(width * height);
    // std::vector<int> transferOffset(worldSize + 1);
    // // cudaMemcpy(scanResult.data(), params.scanInBuffer, width * height * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(transferOffset.data(), params.transferOffset, (worldSize + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    // for(int i = 0; i < transferOffset.size(); i++) {
    //     std::cout << transferOffset[i] << std::endl;
    // }

    // 2D Kernel Launch Parameters

    dim3 THREADS(1024, 1, 1);

    dim3 BLOCKS;

    if (N >= 65536)

        BLOCKS = dim3(64, (N + 65536 - 1) / 65536, 1);

    else if (N <= 1024)

        BLOCKS = dim3(1, 1, 1);

    else

        BLOCKS = dim3((N + 1024 - 1) / 1024, 1, 1);
    
    for (int worldID = 0; worldID < worldSize; worldID++) {

        preKernel<<<BLOCKS, THREADS>>>(N, worldID, params.tempScanBuffer, params.scanInBuffer, params.pathDataBuffer);

        int space = 1;

        // Begin with a stride of 2^0

        int steps = int(log2(float(N + 1.f)));
        // std::cout << "steps: "  << steps << std::endl;

        // Log2N depth dependency of scan

        cudaMemcpy(params.scanResultBuffer, params.tempScanBuffer, sizeof(int) * N, cudaMemcpyDeviceToDevice);

        // Copy Input Array to Output Array

        for (size_t step = 0; step <= steps; step++) {

            Hillis_Steele_Scan_Kernel<<<BLOCKS, THREADS>>>(N, params.scanResultBuffer, params.tempScanBuffer, space, step, steps);

            // Calls the parallel operation
            cudaMemcpy(params.tempScanBuffer, params.scanResultBuffer, sizeof(int) * N, cudaMemcpyDeviceToDevice);

            space *= 2;
        }

        postKernel<<<BLOCKS, THREADS>>>(N, worldID, params.transferOffset, params.scanInBuffer, params.scanResultBuffer, params.transferPathDataBuffer, params.pathDataBuffer);

        cudaDeviceSynchronize();
    }

    // std::vector<float> tempImage(width * height * 3);
    // for (int row = 0; row < height; row++) {
    //     for (int col = 0; col < width; col++) {
    //         int index = row * width + col;
    //         int pixelIndex = 3 * index;
    //         float temp = scanResult[index];
    //         tempImage[pixelIndex] =     temp;
    //         tempImage[pixelIndex + 1] = temp;
    //         tempImage[pixelIndex + 2] = temp;
    //     }  
    // }
    // Image::save(//Debug
    //     width,
    //     height,
    //     tempImage,
    //     "scanInBuffer"+ std::to_string(transferOffset.back()) +".exr"
    // );
    // std::cout << scanResult[N - 1] << std::endl;
    // cudaMemcpy(transferOffset.data(), params.transferOffset, (worldSize + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    // for(int i = 0; i < transferOffset.size(); i++) {
    //     std::cout << transferOffset[i] << std::endl;
    // }
}

__global__ void preKernelNN(int size, int AABBID, int *d_tempScanBuffer, int *d_scanInBuffer, NNPathData *d_NNPathDataBuffer) {

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    int threadIndex = x + (y * gridDim.x * blockDim.x);

    if(threadIndex >= size) return;
    d_tempScanBuffer[threadIndex] = 0;

    d_tempScanBuffer[threadIndex] = d_NNPathDataBuffer[threadIndex].hitAABBID == AABBID ? 1 : 0;
    d_scanInBuffer[threadIndex] = d_tempScanBuffer[threadIndex];
}

__global__ void preKernelNN_HIT_INSIDE(int size, int AABBID, int *d_tempScanBuffer, int *d_scanInBuffer, NNPathData *d_NNPathDataBuffer) {

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    int threadIndex = x + (y * gridDim.x * blockDim.x);

    if(threadIndex >= size) return;
    d_tempScanBuffer[threadIndex] = 0;

    d_tempScanBuffer[threadIndex] = (d_NNPathDataBuffer[threadIndex].hitAABBID == AABBID && d_NNPathDataBuffer[threadIndex].isInside) ? 1 : 0;
    d_scanInBuffer[threadIndex] = d_tempScanBuffer[threadIndex];
}

__global__ void postKernelNN(
    int size, 
    int AABBID, 
    int *d_sceneOffset,
    int *d_scanInBuffer, 
    int *d_scanResultBuffer, 
    NN_Float *d_packedInputDataBuffer, 
    NN_Float *d_inputDataBuffer,
    NNPathData *d_packedNNPathDataBuffer,
    NNPathData *d_NNPathDataBuffer
) {

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    int threadIndex = x + (y * gridDim.x * blockDim.x);

    if(threadIndex == 0) d_sceneOffset[AABBID + 1] = d_scanResultBuffer[size - 1] + d_sceneOffset[AABBID];

    if(threadIndex < size && d_scanInBuffer[threadIndex] == 1) {
        int intersectionID = (d_sceneOffset[AABBID] + d_scanResultBuffer[threadIndex] - 1) * 5;

        d_packedInputDataBuffer[intersectionID + 0] = d_inputDataBuffer[threadIndex * 5 + 0];
        d_packedInputDataBuffer[intersectionID + 1] = d_inputDataBuffer[threadIndex * 5 + 1];
        d_packedInputDataBuffer[intersectionID + 2] = d_inputDataBuffer[threadIndex * 5 + 2];
        d_packedInputDataBuffer[intersectionID + 3] = d_inputDataBuffer[threadIndex * 5 + 3];
        d_packedInputDataBuffer[intersectionID + 4] = d_inputDataBuffer[threadIndex * 5 + 4];

        d_packedNNPathDataBuffer[d_sceneOffset[AABBID] + d_scanResultBuffer[threadIndex] - 1] = d_NNPathDataBuffer[threadIndex];
    }
}

void Hillis_Steele_Scan_for_NN(Renderer::Params &params, int width, int height) {

    int N = params.maxCount * params.pathSize;

    int *d_tempScanBuffer = params.tempScanNNBuffer;
    int *d_scanInBuffer = params.scanInNNBuffer;
    int *d_scanResultBuffer = params.scanResultNNBuffer;
    int *d_sceneOffset = params.sceneOffset;
    // std::vector<float> scanResult(N * 5);
    // std::vector<int> transferOffset(params.sceneSize + 1);
    // // cudaMemcpy(scanResult.data(), params.scanInBuffer, width * height * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(transferOffset.data(), params.sceneOffset, (params.sceneSize + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    // for(int i = 0; i < transferOffset.size(); i++) {
    //     std::cout << transferOffset[i] << std::endl;
    // }

    // 2D Kernel Launch Parameters

    dim3 THREADS(1024, 1, 1);

    dim3 BLOCKS;

    if (N >= 65536)

        BLOCKS = dim3(64, (N + 65536 - 1) / 65536, 1);

    else if (N <= 1024)

        BLOCKS = dim3(1, 1, 1);

    else

        BLOCKS = dim3((N + 1024 - 1) / 1024, 1, 1);

    int AABBID = 0;
    for (int i = 0; i < params.sceneSize; i++) { 

        AABBID = i + 1; //TODO AABBID + 1

        preKernelNN<<<BLOCKS, THREADS>>>(N, AABBID, d_tempScanBuffer, d_scanInBuffer, params.NNPathDataBuffer);

        int space = 1;

        // Begin with a stride of 2^0

        int steps = int(log2(float(N + 1.f)));
        // std::cout << "steps: "  << steps << std::endl;

        // Log2N depth dependency of scan

        cudaMemcpy(d_scanResultBuffer, d_tempScanBuffer, sizeof(int) * N, cudaMemcpyDeviceToDevice);

        // Copy Input Array to Output Array

        for (size_t step = 0; step <= steps; step++) {

            Hillis_Steele_Scan_Kernel<<<BLOCKS, THREADS>>>(N, d_scanResultBuffer, d_tempScanBuffer, space, step, steps);

            // Calls the parallel operation
            cudaMemcpy(d_tempScanBuffer, d_scanResultBuffer, sizeof(int) * N, cudaMemcpyDeviceToDevice);

            space *= 2;
        }

        postKernelNN<<<BLOCKS, THREADS>>>(N, i, d_sceneOffset, d_scanInBuffer, d_scanResultBuffer, params.packedInputDataBuffer, params.inputDataBuffer, params.packedNNPathDataBuffer, params.NNPathDataBuffer);

        cudaDeviceSynchronize();
    }

    // cudaMemcpy(scanResult.data(), params.inputDataBuffer, N * 5 * sizeof(float), cudaMemcpyDeviceToHost);
    // std::vector<float> tempImage(width * height * 3);
    // for (int row = 0; row < height; row++) {
    //     for (int col = 0; col < width; col++) {
    //         int index = row * width + col;
    //         int pixelIndex = 3 * index;
    //         tempImage[pixelIndex]     = scanResult[index * 5 + 3];
    //         tempImage[pixelIndex + 1] = scanResult[index * 5 + 4];
    //         tempImage[pixelIndex + 2] = scanResult[index * 5 + 4];
    //     }  
    // }
    // Image::save(//Debug
    //     width,
    //     height,
    //     tempImage,
    //     "packedInputDataBuffer.exr"
    // );
    // std::cout << scanResult[N - 1] << std::endl;
    // cudaMemcpy(transferOffset.data(), params.sceneOffset, (params.sceneSize + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    // for(int i = 0; i < transferOffset.size(); i++) {
    //     std::cout << transferOffset[i] << std::endl;
    // }
}

//==============================================================================
//optimization cuda kernel

void scanCPU(int n, int *odata, const int *idata) {
    // TODO
    odata[0] = 0;
    for (int i = 1; i < n; i++){
        odata[i] = idata[i-1] + odata[i-1];
    }
}

extern "C" __global__ void prefixFixup ( int *input, int *aux, int len) 
{
    unsigned int t = threadIdx.x;
    unsigned int start = t + 2 * blockIdx.x * SCAN_BLOCKSIZE;   
    if (start < len)                    input[start] += aux[blockIdx.x] ;              // <------------------- fixed code
    if (start + SCAN_BLOCKSIZE < len)   input[start + SCAN_BLOCKSIZE] += aux[blockIdx.x];
}

extern "C" __global__ void prefixSum ( int* input, int* output, int* aux, int len, int zeroff )   //<---- support zero-offsets
{
    __shared__ int scan_array[SCAN_BLOCKSIZE << 1];    
    unsigned int t1 = threadIdx.x + 2 * blockIdx.x * SCAN_BLOCKSIZE;    // <--- store t1,t2 for efficiency
    unsigned int t2 = t1 + SCAN_BLOCKSIZE;

    // Pre-load into shared memory
    scan_array[threadIdx.x] = (t1<len) ? input[t1] : 0;
    scan_array[threadIdx.x + SCAN_BLOCKSIZE] = (t2<len) ? input[t2] : 0;
    __syncthreads();

    // Reduction
    int stride;
    for (stride = 1; stride <= SCAN_BLOCKSIZE; stride <<= 1) {
       int index = (threadIdx.x + 1) * stride * 2 - 1;
       if (index < 2 * SCAN_BLOCKSIZE)
          scan_array[index] += scan_array[index - stride];
       __syncthreads();
    }

    // Post reduction
    for (stride = SCAN_BLOCKSIZE >> 1; stride > 0; stride >>= 1) {
       int index = (threadIdx.x + 1) * stride * 2 - 1;
       if (index + stride < 2 * SCAN_BLOCKSIZE)
          scan_array[index + stride] += scan_array[index];
       __syncthreads();
    }
    __syncthreads();

    // Output values & aux
    if (t1+zeroff < len)    output[t1+zeroff] = scan_array[threadIdx.x];          // <---- support zero-offsets
    if (t2+zeroff < len)    output[t2+zeroff] = (threadIdx.x==SCAN_BLOCKSIZE-1 && zeroff) ? 0 : scan_array[threadIdx.x + SCAN_BLOCKSIZE];   
    if ( threadIdx.x == 0 ) {
        if ( zeroff ) output[0] = 0;                    // <---- support zero-offsets
        if (aux) aux[blockIdx.x] = scan_array[2 * SCAN_BLOCKSIZE - 1];              
    }       
}

void Work_Efficient_Scan(Renderer::Params &params, int width, int height, int worldSize) {

    int N = params.pathSize;

    int *dev_auxArray1, *dev_auxArray2, *dev_auxScan1, *dev_auxScan2;
    int *d_tempScanBuffer, *d_scanInBuffer, *d_scanResultBuffer, *d_sceneOffset;

    d_tempScanBuffer = params.tempScanBuffer;
    d_scanInBuffer = params.scanInBuffer;
    d_scanResultBuffer = params.scanResultBuffer;
    d_sceneOffset = params.transferOffset;

    dev_auxArray1 = params.dev_auxArray1;
    dev_auxArray2 = params.dev_auxArray2;
    dev_auxScan1 = params.dev_auxScan1;
    dev_auxScan2 = params.dev_auxScan2;

    // 2D Kernel Launch Parameters
    int zero_offsets = 0;           // set to 0 if you want classic prefix sums, or 1 if you want zero-offset sums

    int naux = SCAN_BLOCKSIZE << 1;
    int len1 = (N / naux ) + 1;     
    int blks1 = int ( N / naux ) + 1;
    int blks2 = int ( blks1 / naux ) + 1;
    int zon=1;    // used for upper layers


    dim3 THREADS(1024, 1, 1);
    dim3 BLOCKS;

    if (N >= 65536)
    BLOCKS = dim3(64, (N + 65536 - 1) / 65536, 1);
    else if (N <= 1024)
    BLOCKS = dim3(1, 1, 1);
    else
    BLOCKS = dim3((N + 1024 - 1) / 1024, 1, 1);
    
    for (int worldID = 0; worldID < worldSize; worldID++) {

        preKernel<<<BLOCKS, THREADS>>>(N, worldID, d_tempScanBuffer, d_scanInBuffer, params.pathDataBuffer);

        // if ( N > SCAN_BLOCKSIZE*ulong(SCAN_BLOCKSIZE)*SCAN_BLOCKSIZE) {
        //     printf ( "ERROR: Number of elements exceeds prefix sum max. Adjust SCAN_BLOCKSIZE.\n" );
        // }

        // Prefix sum primary array, save last elements of blocks into aux1
        // <--- zero-offset mode option
        prefixSum<<<dim3(blks1, 1, 1), THREADS>>>(d_scanInBuffer, d_scanResultBuffer, dev_auxArray1, N, zero_offsets);
        // cudaDeviceSynchronize();
        
        // Prefix sum of aux1, save last elements of blocks into aux2
        // <--- upper level scans must use zero-offset mode
        prefixSum<<<dim3(blks2, 1, 1), THREADS>>>(dev_auxArray1, dev_auxScan1, dev_auxArray2, len1, zon);
        // cudaDeviceSynchronize();

        // Prefix sum of aux2, don't save the last elems. Implies aux2 must be <BLOCK_SIZE elements, or total cnt limited to BLOCK_SIZE^3
        int *nptr = {0};
        prefixSum<<<dim3(1, 1, 1), THREADS>>>(dev_auxArray2, dev_auxScan2, nptr, blks2, zon);
        // cudaDeviceSynchronize();           

        // Add-in the aux2 elements back into aux1 results
        prefixFixup<<<dim3(blks2, 1, 1), THREADS>>>(dev_auxScan1, dev_auxScan2, len1);
        // cudaDeviceSynchronize();       

        // Add-in the aux1 results back into primary results
        prefixFixup<<<dim3(blks1, 1, 1), THREADS>>>(d_scanResultBuffer, dev_auxScan1, N);
        // cudaDeviceSynchronize();

        cudaDeviceSynchronize();

        postKernel<<<BLOCKS, THREADS>>>(N, worldID, params.transferOffset, d_scanInBuffer, d_scanResultBuffer, params.transferPathDataBuffer, params.pathDataBuffer);

        cudaDeviceSynchronize();
    }


    // std::vector<int> sceneOffsetCPU(worldSize + 1);
    // cudaMemcpy(
    //     reinterpret_cast<void *>(sceneOffsetCPU.data()),
    //     params.transferOffset,
    //     sceneOffsetCPU.size() * sizeof(int),
    //     cudaMemcpyDeviceToHost
    // );
    // cudaMemcpy(sceneOffsetCPU.data(), params.transferOffset, (worldSize + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    // for(auto & offset : sceneOffsetCPU) {        
    //     std::cout << offset << std::endl;
    // }
}

void Work_Efficient_Scan_For_NN(Renderer::Params &params, const int &width, const int &height, const int &size) {
    int N = size;

    int *dev_auxArray1, *dev_auxArray2, *dev_auxScan1, *dev_auxScan2;
    int *d_tempScanBuffer, *d_scanInBuffer, *d_scanResultBuffer, *d_sceneOffset;

    d_tempScanBuffer = params.tempScanNNBuffer;
    d_scanInBuffer = params.scanInNNBuffer;
    d_scanResultBuffer = params.scanResultNNBuffer;
    d_sceneOffset = params.sceneOffset;
    dev_auxArray1 = params.dev_auxArray1;
    dev_auxArray2 = params.dev_auxArray2;
    dev_auxScan1 = params.dev_auxScan1;
    dev_auxScan2 = params.dev_auxScan2;

    // 2D Kernel Launch Parameters
    int zero_offsets = 0;           // set to 0 if you want classic prefix sums, or 1 if you want zero-offset sums

    int naux = SCAN_BLOCKSIZE << 1;
    int len1 = (N / naux ) + 1;     
    int blks1 = int ( N / naux ) + 1;
    int blks2 = int ( blks1 / naux ) + 1;
    int zon=1;    // used for upper layers

    dim3 THREADS(1024, 1, 1);
    dim3 BLOCKS;

    if (N >= 65536)
    BLOCKS = dim3(64, (N + 65536 - 1) / 65536, 1);
    else if (N <= 1024)
    BLOCKS = dim3(1, 1, 1);
    else
    BLOCKS = dim3((N + 1024 - 1) / 1024, 1, 1);


    int AABBID = 0;
    for (int i = 0; i < params.sceneSize; i++) { 

        AABBID = i + 1; //TODO AABBID + 1

        preKernelNN<<<BLOCKS, THREADS>>>(N, AABBID, d_tempScanBuffer, d_scanInBuffer, params.NNPathDataBuffer);

        // Throw error if number of elements exceeds BLOCKSIZE^3
        // if ( N > SCAN_BLOCKSIZE*ulong(SCAN_BLOCKSIZE)*SCAN_BLOCKSIZE) {
        //     printf ( "ERROR: Number of elements exceeds prefix sum max. Adjust SCAN_BLOCKSIZE.\n" );
        // }

        // Prefix sum primary array, save last elements of blocks into aux1
        // <--- zero-offset mode option
        prefixSum<<<dim3(blks1, 1, 1), THREADS>>>(d_scanInBuffer, d_scanResultBuffer, dev_auxArray1, N, zero_offsets);
        // cudaDeviceSynchronize();
        
        // Prefix sum of aux1, save last elements of blocks into aux2
        // <--- upper level scans must use zero-offset mode
        prefixSum<<<dim3(blks2, 1, 1), THREADS>>>(dev_auxArray1, dev_auxScan1, dev_auxArray2, len1, zon);
        // cudaDeviceSynchronize();

        // Prefix sum of aux2, don't save the last elems. Implies aux2 must be <BLOCK_SIZE elements, or total cnt limited to BLOCK_SIZE^3
        int *nptr = {0};
        prefixSum<<<dim3(1, 1, 1), THREADS>>>(dev_auxArray2, dev_auxScan2, nptr, blks2, zon);
        // cudaDeviceSynchronize();           

        // Add-in the aux2 elements back into aux1 results
        prefixFixup<<<dim3(blks2, 1, 1), THREADS>>>(dev_auxScan1, dev_auxScan2, len1);
        // cudaDeviceSynchronize();       

        // Add-in the aux1 results back into primary results
        prefixFixup<<<dim3(blks1, 1, 1), THREADS>>>(d_scanResultBuffer, dev_auxScan1, N);
        // cudaDeviceSynchronize();

        postKernelNN<<<BLOCKS, THREADS>>>(N, i, d_sceneOffset, d_scanInBuffer, d_scanResultBuffer, params.packedInputDataBuffer, params.inputDataBuffer, params.packedNNPathDataBuffer, params.NNPathDataBuffer);

        cudaDeviceSynchronize();
    }

    // CHECK_CUDA(cudaDeviceSynchronize());
    // std::cout << __LINE__ << std::endl;

    // std::vector<int> sceneOffsetCPU(params.sceneSize + 1);
    // cudaMemcpy(
    //     reinterpret_cast<void *>(sceneOffsetCPU.data()),
    //     params.sceneOffset,
    //     sceneOffsetCPU.size() * sizeof(int),
    //     cudaMemcpyDeviceToHost
    // );
    // cudaMemcpy(sceneOffsetCPU.data(), params.sceneOffset, (params.sceneSize + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    // for(auto & offset : sceneOffsetCPU) {        
    //     std::cout << offset << std::endl;
    // }
}

void Work_Efficient_Scan_For_NN_HIT_INSIDE(Renderer::Params &params, int width, int height) {
    int N = params.maxCount * params.shadowPathSize;
    int *dev_auxArray1, *dev_auxArray2, *dev_auxScan1, *dev_auxScan2;
    int *d_tempScanBuffer, *d_scanInBuffer, *d_scanResultBuffer, *d_sceneOffset;

    d_tempScanBuffer = params.tempScanNNBuffer;
    d_scanInBuffer = params.scanInNNBuffer;
    d_scanResultBuffer = params.scanResultNNBuffer;
    d_sceneOffset = params.sceneOffset;
    dev_auxArray1 = params.dev_auxArray1;
    dev_auxArray2 = params.dev_auxArray2;
    dev_auxScan1 = params.dev_auxScan1;
    dev_auxScan2 = params.dev_auxScan2;

    // 2D Kernel Launch Parameters
    int zero_offsets = 0;           // set to 0 if you want classic prefix sums, or 1 if you want zero-offset sums

    int naux = SCAN_BLOCKSIZE << 1;
    int len1 = (N / naux ) + 1;     
    int blks1 = int ( N / naux ) + 1;
    int blks2 = int ( blks1 / naux ) + 1;
    int zon=1;    // used for upper layers

    dim3 THREADS(1024, 1, 1);
    dim3 BLOCKS;

    if (N >= 65536)
    BLOCKS = dim3(64, (N + 65536 - 1) / 65536, 1);
    else if (N <= 1024)
    BLOCKS = dim3(1, 1, 1);
    else
    BLOCKS = dim3((N + 1024 - 1) / 1024, 1, 1);


    int AABBID = 0;
    for (int i = 0; i < params.sceneSize; i++) { 

        AABBID = i + 1; //TODO AABBID + 1

        preKernelNN_HIT_INSIDE<<<BLOCKS, THREADS>>>(N, AABBID, d_tempScanBuffer, d_scanInBuffer, params.NNPathDataBuffer);

        // Throw error if number of elements exceeds BLOCKSIZE^3
        // if ( N > SCAN_BLOCKSIZE*ulong(SCAN_BLOCKSIZE)*SCAN_BLOCKSIZE) {
        //     printf ( "ERROR: Number of elements exceeds prefix sum max. Adjust SCAN_BLOCKSIZE.\n" );
        // }

        // Prefix sum primary array, save last elements of blocks into aux1
        // <--- zero-offset mode option
        prefixSum<<<dim3(blks1, 1, 1), THREADS>>>(d_scanInBuffer, d_scanResultBuffer, dev_auxArray1, N, zero_offsets);
        // cudaDeviceSynchronize();
        
        // Prefix sum of aux1, save last elements of blocks into aux2
        // <--- upper level scans must use zero-offset mode
        prefixSum<<<dim3(blks2, 1, 1), THREADS>>>(dev_auxArray1, dev_auxScan1, dev_auxArray2, len1, zon);
        // cudaDeviceSynchronize();

        // Prefix sum of aux2, don't save the last elems. Implies aux2 must be <BLOCK_SIZE elements, or total cnt limited to BLOCK_SIZE^3
        int *nptr = {0};
        prefixSum<<<dim3(1, 1, 1), THREADS>>>(dev_auxArray2, dev_auxScan2, nptr, blks2, zon);
        // cudaDeviceSynchronize();           

        // Add-in the aux2 elements back into aux1 results
        prefixFixup<<<dim3(blks2, 1, 1), THREADS>>>(dev_auxScan1, dev_auxScan2, len1);
        // cudaDeviceSynchronize();       

        // Add-in the aux1 results back into primary results
        prefixFixup<<<dim3(blks1, 1, 1), THREADS>>>(d_scanResultBuffer, dev_auxScan1, N);
        // cudaDeviceSynchronize();

        postKernelNN<<<BLOCKS, THREADS>>>(N, i, d_sceneOffset, d_scanInBuffer, d_scanResultBuffer, params.packedInputDataBuffer, params.inputDataBuffer, params.packedNNPathDataBuffer, params.NNPathDataBuffer);

        cudaDeviceSynchronize();
    }

    // std::vector<int> sceneOffsetCPU(params.sceneSize + 1);
    // cudaMemcpy(
    //     reinterpret_cast<void *>(sceneOffsetCPU.data()),
    //     params.sceneOffset,
    //     sceneOffsetCPU.size() * sizeof(int),
    //     cudaMemcpyDeviceToHost
    // );
    // cudaMemcpy(sceneOffsetCPU.data(), params.sceneOffset, (params.sceneSize + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    // for(auto & offset : sceneOffsetCPU) {        
    //     std::cout << offset << std::endl;
    // }
}

}
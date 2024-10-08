#include "moana/cuda/frame_buffer_update.hpp"
#include <string>

namespace moana {

//==============================================
// shadow ray vis part
//==============================================

__global__ void shadowOcclusionCharTypeKernel(int size, int maxCount, char *d_shadowOcclusionCharTypeBuffer, NN_Float *d_predBuffer, NNPathData *d_packedNNPathDataBuffer, float *d_contributionBuffer) {

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    const int threadIndex = x + (y * gridDim.x * blockDim.x);

    NNPathData shadowPath = d_packedNNPathDataBuffer[threadIndex];
    if(!shadowPath.isValid) return;

    const int pixelIndex = shadowPath.pixelIndex * 3;
    d_contributionBuffer[pixelIndex + 0] = shadowPath.throughput[0];
    d_contributionBuffer[pixelIndex + 1] = shadowPath.throughput[1];
    d_contributionBuffer[pixelIndex + 2] = shadowPath.throughput[2];

    float threshold = 0.9;
    const int flagIndex = shadowPath.pixelIndex * maxCount + shadowPath.hitScequnce; 

    d_shadowOcclusionCharTypeBuffer[flagIndex] = __half2float(d_predBuffer[threadIndex]) < threshold ? 1 : (-3);
}

__global__ void shadowOcclusionFloatTypeKernel(int size, int shadowPathCount, int maxCount, float *d_shadowOcclusionFloatTypeBuffer, NN_Float *d_predBuffer, NNPathData *d_packedNNPathDataBuffer, float *d_contributionBuffer) {

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    const int threadIndex = x + (y * gridDim.x * blockDim.x);

    const NNPathData& shadowPath = d_packedNNPathDataBuffer[threadIndex];
    if(!shadowPath.isValid) return;

    const int pixelIndex = (shadowPath.pixelIndex * shadowPathCount + shadowPath.shadowPathID) * 3;
    d_contributionBuffer[pixelIndex + 0] = shadowPath.throughput[0];
    d_contributionBuffer[pixelIndex + 1] = shadowPath.throughput[1];
    d_contributionBuffer[pixelIndex + 2] = shadowPath.throughput[2];

    const int flagIndex = (shadowPath.pixelIndex * shadowPathCount + shadowPath.shadowPathID) * maxCount + shadowPath.hitScequnce; 

    float binaryThreshold = 0.5;
    #if SEPARATEDNN
    const int predIndex = threadIndex;
    float predValue = __half2float(d_predBuffer[predIndex]);

    d_shadowOcclusionFloatTypeBuffer[flagIndex] = predValue > binaryThreshold ? 1.0f : 0.0f;
    // d_shadowOcclusionFloatTypeBuffer[flagIndex] = d_predBuffer[predIndex];


    if(shadowPath.isInside && predValue > binaryThreshold) {
        d_shadowOcclusionFloatTypeBuffer[flagIndex] = shadowPath.normalizedT;
        // d_shadowOcclusionFloatTypeBuffer[flagIndex] = (d_predBuffer[threadIndex + (size-1)]) > shadowPath.normalizedT ? 0.f : 1.f;
        // d_shadowOcclusionFloatTypeBuffer[flagIndex] = (d_predBuffer[predIndex + 1] + 0.1) > shadowPath.normalizedT ? 0.f : 1.f;
        // d_shadowOcclusionFloatTypeBuffer[flagIndex] = (d_predBuffer[threadIndex + 2*(size-1)]) > shadowPath.normalizedT ? 0.f : 1.f;
    }
    #else
    const int predIndex = 2 * threadIndex;
    // d_shadowOcclusionFloatTypeBuffer[flagIndex] = d_predBuffer[predIndex];
    d_shadowOcclusionFloatTypeBuffer[flagIndex] = d_predBuffer[predIndex] > binaryThreshold ? 1.0f : 0.0f;

    if(shadowPath.isInside && d_predBuffer[predIndex] > binaryThreshold) {
        d_shadowOcclusionFloatTypeBuffer[flagIndex] = (d_predBuffer[predIndex + 1] + 0.1) > shadowPath.normalizedT ? 0.f : 1.f;
    }
    #endif 
}

__global__ void contributionKernelCharType(int size, int maxCount, char *d_shadowOcclusionCharTypeBuffer,float *d_directLightingBuffer, float *d_contributionBuffer) {

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    const int threadIndex = x + (y * gridDim.x * blockDim.x);

    const int pixelIndex = threadIndex * 3;

    char flag = 0;
    for (int i = 0; i < maxCount; i++) {
        flag += d_shadowOcclusionCharTypeBuffer[pixelIndex + i];
    }

    if(flag > 0) {
        d_directLightingBuffer[pixelIndex + 0] += d_contributionBuffer[pixelIndex + 0];
        d_directLightingBuffer[pixelIndex + 1] += d_contributionBuffer[pixelIndex + 1];
        d_directLightingBuffer[pixelIndex + 2] += d_contributionBuffer[pixelIndex + 2];
    }
}

__global__ void contributionKernelFloatType(int size, int shadowPathCount,int maxCount, float *d_shadowOcclusionFloatTypeBuffer,float *d_directLightingBuffer, float *d_contributionBuffer) {

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    const int threadIndex = x + (y * gridDim.x * blockDim.x);

    if(threadIndex >= size) return;
    const int pixelIndex = threadIndex * 3;

    for (int i = 0; i < shadowPathCount; i++) {

        const int occlusionIndex = (threadIndex * shadowPathCount + i) * maxCount;
        const int contributionIndex = (threadIndex * shadowPathCount + i) * 3;

        float maxOcclusion = 0.f;
        for (int j = 0; j < maxCount; j++) {
            maxOcclusion = maxOcclusion > d_shadowOcclusionFloatTypeBuffer[occlusionIndex + j] ? maxOcclusion : d_shadowOcclusionFloatTypeBuffer[occlusionIndex + j];
        }
    
        d_directLightingBuffer[pixelIndex + 0] += d_contributionBuffer[contributionIndex + 0] * (1.f - maxOcclusion) / float(shadowPathCount);
        d_directLightingBuffer[pixelIndex + 1] += d_contributionBuffer[contributionIndex + 1] * (1.f - maxOcclusion) / float(shadowPathCount);
        d_directLightingBuffer[pixelIndex + 2] += d_contributionBuffer[contributionIndex + 2] * (1.f - maxOcclusion) / float(shadowPathCount);

    }

    for (int i = 1; i < shadowPathCount; i++) {
        d_directLightingBuffer[pixelIndex + 0] += d_directLightingBuffer[size * i * 3 + pixelIndex + 0];
        d_directLightingBuffer[pixelIndex + 1] += d_directLightingBuffer[size * i * 3 + pixelIndex + 1];
        d_directLightingBuffer[pixelIndex + 2] += d_directLightingBuffer[size * i * 3 + pixelIndex + 2];

    }
}

void Frame_Buffer_Update(Renderer::Params &params, int size, int width, int height) {

    float *d_shadowOcclusionFloatTypeBuffer = params.shadowOcclusionFloatTypeBuffer;
    char *d_shadowOcclusionCharTypeBuffer = params.shadowOcclusionCharTypeBuffer;
    NN_Float *d_predBuffer = params.predBuffer;
    NNPathData *d_packedNNPathDataBuffer = params.packedNNPathDataBuffer;
    float *d_contributionBuffer = params.contributionBuffer;
    float *d_directLightingBuffer = params.directLightingBuffer;
    int maxCount = params.maxCount;

    int N = size;

    dim3 THREADS(1024, 1, 1);

    dim3 BLOCKS;

    if (N >= 65536)
        BLOCKS = dim3(64, (N + 65536 - 1) / 65536, 1);
    else if (N <= 1024)
        BLOCKS = dim3(1, 1, 1);
    else
        BLOCKS = dim3((N + 1024 - 1) / 1024, 1, 1);
    
    // shadowOcclusionCharTypeKernel<<< BLOCKS, THREADS >>>(size, maxCount, d_shadowOcclusionCharTypeBuffer, d_predBuffer, d_packedNNPathDataBuffer, d_contributionBuffer);
    shadowOcclusionFloatTypeKernel<<< BLOCKS, THREADS >>>(size, params.shadowPathCount, maxCount, d_shadowOcclusionFloatTypeBuffer, d_predBuffer, d_packedNNPathDataBuffer, d_contributionBuffer);
    N = width * height;

    if (N >= 65536)
        BLOCKS = dim3(64, (N + 65536 - 1) / 65536, 1);
    else if (N <= 1024)
        BLOCKS = dim3(1, 1, 1);
    else
        BLOCKS = dim3((N + 1024 - 1) / 1024, 1, 1);
    
    // contributionKernelCharType<<< BLOCKS, THREADS >>>(width * height, maxCount, d_shadowOcclusionCharTypeBuffer, d_directLightingBuffer, d_contributionBuffer);
    contributionKernelFloatType<<< BLOCKS, THREADS >>>(N, params.shadowPathCount, maxCount, d_shadowOcclusionFloatTypeBuffer, d_directLightingBuffer, d_contributionBuffer);

}

//==============================================
// shadow ray depth part
//==============================================

__global__ void predDepthUpdateKernel(
    int size,
    int shadowPathCount,
    int maxCount,
    NN_Float* d_predBuffer,
    NNPathData* d_packedNNPathDataBuffer,
    NNPathData* d_NNPathDataBuffer
) {

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    int threadIndex = x + (y * gridDim.x * blockDim.x);

    if(threadIndex >= size) return;

    NNPathData shadowPath = d_packedNNPathDataBuffer[threadIndex];
    int pathIndex = shadowPath.pathIndex;

    d_NNPathDataBuffer[pathIndex].normalizedT = __half2float(d_predBuffer[threadIndex]) > float(shadowPath.normalizedT) ? 0.f : 1.f;
}

void Depth_Buffer_Update(Renderer::Params &params, int size, int width, int height) {

    NN_Float *d_predBuffer = params.predBuffer;
    NNPathData *d_packedNNPathDataBuffer = params.packedNNPathDataBuffer;
    NNPathData *d_NNPathDataBuffer = params.NNPathDataBuffer;
    int maxCount = params.maxCount;
    int shadowPathCount = params.shadowPathCount;
    int N = size;

    dim3 THREADS(1024, 1, 1);

    dim3 BLOCKS;

    if (N >= 65536)
        BLOCKS = dim3(64, (N + 65536 - 1) / 65536, 1);
    else if (N <= 1024)
        BLOCKS = dim3(1, 1, 1);
    else
        BLOCKS = dim3((N + 1024 - 1) / 1024, 1, 1);
    
    predDepthUpdateKernel<<< BLOCKS, THREADS >>>(size, shadowPathCount, maxCount, d_predBuffer, d_packedNNPathDataBuffer, d_NNPathDataBuffer);
}

//==============================================
// secondary ray part
//==============================================
static constexpr float rotationOffset = 0.f / 180.f * M_PI;//旋转环境光贴图

__global__ void tMaxFloatTypeKernel(int size, int maxCount, float *d_shadowOcclusionFloatTypeBuffer, NN_Float *d_predBuffer, NNPathData *d_packedNNPathDataBuffer) {

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    const int threadIndex = x + (y * gridDim.x * blockDim.x);

    const NNPathData& secondaryPath = d_packedNNPathDataBuffer[threadIndex];
    if(!secondaryPath.isValid) return;

    const int tMaxIndex = (secondaryPath.pixelIndex * maxCount + secondaryPath.hitScequnce) * 2 + 0;
    const int nodeIDIndex = tMaxIndex + 1;

    float binaryThreshold = 0.5;
    #if SEPARATEDNN
    const int predIndex = threadIndex;

    if(__half2float(d_predBuffer[predIndex]) > binaryThreshold) {
        const float& predMax = secondaryPath.throughput[2] * secondaryPath.throughput[1] * __half2float(d_predBuffer[predIndex + size]);
        const float& aabbMax = secondaryPath.throughput[0];

        if(secondaryPath.isInside) {
            // d_shadowOcclusionFloatTypeBuffer[tMaxIndex] = predMax > aabbMax ? aabbMax : (aabbMax - predMax);
            d_shadowOcclusionFloatTypeBuffer[tMaxIndex] = predMax > aabbMax ? 0.0f : (aabbMax - predMax);
        } else {
            d_shadowOcclusionFloatTypeBuffer[tMaxIndex] = aabbMax + predMax;
        }
    } else {
        d_shadowOcclusionFloatTypeBuffer[tMaxIndex] = 0.0f;
    }

    d_shadowOcclusionFloatTypeBuffer[nodeIDIndex] = secondaryPath.pathIndex;
    #else
    const int predIndex = 2 * threadIndex;
    #endif 
}

__global__ void targetNodeKernelFloatType(int size, int maxCount, int worldID,float *d_shadowOcclusionFloatTypeBuffer, WavefrontPathData *d_pathDataBuffer, cudaTextureObject_t &d_envLightTexture, float *d_envLightingBuffer) {

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    const int threadIndex = x + (y * gridDim.x * blockDim.x);
    if(threadIndex >= size) return;

    WavefrontPathData& secondaryRay = d_pathDataBuffer[threadIndex];
    if(!secondaryRay.isValid) return;
    
    float tMax = secondaryRay.tMax;
    int currentNode = secondaryRay.currentNode;
    
    for (int i = 0; i < maxCount; i++) {
        const int tMaxIndex = (secondaryRay.pixelIndex * maxCount + i) * 2 + 0;
        const int nodeIDIndex = tMaxIndex + 1;

        if(d_shadowOcclusionFloatTypeBuffer[tMaxIndex] < FLT_EPSILON) continue;

        if(tMax > d_shadowOcclusionFloatTypeBuffer[tMaxIndex]) {
            tMax = d_shadowOcclusionFloatTypeBuffer[tMaxIndex];
            currentNode = d_shadowOcclusionFloatTypeBuffer[nodeIDIndex];
        }
        
    }
    
    if(currentNode >= 0) {
        secondaryRay.currentNode = currentNode;
        secondaryRay.targetNode = currentNode;
        secondaryRay.isHit = true;
        secondaryRay.tMax = tMax;

    } else {
        secondaryRay.targetNode = worldID;
        secondaryRay.tMax = 0.0f;
        secondaryRay.isHit = false;
        secondaryRay.isValid = true;
        // float4 environmentLight{0.f, 0.f, 0.f, 0.f};

        // // Pixels that have already been lit in previous bounces
        // float phi, theta;
        // Coordinates::cartesianToSpherical(secondaryRay.direction, &phi, &theta);
    
        // phi += rotationOffset;
        // if (phi > 2.f * M_PI) {
        //     phi -= 2.f * M_PI;
        // }
        // environmentLight = tex2D<float4>(
        //     d_envLightTexture,
        //     phi / (M_PI * 2.f),
        //     theta / M_PI
        // );
        
        // // Vec3 environment = Vec3(environmentLight.x, environmentLight.y, environmentLight.z);

        // secondaryRay.isValid = false;
        // const int pixelIndex = secondaryRay.pixelIndex * 3;
        // // return;
        // float x = secondaryRay.throughput.r() * environmentLight.x;
        // d_envLightingBuffer[pixelIndex + 0] = 1.0f;
        // d_envLightingBuffer[pixelIndex + 1] += secondaryRay.throughput.g() * environmentLight.y;
        // d_envLightingBuffer[pixelIndex + 2] += secondaryRay.throughput.b() * environmentLight.z;        
    }
    // calculateEnvironmentLighting(path);
}

void Target_Node_Update(Renderer::Params &params, int size, int width, int height) {

    float *d_shadowOcclusionFloatTypeBuffer = params.shadowOcclusionFloatTypeBuffer;
    NN_Float *d_predBuffer = params.predBuffer;
    NNPathData *d_packedNNPathDataBuffer = params.packedNNPathDataBuffer;
    NNPathData *d_NNPathDataBuffer = params.NNPathDataBuffer;
    WavefrontPathData *d_pathDataBuffer = params.pathDataBuffer;
    int maxCount = params.maxCount;
    int N = size;
    
    dim3 THREADS(1024, 1, 1);
    
    dim3 BLOCKS;
    
    if (N >= 65536)
    BLOCKS = dim3(64, (N + 65536 - 1) / 65536, 1);
    else if (N <= 1024)
    BLOCKS = dim3(1, 1, 1);
    else
    BLOCKS = dim3((N + 1024 - 1) / 1024, 1, 1);
    
    tMaxFloatTypeKernel<<< BLOCKS, THREADS >>>(size, maxCount, d_shadowOcclusionFloatTypeBuffer, d_predBuffer, d_packedNNPathDataBuffer);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    N = params.pathSize;
    if (N >= 65536)
        BLOCKS = dim3(64, (N + 65536 - 1) / 65536, 1);
    else if (N <= 1024)
        BLOCKS = dim3(1, 1, 1);
    else
        BLOCKS = dim3((N + 1024 - 1) / 1024, 1, 1);

    targetNodeKernelFloatType<<< BLOCKS, THREADS >>>(N, maxCount, params.worldID, d_shadowOcclusionFloatTypeBuffer, d_pathDataBuffer, params.envLightTexture, params.envLightingBuffer);
    CHECK_CUDA(cudaDeviceSynchronize());

}
}
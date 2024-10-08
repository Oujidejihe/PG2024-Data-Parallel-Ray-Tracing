#include "moana/cuda/environment_light.hpp"
#include "moana/macro_define.hpp"

#include "assert_macros.hpp"
#include "moana/core/coordinates.hpp"
#include "moana/core/vec3.hpp"

namespace moana {

// fixme
static constexpr float rotationOffset = 115.f / 180.f * M_PI;//旋转环境光贴图

__global__ static void environmentLightKernel(
    int width,
    int height,
    int spp,
    cudaTextureObject_t textureObject,
    float *occlusionBuffer,
    float *depthBuffer,
    float *directionBuffer,
    float *betaBuffer,
    float *outputImage
) {
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;

    if ((row >= height) || (col >= width)) { return; }

    const int outputIndex = 3 * (row * width + col);
    float4 environment{0.f, 0.f, 0.f, 0.f};

    const int directionIndex = 3 * (row * width + col);
    const int occlusionIndex = 1 * (row * width + col);
    if (occlusionBuffer[occlusionIndex] != 0.f ) { return; }

    Vec3 direction(
        directionBuffer[directionIndex + 0],
        directionBuffer[directionIndex + 1],
        directionBuffer[directionIndex + 2]
    );

    // Pixels that have already been lit in previous bounces
    if (direction.isZero()) { return; }

    float phi, theta;
    Coordinates::cartesianToSpherical(direction, &phi, &theta);

    phi += rotationOffset;
    if (phi > 2.f * M_PI) {
        phi -= 2.f * M_PI;
    }

    environment = tex2D<float4>(
        textureObject,
        phi / (M_PI * 2.f),
        theta / M_PI
    );
    outputImage[outputIndex + 0] += 1.f * environment.x * betaBuffer[outputIndex + 0] * (1.f / spp);
    outputImage[outputIndex + 1] += 1.f * environment.y * betaBuffer[outputIndex + 1] * (1.f / spp);
    outputImage[outputIndex + 2] += 1.f * environment.z * betaBuffer[outputIndex + 2] * (1.f / spp);
}

void EnvironmentLight::calculateEnvironmentLighting(
    int width,
    int height,
    int spp,
    ASArena &arena,
    cudaTextureObject_t textureObject,
    float *devOcclusionBuffer,
    float *devDepthBuffer,
    float *devDirectionBuffer,
    float *betaBuffer,
    std::vector<float> &outputImage
) {
    const size_t outputImageSizeInBytes = outputImage.size() * sizeof(float);
    const size_t betaBufferSizeInBytes = width * height * 3 * sizeof(float);

    CUdeviceptr d_outputImage = 0;
    CUdeviceptr d_betaBuffer = 0;

    d_outputImage = arena.pushTemp(outputImageSizeInBytes);
    d_betaBuffer = arena.pushTemp(betaBufferSizeInBytes);

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_betaBuffer),
        betaBuffer,
        betaBufferSizeInBytes,
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_outputImage),
        outputImage.data(),
        outputImageSizeInBytes,
        cudaMemcpyHostToDevice
    ));

    const int blockWidth = 16;
    const int blockHeight = 16;

    const dim3 blocks((width + blockWidth - 1 ) / blockWidth , ( height + blockHeight - 1 ) / blockHeight);
    const dim3 threads(blockWidth, blockHeight);

    environmentLightKernel<<<blocks, threads>>>(
        width,
        height,
        spp,
        textureObject,
        devOcclusionBuffer,
        devDepthBuffer,
        devDirectionBuffer,
        reinterpret_cast<float *>(d_betaBuffer),
        reinterpret_cast<float *>(d_outputImage)
    );

    CHECK_CUDA(cudaMemcpy(
        outputImage.data(),
        reinterpret_cast<void *>(d_outputImage),
        outputImageSizeInBytes,
        cudaMemcpyDeviceToHost
    ));

    arena.popTemp(); // d_outputImage
    arena.popTemp(); // d_betaBuffer

    CHECK_CUDA(cudaDeviceSynchronize());
}

void EnvironmentLight::queryMemoryRequirements()
{
    std::string moanaRoot = MOANA_ROOT;
    #if TEX_MOANA
    m_texturePtr = std::make_unique<Texture>(moanaRoot + "/island/textures/com_islandsun.exr");
    #elif AIR_DROME
    m_texturePtr = std::make_unique<Texture>(moanaRoot + "/island/textures/envMap.exr");
    #else
    m_texturePtr = std::make_unique<Texture>(moanaRoot + "/island/textures/envMap.exr");
    #endif
    m_texturePtr->determineAndSetPitch();
}

EnvironmentLightState EnvironmentLight::snapshotTextureObject(ASArena &arena)
{
    EnvironmentLightState environmentState;
    environmentState.textureObject = m_texturePtr->createTextureObject(arena);
    environmentState.snapshot = arena.createSnapshot();

    return environmentState;
}

}

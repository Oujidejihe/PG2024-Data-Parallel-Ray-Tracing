#include <optix.h>

#include <stdio.h>
#include <float.h>

#include "bsdfs/lambertian.hpp"
#include "bsdfs/water.hpp"
#include "moana/core/bsdf_sample_record.hpp"
#include "moana/core/camera.hpp"
#include "moana/core/ray.hpp"
#include "moana/cuda/bsdf.hpp"
#include "moana/driver.hpp"
#include "moana/render/renderer.hpp"
#include "optix_sdk.hpp"
#include "random.hpp"
#include "ray_data.hpp"
#include "util.hpp"

#include "sample.hpp"
#include "moana/cuda/triangle.hpp"
#include "moana/core/frame.hpp"

#include "moana/core/coordinates.hpp"

using namespace moana;

extern "C" {
    __constant__ Renderer::Params params;
}

extern "C" __global__ void __closesthit__ch()
{

}

extern "C" __global__ void __miss__ms()
{

}

extern "C" __global__ void __anyhit__ah()
{
    
}

__forceinline__ __device__ static WavefrontPathData pathgenCamera()
{   
    const uint3 index = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const int row = index.y;
    const int col = index.x;

    unsigned int seed = tea<4>(index.y * dim.x + index.x, params.sampleCount);
    float xi1 = rnd(seed);
    float xi2 = rnd(seed);

    const Ray cameraRay = params.camera.generateRay(
        row, col,
        float2{ xi1, xi2 }
    );

    // WavefrontPathData path = {
    //     .origin = cameraRay.origin(),
    //     .direction = cameraRay.direction(),
    //     .tMax = __FLT_MAX__,
    //     .throughput = 1.f,
    //     .pixelIndex = index.y * dim.x + index.x,
    //     .visitedMask = 0,
    //     .currentNode = -1,
    //     .targetNode = -1,
    //     false,
    //     false,
    //     false,
    //     false
    // };

    WavefrontPathData path;
    path.origin = cameraRay.origin();
    path.direction = cameraRay.direction();
    path.pixelIndex = index.y * dim.x + index.x;
    path.throughput = Vec3(1.f, 1.f, 1.f);
    path.currentNode = -1;
    path.isValid = true;
    path.isShadowRay = false;
    path.isHit = false;
    path.isDelta = false;
    path.tMax = FLT_MAX;

    return path;
}


extern "C" __global__ void __raygen__rg()
{
    const uint3 index = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const int threadIndex = index.y * dim.x + index.x;

    WavefrontPathData path;
    path = pathgenCamera();

    params.pathDataBuffer[threadIndex] = path;
}

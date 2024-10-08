#include <optix.h>

#include <stdio.h>

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

#include "moana/core/coordinates.hpp"

using namespace moana;

extern "C" {
    __constant__ Renderer::Params params;
}

__forceinline__ __device__ static PerRayData *getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<PerRayData *>(util::unpackPointer(u0, u1));
}

extern "C" __global__ void __anyhit__ah()
{   
    if (optixIsTriangleHit()) {
        unsigned int primitiveIndex = optixGetPrimitiveIndex();
        HitGroupData *hitgroupData = reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
        
        const int textureIndex = hitgroupData->textureIndex;

        if (textureIndex > -1)
        {
            int textureIndex0 = hitgroupData->texCoordsIndices[primitiveIndex * 3 + 0];
            int textureIndex1 = hitgroupData->texCoordsIndices[primitiveIndex * 3 + 1];
            int textureIndex2 = hitgroupData->texCoordsIndices[primitiveIndex * 3 + 2];

            float t0x = hitgroupData->texCoords[textureIndex0 * 2 + 0];
            float t0y = hitgroupData->texCoords[textureIndex0 * 2 + 1];

            float t1x = hitgroupData->texCoords[textureIndex1 * 2 + 0];
            float t1y = hitgroupData->texCoords[textureIndex1 * 2 + 1];

            float t2x = hitgroupData->texCoords[textureIndex2 * 2 + 0];
            float t2y = hitgroupData->texCoords[textureIndex2 * 2 + 1];

            const float2 barycentrics = optixGetTriangleBarycentrics();
            const float alpha = barycentrics.x;
            const float beta = barycentrics.y;
            const float gamma = 1.f - alpha - beta;

            float tx = gamma * t0x + alpha * t1x + beta * t2x;
            float ty = gamma * t0y + alpha * t1y + beta * t2y;

            float4 albedo{0.f, 0.f, 0.f, 0.f};
            albedo = tex2D<float4>(
                params.albedoTextures[textureIndex],
                tx,
                ty
            );

            const float opacity = albedo.w;

            // Stochastic alpha test to get an alpha blend effect.
            if (opacity < 0.05f)// No need to calculate an expensive random number if the test is going to fail anyway.
            {
                optixIgnoreIntersection();
            }

        }
    }
}

extern "C" __global__ void __closesthit__ch()
{
    PerRayData *prd = getPRD();
    prd->isHit = true;
    prd->t = optixGetRayTmax();
}

extern "C" __global__ void __miss__ms()
{
    float3 direction = optixGetWorldRayDirection();
    PerRayData *prd = getPRD();
    prd->isHit = false;
    prd->materialID = -1;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 index = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    Ray ray;

    const int rayIndex = index.y * dim.x + index.x;
    if(rayIndex >= params.pathSize) return;
    
    ray = params.rayBuffer[rayIndex];
    if (!ray.isValid) return;

    const Vec3 origin = ray.origin();
    const Vec3 direction = ray.direction();

    ray.isValid = true;

    PerRayData prd;
    // const int depthIndex = index.y * dim.x + index.x;
    // int stripe = params.handleSize;
    // int offset = 1;
    
    prd.isHit = false;

    const float tMax = __FLT_MAX__;

    unsigned int p0, p1;
    util::packPointer(&prd, p0, p1);
    optixTrace(
        params.accelerationStructures[params.startObj].originHandle,
        float3{ origin.x(), origin.y(), origin.z() },
        float3{ direction.x(), direction.y(), direction.z() },
        1e-5,
        tMax,
        0.f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
        0, 1, 0, // SBT params
        p0, p1
    );

    if (prd.isHit) {
        ray.tMax = prd.t;
    } else {
        ray.tMax = tMax;
    }

    params.rayBuffer[rayIndex] = ray;

    //Train Data
    float phi, theta;
    Coordinates::cartesianToSphericalForTrain(direction, &phi, &theta);

    const aabbRecord AABBInfo = params.accelerationStructures[params.startObj].AABBInfo;

    params.originBuffer[rayIndex * 3 + 0] = (origin.x() - AABBInfo.m_minX) / (AABBInfo.m_maxX - AABBInfo.m_minX);
    params.originBuffer[rayIndex * 3 + 1] = (origin.y() - AABBInfo.m_minY) / (AABBInfo.m_maxY - AABBInfo.m_minY);
    params.originBuffer[rayIndex * 3 + 2] = (origin.z() - AABBInfo.m_minZ) / (AABBInfo.m_maxZ - AABBInfo.m_minZ);

    params.directionBuffer[rayIndex * 3 + 0] = phi / (2 * M_PI);
    params.directionBuffer[rayIndex * 3 + 1] = theta / M_PI;
    params.directionBuffer[rayIndex * 3 + 2] = prd.isHit ? (ray.tMax / AABBInfo.m_maxLength) : 1.f;
}
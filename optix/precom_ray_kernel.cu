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

extern "C" __global__ void __closesthit__ch() {
    PerRayData *prd = getPRD();
    prd->isHit = true;
    prd->t = optixGetRayTmax();

    const float3 rayDirection = optixGetWorldRayDirection();
    const float3 point = getHitPoint();
    prd->point = point;
    prd->pointLocal = optixTransformPointFromWorldToObjectSpace(point);
    prd->originLocal = optixTransformPointFromWorldToObjectSpace(optixGetWorldRayOrigin());
    prd->directionLocal = optixTransformVectorFromWorldToObjectSpace(rayDirection);

    const unsigned int primitiveIndex = optixGetPrimitiveIndex();
    prd->primitiveID = primitiveIndex;
    prd->instanceID  = optixGetInstanceIndex();
    const Vec3 woWorld = -Vec3(rayDirection.x, rayDirection.y, rayDirection.z);
    Vec3 normal;

   if (optixIsTriangleHit()) {
        HitGroupData *hitgroupData = reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
        int normalIndex0 = hitgroupData->normalIndices[primitiveIndex * 3 + 0];

        float n0x = hitgroupData->normals[normalIndex0 * 3 + 0];
        float n0y = hitgroupData->normals[normalIndex0 * 3 + 1];
        float n0z = hitgroupData->normals[normalIndex0 * 3 + 2];

        float3 n0Object{n0x, n0y, n0z};

        float3 n0World = optixTransformNormalFromObjectToWorldSpace(n0Object);

        const Vec3 n0 = normalized(Vec3(n0World.x, n0World.y, n0World.z));

        normal = n0;
    } else {
        const float3 normalFLT3 = normalCubic(primitiveIndex);
        normal = normalized(Vec3(normalFLT3.x, normalFLT3.y, normalFLT3.z));
    }

    // prd->isInside = true;
    prd->isInside = !optixIsFrontFaceHit(optixGetHitKind());

    if (prd->isInside) {
        normal = -1.f * normal;
        prd->isInside = true;
    }
}

extern "C" __global__ void __miss__ms()
{
    float3 direction = optixGetWorldRayDirection();
    PerRayData *prd = getPRD();
    prd->isHit = false;
    prd->materialID = -1;
}

// extern "C" __global__ void __raygen__rg()
// {
//     const uint3 index = optixGetLaunchIndex();
//     const uint3 dim = optixGetLaunchDimensions();

//     Ray ray;

//     const int rayIndex = index.y * dim.x + index.x;
//     if(rayIndex >= params.pathSize) return;
    
//     ray = params.rayBuffer[rayIndex];
//     if (!ray.isValid) return;

//     const Vec3 origin = ray.origin();
//     const Vec3 direction = ray.direction();

//     ray.isValid = true;

//     PerRayData prd;
//     // const int depthIndex = index.y * dim.x + index.x;
//     // int stripe = params.handleSize;
//     // int offset = 1;
    
//     prd.isHit = false;

//     const float tMax = __FLT_MAX__;

//     unsigned int p0, p1;
//     util::packPointer(&prd, p0, p1);
//     optixTrace(
//         params.accelerationStructures[params.startObj].originHandle,
//         float3{ origin.x(), origin.y(), origin.z() },
//         float3{ direction.x(), direction.y(), direction.z() },
//         1e-2,
//         tMax,
//         0.f,
//         OptixVisibilityMask(255),
//         OPTIX_RAY_FLAG_NONE,
//         0, 1, 0, // SBT params
//         p0, p1
//     );

//     if (prd.isHit) {
//         ray.tMax = prd.t;
//     } else {
//         ray.tMax = tMax;
//     }

//     params.rayBuffer[rayIndex] = ray;

//     // params.missDirectionBuffer[missDirectionIndex + 0] = direction.x();
//     // params.missDirectionBuffer[missDirectionIndex + 1] = direction.y();
//     // params.missDirectionBuffer[missDirectionIndex + 2] = direction.z();
// }

extern "C" __global__ void __raygen__rg()
{

    const uint3 index = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const int threadIndex = index.y * dim.x + index.x;

    if (threadIndex >= params.pathSize) return;

    WavefrontPathData path;
    path = params.pathDataBuffer[threadIndex];

    if (!path.isValid) { return; }

    const Vec3 origin = path.origin;
    const Vec3 direction = path.direction;

    PerRayData prd;
    int hitAABBIndex;
    float proxy_t_max, geo_t_max;
    bool isInside;
    int instanceID, primitiveID;

    // intersect with local geometries
    for (int i = 0; i < params.sceneSize; i++) {

        if(!params.accelerationStructures[i].isProxy) continue;

        prd.isHit = false;

        unsigned int p0, p1;
        util::packPointer(&prd, p0, p1);
        optixTrace(
            params.accelerationStructures[i].aabbHandle,
            float3{ origin.x(), origin.y(), origin.z() },
            float3{ direction.x(), direction.y(), direction.z() },
            1e-2,
            path.tMax,
            0.f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0, // SBT params
            p0, p1
        );

        if (prd.isHit) {
            path.tMax = prd.t;
            path.isHit = true;
            hitAABBIndex = i;
            isInside = prd.isInside;
            instanceID = prd.instanceID;
            primitiveID = prd.primitiveID;
        } 
    }

    if (prd.isHit) {
        //Train Data
        float phi, theta;
        float flag = prd.isInside ? -1.f : 1.f;
        Vec3 directionLocal = Vec3(prd.directionLocal.x * flag, prd.directionLocal.y * flag, prd.directionLocal.z * flag);
        Coordinates::cartesianToSphericalForTrain(normalized(directionLocal), &phi, &theta);

        const aabbRecord AABBInfo = params.accelerationStructures[hitAABBIndex].AABBInfo;
        params.originBuffer[path.pixelIndex * 3 + 0] =  (prd.pointLocal.x - AABBInfo.m_minX) / (AABBInfo.m_maxX - AABBInfo.m_minX);
        params.originBuffer[path.pixelIndex * 3 + 1] =  (prd.pointLocal.y - AABBInfo.m_minY) / (AABBInfo.m_maxY - AABBInfo.m_minY);
        params.originBuffer[path.pixelIndex * 3 + 2] =  (prd.pointLocal.z - AABBInfo.m_minZ) / (AABBInfo.m_maxZ - AABBInfo.m_minZ);

        params.directionBuffer[path.pixelIndex * 3 + 0] = phi / (2 * M_PI);
        params.directionBuffer[path.pixelIndex * 3 + 1] = theta / M_PI;
        proxy_t_max = path.tMax;
    }

    path.tMax = __FLT_MAX__;
    for (int i = 0; i < params.sceneSize; i++) {

        if(params.accelerationStructures[i].isProxy) continue;

        prd.isHit = false;

        unsigned int p0, p1;
        util::packPointer(&prd, p0, p1);
        optixTrace(
            params.accelerationStructures[i].originHandle,
            float3{ origin.x(), origin.y(), origin.z() },
            float3{ direction.x(), direction.y(), direction.z() },
            1e-2,
            path.tMax,
            0.f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
            0, 1, 0, // SBT params
            p0, p1
        );

        if (prd.isHit) {
            path.tMax = prd.t;
            path.isHit = true;
        } 
    }

    if (prd.isHit) {
        geo_t_max = path.tMax;
        params.directionBuffer[path.pixelIndex * 3 + 2] = (geo_t_max - proxy_t_max) / params.accelerationStructures[0].AABBInfo.m_maxLength;
        // params.directionBuffer[path.pixelIndex * 3 + 2] = 1.f;
    }
}

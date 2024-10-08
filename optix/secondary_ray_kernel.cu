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

// static constexpr float rotationOffset = 0.f / 180.f * M_PI;//旋转环境光贴图

extern "C" {
    __constant__ Renderer::Params params;
}

__forceinline__ __device__ static void setbit(unsigned int &visitedMask, int worldID) {
    visitedMask |= (1 << worldID); 
}

__device__ static void calculateEnvironmentLighting(WavefrontPathData &path) {
    float4 environmentLight{0.f, 0.f, 0.f, 0.f};

    // Pixels that have already been lit in previous bounces
    float phi, theta;
    Coordinates::cartesianToSpherical(path.direction, &phi, &theta);

    phi += rotationOffset;
    if (phi > 2.f * M_PI) {
        phi -= 2.f * M_PI;
    }

    environmentLight = tex2D<float4>(
        params.envLightTexture,
        phi / (M_PI * 2.f),
        theta / M_PI
    );

    Vec3 environment = Vec3(environmentLight.x, environmentLight.y, environmentLight.z);
    path.throughput = path.throughput * environment;
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

        if (textureIndex >= 0)
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
    // prd->primitiveID = primitiveIndex;
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

    if (dot(normal, woWorld) < 0.f) {
        normal = -1.f * normal;
        prd->isInside = true;
    } else {
        prd->isInside = false;
    }

    prd->isInside = !optixIsFrontFaceHit(optixGetHitKind());
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

    const int threadIndex = index.y * dim.x + index.x;

    WavefrontPathData path;
    path = params.pathDataBuffer[threadIndex];

    if(threadIndex >= params.pathSize) return;
    if (!path.isValid) { return; }

    const Vec3 origin = path.origin;
    const Vec3 direction = path.direction;

    PerRayData prd;

    // intersect with local geometries
    for (int i = 0; i < params.sceneSize; i++) {
        setbit(path.visitedMask, params.accelerationStructures[i].nodeID);

        if(params.accelerationStructures[i].isProxy) { continue; }

        prd.isHit = false;

        unsigned int p0, p1;
        util::packPointer(&prd, p0, p1);
        optixTrace(
            params.accelerationStructures[i].handle,
            float3{ origin.x(), origin.y(), origin.z() },
            float3{ direction.x(), direction.y(), direction.z() },
            util::Epsilon,
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
            path.currentNode = params.worldID;
        } 
    }
    
    // set visitedMask
    // setbit(path.visitedMask, params.worldID);

    bool isHit = true;
    bool isInside;
    float tMin = 0;
    int count = 0;
    int hitAABBIndex;
    int maxCount = params.maxCount;

    while (isHit && count < maxCount) {

        isHit = false;
        float tMax = path.tMax;

        for (int i = 0; i < params.sceneSize; i++) {
            const Renderer::AccelerationStructure &AS = params.accelerationStructures[i];

            if(!AS.isProxy) continue; //判断是否为代理加速结构
    
            prd.isHit = false;
            
            unsigned int p0, p1;
            util::packPointer(&prd, p0, p1);
            optixTrace(
                AS.aabbHandle,
                float3{ origin.x(), origin.y(), origin.z() },
                float3{ direction.x(), direction.y(), direction.z() },
                tMin + util::Epsilon,
                tMax,
                0.f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                0, 1, 0, // SBT params
                p0, p1
            );

            if (prd.isHit) {
                tMax = prd.t;
                isHit = true;
                hitAABBIndex = i;
                isInside = prd.isInside;
            }  
        }

        aabbRecord AABBInfo;
        float aabbX, aabbY, aabbZ;

        if (isHit) {
            AABBInfo = params.accelerationStructures[hitAABBIndex].AABBInfo;
            tMin = tMax;
            aabbX = (AABBInfo.m_maxX - AABBInfo.m_minX);
            aabbY = (AABBInfo.m_maxY - AABBInfo.m_minY);
            aabbZ = (AABBInfo.m_maxZ - AABBInfo.m_minZ);
        }

        if (isHit && !isInside) {
            float phi, theta;
            Vec3 directionLocal = normalized(Vec3(prd.directionLocal.x, prd.directionLocal.y, prd.directionLocal.z));
            Coordinates::cartesianToSphericalForTrain(directionLocal, &phi, &theta);

            const int inputDataOffset = threadIndex * 5 * maxCount + count * 5;

            params.inputDataBuffer[inputDataOffset + 0] = __float2half((prd.pointLocal.x - AABBInfo.m_minX) / aabbX);
            params.inputDataBuffer[inputDataOffset + 1] = __float2half((prd.pointLocal.y - AABBInfo.m_minY) / aabbY);
            params.inputDataBuffer[inputDataOffset + 2] = __float2half((prd.pointLocal.z - AABBInfo.m_minZ) / aabbZ);
            params.inputDataBuffer[inputDataOffset + 3] = __float2half(phi / (2 * M_PI));
            params.inputDataBuffer[inputDataOffset + 4] = __float2half(theta / M_PI);
            
            NNPathData tempPathData;
            //通道1：当前交点的tMax
            tempPathData.throughput[0] = tMax;
            tempPathData.throughput[1] = AABBInfo.m_maxLength;
            tempPathData.throughput[2] = tMax / length(prd.originLocal - prd.pointLocal);

            tempPathData.pixelIndex = path.pixelIndex;
            tempPathData.hitScequnce = count;
            tempPathData.hitAABBID = hitAABBIndex + 1;
            tempPathData.isValid = true;
            // tempPathData.shadowPathID = path.shadowPathID;
            tempPathData.isInside = false;
            tempPathData.instanceID = prd.instanceID;
            tempPathData.pathIndex = params.accelerationStructures[hitAABBIndex].nodeID;

            params.NNPathDataBuffer[threadIndex * maxCount + count] = tempPathData; //TODO: AABBID + 1
            
            count++;
        } 
        else if (isHit && isInside) {

            bool skip = false;
            for(int i = 0; i < count; i++) {
                if (params.NNPathDataBuffer[threadIndex * maxCount + i].hitAABBID == (hitAABBIndex + 1)
                    && params.NNPathDataBuffer[threadIndex * maxCount + i].instanceID == prd.instanceID)
                    skip = true;
            }
            if(skip && count) continue;

            float phi, theta;
            Vec3 directionLocal = Vec3(-prd.directionLocal.x, -prd.directionLocal.y, -prd.directionLocal.z);
            Coordinates::cartesianToSphericalForTrain(normalized(directionLocal), &phi, &theta);

            const int inputDataOffset = threadIndex * 5 * maxCount + count * 5;

            params.inputDataBuffer[inputDataOffset + 0] = __float2half((prd.pointLocal.x - AABBInfo.m_minX) / aabbX);
            params.inputDataBuffer[inputDataOffset + 1] = __float2half((prd.pointLocal.y - AABBInfo.m_minY) / aabbY);
            params.inputDataBuffer[inputDataOffset + 2] = __float2half((prd.pointLocal.z - AABBInfo.m_minZ) / aabbZ);
            params.inputDataBuffer[inputDataOffset + 3] = __float2half(phi / (2 * M_PI));
            params.inputDataBuffer[inputDataOffset + 4] = __float2half(theta / M_PI);
            
            NNPathData tempPathData;
            //通道1：当前交点的tMax
            tempPathData.throughput[0] = tMax;
            tempPathData.throughput[1] = AABBInfo.m_maxLength;
            tempPathData.throughput[2] = tMax / length(prd.originLocal - prd.pointLocal);

            // TODO : 预测的tMax使用有误

            // tempPathData.throughput[2] = path.throughput[2];
            
            tempPathData.pixelIndex = path.pixelIndex;
            tempPathData.hitScequnce = count;
            tempPathData.hitAABBID = hitAABBIndex + 1;
            tempPathData.isValid = true;
            // tempPathData.shadowPathID = path.shadowPathID;
            tempPathData.instanceID = prd.instanceID;
            tempPathData.pathIndex = params.accelerationStructures[hitAABBIndex].nodeID;

            tempPathData.isInside = true;
            tempPathData.normalizedT = tMax / AABBInfo.m_maxLength;

            params.NNPathDataBuffer[threadIndex * maxCount + count] = tempPathData; //TODO: AABBID + 1

            count++;
        } 
        else if (!isHit && count == 0 && !path.isHit) {
            calculateEnvironmentLighting(path);
            path.isValid = false;
            const int pixelIndex = path.pixelIndex * 3;
            params.envLightingBuffer[pixelIndex + 0] += path.throughput.r();
            params.envLightingBuffer[pixelIndex + 1] += path.throughput.g();
            params.envLightingBuffer[pixelIndex + 2] += path.throughput.b();
        }  
    }

    // params.missDirectionBuffer[missDirectionIndex + 0] = direction.x();
    // params.missDirectionBuffer[missDirectionIndex + 1] = direction.y();
    // params.missDirectionBuffer[missDirectionIndex + 2] = direction.z();
    params.pathDataBuffer[threadIndex] = path;
}

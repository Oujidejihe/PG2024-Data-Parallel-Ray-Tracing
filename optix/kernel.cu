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
#include "sample.hpp"
#include "moana/cuda/triangle.hpp"

using namespace moana;

extern "C" {
    __constant__ Renderer::Params params;
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

__forceinline__ __device__ static BSDFSampleRecord createSamplingRecord(
    const PerRayData &prd,
    const WavefrontPathData &path
) {
    unsigned int seed = tea<4>(path.pixelIndex, params.sampleCount);
    const float xi1 = rnd(seed);
    const float xi2 = rnd(seed);

    if (prd.bsdfType == BSDFType::Water) {
        return Water::sample(xi1, xi2, prd);
    } else { // prd.bsdfType == BSDFType::Diffuse
        return Lambertian::sample(xi1, xi2, prd);
    }

}

__forceinline__ __device__ static WavefrontPathData generateShadowPath(
    const BSDFSampleRecord &sampleRecord,
    const WavefrontPathData &path,
    const Triangle *lights,
    SurfaceSample &lightSample,
    int shadowPathCount,
    int shadowPathID,
    int lightCount
) {
    unsigned int seed = tea<4>(path.pixelIndex * shadowPathCount + shadowPathID, params.sampleCount);
    const float xi1 = rnd(seed);
    const float xi2 = rnd(seed);
    const float xi3 = rnd(seed);

    //===MOANA===
    // const Triangle t1(
    //     Vec3(101346.539, 202660.438, 189948.188),
    //     Vec3(106779.617, 187339.562, 201599.453),
    //     Vec3(83220.3828, 202660.438, 198400.547)
    // );

    // const Triangle t2(
    //     Vec3(101346.539, 202660.438, 189948.188),
    //     Vec3(88653.4609, 187339.562, 210051.812),
    //     Vec3(88653.4609, 187339.562, 210051.812)
    // );

    const int lightIndex = (int)floorf(xi1 * lightCount);

    const Triangle light = lights[lightIndex];
    lightSample = light.sample(xi2, xi3);

    const float lightChoicePDF = 1.f / lightCount;
    lightSample.areaPDF = lightSample.areaPDF * lightChoicePDF;
    // const Triangle *sampleTriangle;
    // if (xi1 < 0.5f) {
    //     sampleTriangle = &t1;
    // } else {
    //     sampleTriangle = &t2;
    // }
    // const SurfaceSample lightSample = sampleTriangle->sample(xi2, xi3);

    Vec3 origin(sampleRecord.point.x, sampleRecord.point.y, sampleRecord.point.z);
    // origin += sampleRecord.normal * 0.001f; //Plan B:offset point
    const Vec3 lightPoint = lightSample.point;
    const Vec3 lightDirection = lightPoint - origin;
    const Vec3 wi = normalized(lightDirection);
    const float tMax = lightDirection.length();

    WavefrontPathData shadowPath = {
        .origin = origin,
        .direction = wi,
        .tMax = tMax,
        .throughput = Vec3(0.f, 0.f, 0.f),
        .pixelIndex = path.pixelIndex,
        .shadowPathID = shadowPathID,
        .visitedMask = 0,
        .currentNode = -1,
        .targetNode = -1,
        true,    // isShadowRay
        false,    // isDelta
        path.isValid,    // isValid
        false     // isHit
    };

    return shadowPath;
}

__forceinline__ __device__ static WavefrontPathData generateNextNewPath(
    const BSDFSampleRecord &sampleRecord,
    const WavefrontPathData &path
) {
    Vec3 origin(sampleRecord.point.x, sampleRecord.point.y, sampleRecord.point.z);

    const Vec3 &wiLocal = sampleRecord.wiLocal;

    Frame frame(sampleRecord.normal);
    const Vec3 wiWorld = frame.toWorld(wiLocal);

    WavefrontPathData newPath = {
        .origin = origin,
        .direction = normalized(wiWorld),
        .tMax = __FLT_MAX__,
        .throughput = Vec3(1.f, 1.f, 1.f),
        .pixelIndex = path.pixelIndex,
        .shadowPathID = -1,
        .visitedMask = 0,
        .currentNode = -1,
        .targetNode = -1,
        false,    // isShadowRay
        false,    // isDelta
        path.isValid,    // isValid
        false     // isHit
    };

    return newPath;
}

__forceinline__ __device__ static PerRayData *getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<PerRayData *>(util::unpackPointer(u0, u1));
}

extern "C" __global__ void __closesthit__ch()
{
    PerRayData *prd = getPRD();
    prd->isHit = true;
    prd->t = optixGetRayTmax();
    prd->point = getHitPoint();

    const float3 rayDirection = optixGetWorldRayDirection();
    prd->woWorld = -Vec3(rayDirection.x, rayDirection.y, rayDirection.z);

    HitGroupData *hitgroupData = reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
    prd->baseColor = hitgroupData->baseColor;
    prd->materialID = hitgroupData->materialID;
    prd->bsdfType = hitgroupData->bsdfType;
    prd->textureIndex = hitgroupData->textureIndex;

    // const unsigned int primitiveIndex = optixGetPrimitiveIndex();
    // prd->primitiveID = primitiveIndex;

    if (optixIsTriangleHit()) {
        OptixTraversableHandle gas = optixGetGASTraversableHandle();
        unsigned int primitiveIndex = optixGetPrimitiveIndex();
        unsigned int sbtIndex = optixGetSbtGASIndex();
        float time = optixGetRayTime();

        float3 vertices[3];
        optixGetTriangleVertexData(
            gas,
            primitiveIndex,
            sbtIndex,
            time,
            vertices
        );

        vertices[0] = optixTransformPointFromObjectToWorldSpace(vertices[0]);
        vertices[1] = optixTransformPointFromObjectToWorldSpace(vertices[1]);
        vertices[2] = optixTransformPointFromObjectToWorldSpace(vertices[2]);

        //normal
        int normalIndex0 = hitgroupData->normalIndices[primitiveIndex * 3 + 0];
        int normalIndex1 = hitgroupData->normalIndices[primitiveIndex * 3 + 1];
        int normalIndex2 = hitgroupData->normalIndices[primitiveIndex * 3 + 2];

        float n0x = hitgroupData->normals[normalIndex0 * 3 + 0];
        float n0y = hitgroupData->normals[normalIndex0 * 3 + 1];
        float n0z = hitgroupData->normals[normalIndex0 * 3 + 2];

        float n1x = hitgroupData->normals[normalIndex1 * 3 + 0];
        float n1y = hitgroupData->normals[normalIndex1 * 3 + 1];
        float n1z = hitgroupData->normals[normalIndex1 * 3 + 2];

        float n2x = hitgroupData->normals[normalIndex2 * 3 + 0];
        float n2y = hitgroupData->normals[normalIndex2 * 3 + 1];
        float n2z = hitgroupData->normals[normalIndex2 * 3 + 2];

        float3 n0Object{n0x, n0y, n0z};
        float3 n1Object{n1x, n1y, n1z};
        float3 n2Object{n2x, n2y, n2z};

        float3 n0World = optixTransformNormalFromObjectToWorldSpace(n0Object);
        float3 n1World = optixTransformNormalFromObjectToWorldSpace(n1Object);
        float3 n2World = optixTransformNormalFromObjectToWorldSpace(n2Object);

        const Vec3 n0 = normalized(Vec3(n0World.x, n0World.y, n0World.z));
        const Vec3 n1 = normalized(Vec3(n1World.x, n1World.y, n1World.z));
        const Vec3 n2 = normalized(Vec3(n2World.x, n2World.y, n2World.z));

        const float2 barycentrics = optixGetTriangleBarycentrics();
        const float alpha = barycentrics.x;
        const float beta = barycentrics.y;
        const float gamma = 1.f - alpha - beta;

        const Vec3 normal = gamma * n0
            + alpha * n1
            + beta * n2;

        // const Vec3 p0(vertices[0].x, vertices[0].y, vertices[0].z);
        // const Vec3 p1(vertices[1].x, vertices[1].y, vertices[1].z);
        // const Vec3 p2(vertices[2].x, vertices[2].y, vertices[2].z);

        // // Debug: face normals
        // const Vec3 e1 = p1 - p0;
        // const Vec3 e2 = p2 - p0;
        // const Vec3 normal = normalized(cross(e1, e2));

        //texture
        if(prd->textureIndex >= 0) {
            int textureIndex0 = hitgroupData->texCoordsIndices[primitiveIndex * 3 + 0];
            int textureIndex1 = hitgroupData->texCoordsIndices[primitiveIndex * 3 + 1];
            int textureIndex2 = hitgroupData->texCoordsIndices[primitiveIndex * 3 + 2];

            float t0x = hitgroupData->texCoords[textureIndex0 * 2 + 0];
            float t0y = hitgroupData->texCoords[textureIndex0 * 2 + 1];

            float t1x = hitgroupData->texCoords[textureIndex1 * 2 + 0];
            float t1y = hitgroupData->texCoords[textureIndex1 * 2 + 1];

            float t2x = hitgroupData->texCoords[textureIndex2 * 2 + 0];
            float t2y = hitgroupData->texCoords[textureIndex2 * 2 + 1];

            float tx = gamma * t0x + alpha * t1x + beta * t2x;
            float ty = gamma * t0y + alpha * t1y + beta * t2y;

            float4 albedo{0.f, 0.f, 0.f, 0.f};
            albedo = tex2D<float4>(
                params.albedoTextures[prd->textureIndex],
                tx,
                ty
            );

            prd->baseColor = float3{albedo.x, albedo.y, albedo.z};
            // prd->baseColor = float3{tx, ty, 0.f};
        }

        prd->normal = normalized(normal);
        prd->barycentrics = optixGetTriangleBarycentrics();
    } else {
        const unsigned int primitiveIndex = optixGetPrimitiveIndex();
        const float3 normal = normalCubic(primitiveIndex);
        prd->normal = normalized(Vec3(normal.x, normal.y, normal.z));
        prd->barycentrics = float2{0.f, 0.f};
    }

    if (dot(prd->normal, prd->woWorld) < 0.f) {
        prd->normal = -1.f * prd->normal;
        prd->isInside = true;
    } else {
        prd->isInside = false;
    }
}

extern "C" __global__ void __miss__ms()
{
    float3 direction = optixGetWorldRayDirection();
    PerRayData *prd = getPRD();
    prd->isHit = false;
    prd->materialID = -1;
}

// One anyhit program for the radiance ray for all materials with cutout opacity!
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

//处理有交点的次级光线
extern "C" __global__ void __raygen__rg()
{
    const uint3 index = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const int threadIndex = index.y * dim.x + index.x;

    if(threadIndex >= params.pathSize) return;

    WavefrontPathData path;
    path = params.pathDataBuffer[threadIndex];

    if (!path.isValid) { return; }

    const Vec3 origin = path.origin;
    const Vec3 direction = path.direction;

    PerRayData prd;
    
    bool isHit = false;
    float tMax = __FLT_MAX__;
    BSDFSampleRecord sampleRecord;
    Vec3 albedo = {1.f, 1.f, 1.f};

    for (int i = 0; i < params.sceneSize; i++) {

        if(params.accelerationStructures[i].isProxy) continue;

        prd.isHit = false;

        unsigned int p0, p1;
        util::packPointer(&prd, p0, p1);
        optixTrace(
            params.accelerationStructures[i].handle,
            float3{ origin.x(), origin.y(), origin.z() },
            float3{ direction.x(), direction.y(), direction.z() },
            util::Epsilon,
            tMax,
            0.f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
            0, 1, 0, // SBT params
            p0, p1
        );

        if (prd.isHit) {
            tMax = prd.t;
            isHit = true;
            sampleRecord = createSamplingRecord(prd, path);
            albedo = Vec3(prd.baseColor.x, prd.baseColor.y, prd.baseColor.z);
        }
    }
    
    if (!isHit) {
        calculateEnvironmentLighting(path);
        path.isValid = false;
        const int pixelIndex = path.pixelIndex * 3;
        params.envLightingBuffer[pixelIndex + 0] += path.throughput.r();
        params.envLightingBuffer[pixelIndex + 1] += path.throughput.g();
        params.envLightingBuffer[pixelIndex + 2] += path.throughput.b();
    }

    WavefrontPathData nextNewPath = generateNextNewPath(sampleRecord, path);
    
    // const Vec3 L = 10.f * Vec3(505.928150, 505.928150, 505.928150);
    // const Vec3 lightNormal = Vec3(0.0f, 0.0f, 1.0f);
    // const Vec3 L = 0.5 * Vec3(891.443777, 505.928150, 154.625939);
    // const Vec3 L = Vec3(891.443777, 505.928150, 154.625939);
    // const Vec3 lightNormal = Vec3(-0.323744059f, -0.642787874f, -0.694271863f);


    const float cosThetaWi = fabsf(sampleRecord.wiLocal.z());

    Vec3 throughput = path.throughput * sampleRecord.weight * cosThetaWi * albedo;

    nextNewPath.throughput = throughput;

    params.pathDataBuffer[threadIndex] = nextNewPath;

    // params.directLightingBuffer[path.pixelIndex * 3 + 0] = albedo.b();
    // params.directLightingBuffer[path.pixelIndex * 3 + 1] = albedo.b();
    // params.directLightingBuffer[path.pixelIndex * 3 + 2] = albedo.b();
    SurfaceSample lightSample;
    for (int i = 0; i < params.shadowPathCount; i++) {
        WavefrontPathData shadowPath = generateShadowPath(sampleRecord, path, params.directLights, lightSample, params.shadowPathCount, i, params.lightCount);

        if (sampleRecord.isDelta) {shadowPath.isValid = false; continue;}

        const Vec3 wi = shadowPath.direction;

        // Vec3 contribution = L * path.throughput * albedo
        // * fabsf(dot(lightNormal, -wi)) 
        // * fmaxf(0.f, dot(wi, sampleRecord.normal)) 
        // * (20000.f * 20000.f) / (shadowPath.tMax * shadowPath.tMax) 
        // * (1.f / M_PI);
        Vec3 contribution = lightSample.Le * path.throughput * albedo
        * fmaxf(0.f, dot(lightSample.normal, -wi)) 
        * fmaxf(0.f, dot(wi, sampleRecord.normal)) 
        / (lightSample.areaPDF) / (shadowPath.tMax * shadowPath.tMax) 
        * (1.f / M_PI);

        shadowPath.throughput = contribution;
        params.pathDataBuffer[threadIndex * params.shadowPathCount + i + params.pathSize] = shadowPath;
    }
}

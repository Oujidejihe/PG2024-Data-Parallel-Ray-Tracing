#include "moana/cuda/bvh_intersection.hpp"

#include "assert_macros.hpp"
#include "moana/core/coordinates.hpp"
#include "moana/core/vec3.hpp"

#include "../optix/random.hpp"
#include "../optix/sample.hpp"
#include "moana/cuda/triangle.hpp"

namespace moana {

__device__ inline float clamp(float value, float lowest, float highest)
{
    return fminf(highest, fmaxf(value, lowest));
}

__device__ inline void cartesianToSpherical(const Vec3 &cartesian, float &phi, float &theta)
{
    phi = atan2f(cartesian.y(), cartesian.x());
    if (phi < 0.f) {
        phi += 2 * M_PI;
    }

    theta = acosf(clamp(cartesian.z(), -1.f, 1.f));
}


__device__ static AABBHit miss() {
    return AABBHit({ false, Vec3(0.f, 0.f, 0.f), Vec3(0.f, 0.f, 0.f) });
}

__device__ static AABBHit intersectWithAABB(const AABB &aabb, const Ray &ray)
{
    Vec3 invDirection(
        1.f / ray.direction().x(),
        1.f / ray.direction().y(),
        1.f / ray.direction().z()
    );

    Vec3 origin = ray.origin();

    float t1 = (aabb.m_minX - origin.x()) * invDirection.x(); // FAR
    float t2 = (aabb.m_maxX - origin.x()) * invDirection.x(); // NEAR
    float t3 = (aabb.m_minY - origin.y()) * invDirection.y(); // LEFT
    float t4 = (aabb.m_maxY - origin.y()) * invDirection.y(); // RIGHT
    float t5 = (aabb.m_minZ - origin.z()) * invDirection.z(); // BOTTOM
    float t6 = (aabb.m_maxZ - origin.z()) * invDirection.z(); // TOP

    float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
    float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

    if (tmin >= tmax) { return miss(); }
    if (tmin < 0 && tmax == 0) { return miss(); }

    FACE_TYPE enterFaceType = FACE_TYPE::Null;
    if ((int&)tmin == (int&)t1) enterFaceType = FACE_TYPE::FAR;
    if ((int&)tmin == (int&)t2) enterFaceType = FACE_TYPE::NEAR;
    if ((int&)tmin == (int&)t3) enterFaceType = FACE_TYPE::LEFT;
    if ((int&)tmin == (int&)t4) enterFaceType = FACE_TYPE::RIGHT;
    if ((int&)tmin == (int&)t5) enterFaceType = FACE_TYPE::BOTTOM;
    if ((int&)tmin == (int&)t6) enterFaceType = FACE_TYPE::TOP;

    FACE_TYPE exitFaceType = FACE_TYPE::Null;
    if ((int&)tmax == (int&)t1) exitFaceType = FACE_TYPE::FAR;
    if ((int&)tmax == (int&)t2) exitFaceType = FACE_TYPE::NEAR;
    if ((int&)tmax == (int&)t3) exitFaceType = FACE_TYPE::LEFT;
    if ((int&)tmax == (int&)t4) exitFaceType = FACE_TYPE::RIGHT;
    if ((int&)tmax == (int&)t5) exitFaceType = FACE_TYPE::BOTTOM;
    if ((int&)tmax == (int&)t6) exitFaceType = FACE_TYPE::TOP;

    if (tmin >= 0 && !std::isinf(tmin) && tmax >= 0 && !std::isinf(tmax)) {
        return AABBHit({
            true,
            ray.at(tmin),
            ray.at(tmax),
            tmin,
            tmax,
            enterFaceType,
            exitFaceType,
            RAY_TYPE::OUTSIDE
        });
    }

    if (tmax >= 0 && !std::isinf(tmax) && tmin < 0) {
        return AABBHit({
            true,
            ray.origin(),
            ray.at(tmax),
            0.f,
            tmax,
            enterFaceType,
            exitFaceType,
            RAY_TYPE::INSIDE
        });
    }

    return miss();
}

__device__ static int getEnterVisibilityIndex(const AABB &aabb, const AABBHit& intersection, const Ray& ray, int &directionIndex, int &coordinateIndex) {
    float phi, theta, row, column;
    Vec3 direcition;

    int m_minX = aabb.m_minX;
    int m_maxX = aabb.m_maxX;
    int m_minY = aabb.m_minY;
    int m_maxY = aabb.m_maxY;
    int m_minZ = aabb.m_minZ;
    int m_maxZ = aabb.m_maxZ;

    if (intersection.enterFaceType == FACE_TYPE::NEAR || intersection.enterFaceType == FACE_TYPE::FAR)
    {
        if(ray.direction().x() > 0) {
            direcition = Vec3{ray.direction().y(), ray.direction().z(), ray.direction().x()};
        } else {
            direcition = Vec3{-ray.direction().y(), -ray.direction().z(), -ray.direction().x()};
        }

        column = (intersection.enterPoint.y() - m_minY) / (m_maxY - m_minY);
        row = (m_maxZ - intersection.enterPoint.z()) / (m_maxZ - m_minZ);
    } 
    else if (intersection.enterFaceType == FACE_TYPE::LEFT || intersection.enterFaceType == FACE_TYPE::RIGHT) 
    {
        if(ray.direction().y() > 0) {
            direcition = Vec3{ray.direction().z(), ray.direction().x(), ray.direction().y()};
        } else {
            direcition = Vec3{-ray.direction().z(), -ray.direction().x(), -ray.direction().y()};
        }

        column = (m_maxX - intersection.enterPoint.x()) / (m_maxX - m_minX);
        row = (m_maxZ - intersection.enterPoint.z()) / (m_maxZ - m_minZ);
    }
    else if (intersection.enterFaceType == FACE_TYPE::TOP || intersection.enterFaceType == FACE_TYPE::BOTTOM) 
    {   
        if(ray.direction().z() > 0) {
            direcition = Vec3{ray.direction().x(), ray.direction().y(), ray.direction().z()};
        } else {
            direcition = Vec3{-ray.direction().x(), -ray.direction().y(), -ray.direction().z()};
        }

        column = (intersection.enterPoint.y() - m_minY) / (m_maxY - m_minY);
        row = (intersection.enterPoint.x() - m_minX) / (m_maxX - m_minX);
    }

    cartesianToSpherical(direcition, phi, theta);

    directionIndex = int(float(aabb.angle) * phi / (2 * M_PI));
    coordinateIndex = int(row * aabb.height) * aabb.width + int(column * aabb.width);
    
    return 2 * (int(intersection.enterFaceType) * (aabb.width * aabb.height * aabb.angle) +  coordinateIndex * aabb.angle + directionIndex);
}

__device__ static void testRayWithBVH(
    bool* d_vibilityBuffer,
    BSDFSampleRecord &sampleRecord,
    AABB &aabb,
    Ray &bounceRay,
    bool &isPassthrough,
    bool &isHit,
    int offset
) {
    AABBHit intersection = intersectWithAABB(aabb, bounceRay);

    if (!intersection.isHit) { sampleRecord.isValid = false; return; }
    if (intersection.rayType == RAY_TYPE::INSIDE) { sampleRecord.isValid = true; return; }

    isHit = true;

    int directionIndex;
    int CoordinateIndex;

    int enterIndex = getEnterVisibilityIndex(aabb, intersection, bounceRay, directionIndex, CoordinateIndex);

    if (d_vibilityBuffer[offset + enterIndex + 1]) { sampleRecord.isValid = false; isPassthrough = true; return; }

    sampleRecord.isValid = true;
    return;
}

__device__ static bool sampleShadowRay(
    BSDFSampleRecord* d_sampleRecordBuffer,
    Ray &ray,
    int frames,
    int sampleCount,
    int bounce,
    int width,
    int height,
    const int sampleRecordIndex
) {   
    const BSDFSampleRecord &sampleRecord = d_sampleRecordBuffer[sampleRecordIndex];
    
    unsigned int seed = tea<4>(
        sampleRecordIndex * frames,
        sampleCount + (height * width * bounce)

    );
    const float xi1 = rnd(seed);
    const float xi2 = rnd(seed);
    const float xi3 = rnd(seed);

    const Triangle t1(
        Vec3(101346.539, 202660.438, 189948.188),
        Vec3(106779.617, 187339.562, 201599.453),
        Vec3(83220.3828, 202660.438, 198400.547)
    );

    const Triangle t2(
        Vec3(101346.539, 202660.438, 189948.188),
        Vec3(88653.4609, 187339.562, 210051.812),
        Vec3(88653.4609, 187339.562, 210051.812)
    );

    const Triangle *sampleTriangle;
    if (xi1 < 0.5f) {
        sampleTriangle = &t1;
    } else {
        sampleTriangle = &t2;
    }
    const SurfaceSample lightSample = sampleTriangle->sample(xi2, xi3);

    Vec3 origin(sampleRecord.point.x, sampleRecord.point.y, sampleRecord.point.z);
    const Vec3 lightPoint = lightSample.point;
    const Vec3 lightDirection = lightPoint - origin;
    const Vec3 wi = normalized(lightDirection);
    const float tMax = lightDirection.length();

    ray = Ray(origin, wi, true);
    ray.tMax = tMax;

    return true;
}

__global__ static void bvhIntersectionKernel(
    int width,
    int height,
    int aabbNumber,
    int frames,
    int sampleCount,
    int bounce,
    int offset,
    AABB* d_aabbRecords,
    BSDFSampleRecord* d_sampleRecordBuffer,
    bool* d_vibilityBuffer
) {
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;

    if ((row >= height) || (col >= width)) { return; }

    const int index = row * width + col;

    BSDFSampleRecord &sampleRecord = d_sampleRecordBuffer[index];
    if (!sampleRecord.isValid) { return; }
    
    // totalCount++;
    //===================================
    
    bool isHit = false;
    bool isPassthrough = false;
    bool isHit1 = false;
    bool isPassthrough1 = false;

    for(int i = 0; i < aabbNumber; i++) {
        {
        float3 origin = sampleRecord.point;
        Vec3 wiLocal = sampleRecord.wiLocal;
        const Frame frame(sampleRecord.normal);
        Vec3 wiWorld = frame.toWorld(wiLocal);
        Ray bounceRay(Vec3(origin.x, origin.y, origin.z), normalized(wiWorld));

        testRayWithBVH(
            d_vibilityBuffer,
            sampleRecord,
            d_aabbRecords[i],
            bounceRay,
            isPassthrough,
            isHit,
            offset * i
        );
        }
        if (!sampleRecord.isValid) {    
            Ray shadowRay;

            sampleShadowRay(
                d_sampleRecordBuffer,
                shadowRay,
                frames,
                sampleCount,
                bounce,
                width,
                height,
                index
            );
            testRayWithBVH(
                d_vibilityBuffer,
                sampleRecord,
                d_aabbRecords[i],
                shadowRay,
                isPassthrough1,
                isHit1,
                offset * i
            );
        }
        if (sampleRecord.isValid) break;
    }
    //===================================

    // if(isHit || isHit1) hitCount++;
}

void BVHIntersection::bvhIntersectWithCuda(
    int width,
    int height,
    int frames,
    int sampleCount,
    int bounce,
    ASArena &arena,
    std::vector<AABB> &aabbRecords,
    std::vector<BSDFSampleRecord> &sampleRecordBuffer 
) { 
    int count = 0;
    for(auto &record : sampleRecordBuffer) {
        if (record.isValid) count++;
    }
    std::cout << count << std::endl;

    const size_t aabbRecordsSizeInBytes = aabbRecords.size() * sizeof(AABB);
    const size_t sampleRecordBufferSizeInBytes = width * height * sizeof(BSDFSampleRecord);
    const size_t visibilitySizeInByte = aabbRecords.size() * aabbRecords[0].height * aabbRecords[0].width * aabbRecords[0].angle * 6 * 2 * sizeof(bool);

    CUdeviceptr d_aabbRecords = 0;
    CUdeviceptr d_sampleRecordBuffer = 0;
    CUdeviceptr d_vibilityBuffer = 0;

    d_aabbRecords = arena.pushTemp(aabbRecordsSizeInBytes);
    d_sampleRecordBuffer = arena.pushTemp(sampleRecordBufferSizeInBytes);
    d_vibilityBuffer = arena.pushTemp(visibilitySizeInByte);

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_vibilityBuffer),
        aabbRecords[0].visibility.data,
        visibilitySizeInByte,
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_aabbRecords),
        aabbRecords.data(),
        aabbRecordsSizeInBytes,
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_sampleRecordBuffer),
        sampleRecordBuffer.data(),
        sampleRecordBufferSizeInBytes,
        cudaMemcpyHostToDevice
    ));

    const int blockWidth = 16;
    const int blockHeight = 16;

    const dim3 blocks((width + blockWidth - 1 ) / blockWidth , ( height + blockHeight - 1 ) / blockHeight);
    const dim3 threads(blockWidth, blockHeight);

    bvhIntersectionKernel<<<blocks, threads>>>(
        width,
        height,
        aabbRecords.size(),
        frames,
        sampleCount,
        bounce,
        aabbRecords[0].height * aabbRecords[0].width * aabbRecords[0].angle * 6 * 2,
        reinterpret_cast<AABB *>(d_aabbRecords),
        reinterpret_cast<BSDFSampleRecord *>(d_sampleRecordBuffer),
        reinterpret_cast<bool *>(d_vibilityBuffer)
    );

    // TEST
    CHECK_CUDA(cudaMemcpy(
        sampleRecordBuffer.data(),
        reinterpret_cast<void *>(d_sampleRecordBuffer),
        sampleRecordBufferSizeInBytes,
        cudaMemcpyDeviceToHost
    ));

    count = 0;
    for(auto &record : sampleRecordBuffer) {
        if (record.isValid) count++;
    }
    std::cout << count << std::endl;
}

}
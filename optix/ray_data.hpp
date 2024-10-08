#pragma once

#include "moana/core/vec3.hpp"
#include "moana/cuda/bsdf.hpp"

namespace moana {

struct PerRayData {
    float3 point;
    float3 pointLocal;
    float3 directionLocal;
    float3 originLocal;
    float3 baseColor;
    Vec3 normal;
    Vec3 woWorld;

    float2 barycentrics;

    float t;
    int materialID;
    int primitiveID;
    int textureIndex;
    int instanceID;
    BSDFType bsdfType;
    bool isInside;
    bool isHit;
    
};

}

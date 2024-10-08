#pragma once

#include "moana/core/bsdf_sample_record.hpp"
#include "moana/core/frame.hpp"
#include "ray_data.hpp"
#include "sample.hpp"

namespace moana { namespace Lambertian {

__forceinline__ __device__ BSDFSampleRecord sample(
    const float xi1,
    const float xi2,
    const PerRayData &prd
) {

    const Frame frame(prd.normal);
    const Vec3 wiLocal = Sample::uniformHemisphere(xi1, xi2);
    const float weight = 2.f;

    const BSDFSampleRecord record = {
        .point = prd.point,
        .wiLocal = wiLocal,
        .normal = prd.normal,
        // .frame = frame,
        .weight = weight,
        .depth = prd.t,
        .isDelta = false,
        .isValid = true,
    };

    return record;
}

} }

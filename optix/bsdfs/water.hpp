#pragma once

#include "moana/core/bsdf_sample_record.hpp"
#include "moana/core/frame.hpp"
#include "moana/cuda/fresnel.hpp"
#include "moana/cuda/tangent_frame.hpp"
#include "ray_data.hpp"
#include "util.hpp"

namespace moana { namespace Water {

__forceinline__ __device__ BSDFSampleRecord sample(
    const float xi1,
    const float xi2,
    const PerRayData &prd
) {
    const Frame frame(prd.normal);

    const Vec3 wo = frame.toLocal(prd.woWorld);

    float etaIncident = 1.f;
    float etaTransmitted = 1.33f;

    if (prd.isInside) {
        const float temp = etaIncident;
        etaIncident = etaTransmitted;
        etaTransmitted = temp;
    }

    Vec3 wi(0.f);
    const bool doesRefract = Snell::refract(
        wo,
        &wi,
        etaIncident,
        etaTransmitted
    );

    const float fresnelReflectance = Fresnel::dielectricReflectance(
        TangentFrame::absCosTheta(wo),
        etaIncident,
        etaTransmitted
    );

    if (xi1 < fresnelReflectance) {
        wi = wo.reflect(Vec3(0.f, 0.f, 1.f));

        const float cosTheta = TangentFrame::absCosTheta(wi);
        const float throughput = cosTheta == 0.f
            ? 0.f
            : fresnelReflectance / cosTheta
        ;
        const BSDFSampleRecord record = {
            .point = prd.point,
            .wiLocal = wi,
            .normal = prd.normal,
            // .frame = frame,
            .weight = throughput / fresnelReflectance,
            .depth = prd.t,
            .isDelta = true,
            .isValid = true,
        };

        return record;
    } else {
        const float fresnelTransmittance = 1.f - fresnelReflectance;

        const float cosTheta = TangentFrame::absCosTheta(wi);
        const float throughput = cosTheta == 0.f
            ? 0.f
            : fresnelTransmittance / cosTheta
        ;

        // PBRT page 961 "Non-symmetry Due to Refraction"
        // Always incident / transmitted because we swap at top of
        // function if we're going inside-out
        const float nonSymmetricEtaCorrection = util::square(
            etaIncident / etaTransmitted
        );

        const BSDFSampleRecord record = {
            .point = prd.point,
            .wiLocal = wi,
            .normal = prd.normal,
            // .frame = frame,
            .weight = throughput * nonSymmetricEtaCorrection / fresnelTransmittance,
            .depth = prd.t,
            .isDelta = true,
            .isValid = true,
        };

        return record;
    }

}

} }

#include "moana/render/pipeline.hpp"

#include "kernel.hpp"
#include "shadow_ray_kernel.hpp"
#include "path_gen_kernel.hpp"
#include "precom_ray_kernel.hpp"
#include "vis_ray_kernel.hpp"
#include "pipeline_helper.hpp"

#include "distributed_traversal_kernel.hpp"
#include "secondary_ray_kernel.hpp"

namespace moana { namespace Pipeline {

void initOptixState(
    OptixState &optixState,
    OptixDeviceContext context,
    SceneState &sceneState
) {
    return PipelineHelper::initOptixState(
        optixState,
        context,
        sceneState,
        PTX::mainRaySource
    );
}

} }

namespace moana { namespace ShadowPipeline {

void initOptixState(
    OptixState &optixState,
    OptixDeviceContext context,
    SceneState &sceneState
) {
    return PipelineHelper::initOptixState(
        optixState,
        context,
        sceneState,
        PTX::shadowRaySource
    );
}

} }

namespace moana { namespace SecondaryPipeline {

void initOptixState(
    OptixState &optixState,
    OptixDeviceContext context,
    SceneState &sceneState
) {
    return PipelineHelper::initOptixState(
        optixState,
        context,
        sceneState,
        PTX::secondaryRaySource
    );
}

} }

namespace moana { namespace PathGenPipeline {

void initOptixState(
    OptixState &optixState,
    OptixDeviceContext context,
    SceneState &sceneState
) {
    return PipelineHelper::initOptixState(
        optixState,
        context,
        sceneState,
        PTX::pathGenRaySource
    );
}

} }

namespace moana { namespace PrecomPipeline {

void initOptixState(
    OptixState &optixState,
    OptixDeviceContext context,
    SceneState &sceneState
) {
    return PipelineHelper::initOptixState(
        optixState,
        context,
        sceneState,
        PTX::precomRaySource
    );
}

} }

namespace moana { namespace VisPipeline {

void initOptixState(
    OptixState &optixState,
    OptixDeviceContext context,
    SceneState &sceneState
) {
    return PipelineHelper::initOptixState(
        optixState,
        context,
        sceneState,
        PTX::visRaySource
    );
}

} }

namespace moana { namespace TraversalPipeline {

void initOptixState(
    OptixState &optixState,
    OptixDeviceContext context,
    SceneState &sceneState
) {
    return PipelineHelper::initOptixState(
        optixState,
        context,
        sceneState,
        PTX::traversalSource
    );
}

} }

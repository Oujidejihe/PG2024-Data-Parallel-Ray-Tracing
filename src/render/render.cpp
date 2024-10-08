#include "moana/render/renderer.hpp"

#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <omp.h>
#include <assert.h>
#include <string>

#include "assert_macros.hpp"
#include "core/ptex_texture.hpp"
#include "moana/io/image.hpp"
#include "render/timing.hpp"
#include "scene/texture_offsets.hpp"
#include "util/enumerate.hpp"

#include "../optix/random.hpp"
#include "../optix/sample.hpp"
#include "moana/cuda/triangle.hpp"

#include "moana/cuda/cuda_compaction.hpp"
#include "moana/cuda/frame_buffer_update.hpp"

#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>

#include <moana/macro_define.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <moana/stb_image.h>

#include "nerual_setting.hpp"

#define SEPARATEDNN 1

namespace moana { namespace Renderer {

struct OutputBuffers {
    std::vector<float> cosThetaWiBuffer;
    std::vector<float> barycentricBuffer;
    std::vector<int> idBuffer;
    std::vector<float> colorBuffer;
    std::vector<float> occlusionBuffer;
    std::vector<BSDFSampleRecord> sampleRecordInBuffer;
    std::vector<BSDFSampleRecord> sampleRecordOutBuffer;
    std::vector<BSDFSampleRecord> sampleRecordAABBBuffer;
    std::vector<char> shadowOcclusionBuffer;
    std::vector<float> shadowWeightBuffer;

    std::vector<float> depthBuffer;

    //Wavefront
    std::vector<Ray> rayBuffer;
    std::vector<WavefrontPathData> pathDataBuffer;
    std::vector<WavefrontPathData> transferPathDataBuffer;
    std::vector<int> transferOffset;
    std::vector<int> transferCount;
    std::vector<float> directLightingBuffer;
    std::vector<float> envLightingBuffer;

    std::vector<float> debugVisBuffer;

    //Train_data
    std::vector<float> originBuffer;
    std::vector<float> directionBuffer;

};

struct ShadowBuffers
{
    std::vector<char> shadowOcclusionBuffer;
    std::vector<float> shadowWeightBuffer;
};


struct BufferManager {
    size_t rayBufferSizeInBytes;
    size_t indexBufferSizeInBytes;

    size_t visibilityBufferSizeInBytes;

    size_t aabbBufferSizeInBytes;

    size_t colorBufferSizeInBytes;
    size_t accelerationStructuresSizeInBytes;
    size_t pathDataBufferSizeInBytes;
    size_t transferOffsetSizeInBytes;
    size_t scanBufferSizeInBytes;

    size_t inputDataBufferSizeInBytes;
    size_t scanBufferNNSizeInBytes;
    size_t sceneOffsetSizeInBytes;
    size_t NNPathDataBufferSizeInBytes;
    size_t predBufferSizeInBytes;
    size_t shadowOcclusionCharTypeBufferSizeInBytes;
    size_t shadowOcclusionFloatTypeBufferSizeInBytes;
    size_t contributionBufferSizeInBytes;

    OutputBuffers output;
};

struct RecordOnlyWithDepth {
    int index;
    float depth;
    bool isValid;
    RecordOnlyWithDepth():isValid(false) {}
    RecordOnlyWithDepth(int Id, float Depth, bool IsValid) {
        index = Id;
        depth = Depth;
        isValid = IsValid;
    }
};

struct RecordWithIndex {
    int index;
    BSDFSampleRecord samplerecord;
    Vec3 albedo;
    float cosTheta;
    RecordWithIndex() {}
    RecordWithIndex(int id, BSDFSampleRecord record, Vec3 ialbedo, float icosTheta) {
        index = id;
        samplerecord = record;
        albedo = ialbedo;
        cosTheta = icosTheta;
    }
};

struct BcastSampleRecord {
    int index;
    BSDFSampleRecord samplerecord;
    Vec3 beta;

    BcastSampleRecord() {}
    BcastSampleRecord(int id, BSDFSampleRecord record, Vec3 ibeta) {
        index = id;
        samplerecord = record;
        beta = ibeta;
    }
};

struct MpiBuffers {
    // std::vector<std::vector<int>> labelMapPerNode;
    std::vector<float> imageRecvBuffer;
    std::vector<RecordWithIndex> compressedRecord;
    std::vector<BcastSampleRecord> compressedBcastRecord;
    std::vector<RecordOnlyWithDepth> compressedDepthRecord;
    std::vector<RecordOnlyWithDepth> depthRecordBcast;
    std::vector<int> offsets;
    std::vector<int> displs;
    std::vector<char> labelMap;
    std::vector<float> tempImage;
    std::vector<RecordOnlyWithDepth> tempDepthRecord;
    std::vector<char> shadowBuffer_Occlusion;
    std::vector<float> ShadowBuffer_Weight;

    std::vector<float> normalImage;
    std::vector<float> albedoImage;

    std::vector<bool> validPixel;

    std::vector<std::vector<BcastSampleRecord>> parallelCompressedBcastRecord;
};

//===========================================================
//init
//===========================================================

static void mallocMpiBuffer(
    MpiBuffers &mpiBuffer,
    int comm_size,
    int width,
    int height
) {
    // mpiBuffer.labelMapPerNode.resize(comm_size);
    // mpiBuffer.imageRecvBuffer.resize(width * height * 3 * comm_size);
    // mpiBuffer.compressedRecord.resize(width * height * 1);
    // mpiBuffer.compressedBcastRecord.resize(width * height * 1);
    // mpiBuffer.compressedDepthRecord.resize(width * height * 1);
    // mpiBuffer.depthRecordBcast.resize(width * height * 1);
    mpiBuffer.offsets.resize(comm_size);
    mpiBuffer.displs.resize(comm_size);
    mpiBuffer.labelMap.resize(width * height);
    mpiBuffer.tempImage.resize(width * height * 3);
    // mpiBuffer.tempDepthRecord.resize(width * height * 1);
    // mpiBuffer.shadowBuffer_Occlusion.resize(width * height * 1 * comm_size);
    // mpiBuffer.ShadowBuffer_Weight.resize(width * height * 1 * comm_size);

    // mpiBuffer.normalImage.resize(width * height * 3);
    // mpiBuffer.albedoImage.resize(width * height * 3);

    // mpiBuffer.validPixel.resize(width * height * 1);

    // mpiBuffer.parallelCompressedBcastRecord = std::vector<std::vector<BcastSampleRecord>>(30, std::vector<BcastSampleRecord>(width * height / 30));
}

static void copyOutputBuffers(
    BufferManager &buffers,
    int width,
    int height,
    Params &params,
    int worldSize
) {
    buffers.output.cosThetaWiBuffer.resize(width * height * 1);
    buffers.output.barycentricBuffer.resize(width * height * 2);
    buffers.output.idBuffer.resize(width * height * 3);
    buffers.output.colorBuffer.resize(width * height * 3);
    buffers.output.occlusionBuffer.resize(width * height);
    buffers.output.sampleRecordInBuffer.resize(width * height);
    buffers.output.sampleRecordOutBuffer.resize(width * height);
    buffers.output.sampleRecordAABBBuffer.resize(width *height);
    buffers.output.shadowOcclusionBuffer.resize(width * height * 1);
    buffers.output.shadowWeightBuffer.resize(width * height * 1);
    buffers.output.depthBuffer.resize(width * height * 1);
    buffers.output.rayBuffer.resize(width * height * 1);

    buffers.output.pathDataBuffer.resize(width * height);
    buffers.output.transferPathDataBuffer.resize(width * height);
    buffers.output.transferOffset.resize(worldSize + 1);
    buffers.output.transferCount.resize(worldSize);

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(buffers.output.pathDataBuffer.data()),
        params.pathDataBuffer,
        buffers.pathDataBufferSizeInBytes,
        cudaMemcpyDeviceToHost
    ));

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(buffers.output.transferOffset.data()),
        params.transferOffset,
        buffers.transferOffsetSizeInBytes,
        cudaMemcpyDeviceToHost
    ));

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(buffers.output.transferPathDataBuffer.data()),
        params.transferPathDataBuffer,
        buffers.pathDataBufferSizeInBytes,
        cudaMemcpyDeviceToHost
    ));

}

static void copyOutputBuffersForPrecom(
    BufferManager &buffers,
    const int totalRays,
    Params &params
) {
    buffers.output.rayBuffer.resize(totalRays);
    buffers.output.originBuffer.resize(totalRays * 3);
    buffers.output.directionBuffer.resize(totalRays * 2);

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(buffers.output.rayBuffer.data()),
        params.rayBuffer,
        sizeof(Ray) * totalRays,
        cudaMemcpyDeviceToHost
    ));
}

static void copyOutputBuffersForTrainData(
    BufferManager &buffers,
    const int totalRays,
    Params &params
) {
    buffers.output.originBuffer.resize(totalRays * 3);
    buffers.output.directionBuffer.resize(totalRays * 3);

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(buffers.output.originBuffer.data()),
        params.originBuffer,
        3 * sizeof(float) * totalRays,
        cudaMemcpyDeviceToHost
    ));

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(buffers.output.directionBuffer.data()),
        params.directionBuffer,
        3 * sizeof(float) * totalRays,
        cudaMemcpyDeviceToHost
    ));
}

static void copyOutputBuffersForColor(
    BufferManager &buffers,
    int width,
    int height,
    Params &params
) {
    buffers.output.directLightingBuffer.resize(width * height * 3); //
    buffers.output.envLightingBuffer.resize(width * height * 3); //

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(buffers.output.directLightingBuffer.data()),
        params.directLightingBuffer,
        buffers.colorBufferSizeInBytes,
        cudaMemcpyDeviceToHost
    ));

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(buffers.output.envLightingBuffer.data()),
        params.envLightingBuffer,
        buffers.colorBufferSizeInBytes,
        cudaMemcpyDeviceToHost
    ));

}

static void resetSampleBuffers(
    BufferManager &buffers,
    int width,
    int height,
    Params &params
) {
    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.transferOffset),
        0,
        buffers.transferOffsetSizeInBytes
    ));

    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.sceneOffset),
        0,
        buffers.sceneOffsetSizeInBytes
    ));

    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.pathDataBuffer),
        0,
        buffers.pathDataBufferSizeInBytes * (1 + params.shadowPathCount)
    ));

    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.transferPathDataBuffer),
        0,
        buffers.pathDataBufferSizeInBytes * (1)
    ));
}

static void resetNNBuffers(
    BufferManager &buffers,
    int width,
    int height,
    Params &params
) {
    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.inputDataBuffer),
        0,
        buffers.inputDataBufferSizeInBytes
    ));

    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.NNPathDataBuffer),
        0,
        buffers.NNPathDataBufferSizeInBytes
    ));

    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.packedNNPathDataBuffer),
        0,
        buffers.NNPathDataBufferSizeInBytes
    ));

    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.predBuffer),
        0,
        buffers.predBufferSizeInBytes
    ));

    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.shadowOcclusionFloatTypeBuffer),
        0,
        buffers.shadowOcclusionFloatTypeBufferSizeInBytes
    ));

    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.contributionBuffer),
        0,
        buffers.contributionBufferSizeInBytes
    ));

    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.directLightingBuffer + (width * height * 3)),
        0,
        buffers.colorBufferSizeInBytes * (params.shadowPathCount - 1)
    ));
}

static void resetFrameBuffers(
    BufferManager &buffers,
    int width,
    int height,
    Params &params
) {
    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.directLightingBuffer),
        0,
        buffers.colorBufferSizeInBytes * params.shadowPathCount
    ));

    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.envLightingBuffer),
        0,
        buffers.colorBufferSizeInBytes
    ));

    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.contributionBuffer),
        0,
        buffers.colorBufferSizeInBytes
    ));

    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.shadowOcclusionFloatTypeBuffer),
        0,
        buffers.shadowOcclusionFloatTypeBufferSizeInBytes
    ));
}

static void resetBounceBuffers(
    BufferManager &buffers,
    int width,
    int height,
    Params &params
) {
    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.pathDataBuffer),
        0,
        buffers.pathDataBufferSizeInBytes * (1 + params.shadowPathCount)
    ));
}

static void resetScanBuffers(
    BufferManager &buffers,
    int width,
    int height,
    Params &params
) {
    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.tempScanBuffer),
        0,
        buffers.scanBufferSizeInBytes
    ));

    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.scanResultBuffer),
        0,
        buffers.scanBufferSizeInBytes
    ));

    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.scanInBuffer),
        0,
        buffers.scanBufferSizeInBytes
    ));
}

static void resetScanNNBuffers(
    BufferManager &buffers,
    int width,
    int height,
    Params &params
) {
    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.tempScanNNBuffer),
        0,
        buffers.scanBufferNNSizeInBytes
    ));

    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.scanResultNNBuffer),
        0,
        buffers.scanBufferNNSizeInBytes
    ));

    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.scanInNNBuffer),
        0,
        buffers.scanBufferNNSizeInBytes
    ));
}

static void resetMpiBuffer(
    MpiBuffers &mpibuffer,
    int comm_size,
    int width,
    int height
) {
    memset(mpibuffer.tempImage.data(), 0, sizeof(float) * width * height * 3);
    memset(mpibuffer.labelMap.data(), 0, sizeof(char) * mpibuffer.labelMap.size());
    // memset(mpibuffer.depthRecordBcast.data(), 0, sizeof(RecordOnlyWithDepth) * mpibuffer.depthRecordBcast.size());
    // std::fill(mpibuffer.validPixel.begin(), mpibuffer.validPixel.end(), false);
}

static void freeBuffers(ASArena &arena)
{
    arena.popTemp(); // index
    arena.popTemp(); // ray
    arena.popTemp(); // depth
    arena.popTemp(); // xi
    arena.popTemp(); // cosThetaWi
    arena.popTemp(); // sampleRecordIn
    arena.popTemp(); // sampleRecordOut
    arena.popTemp(); // occlusion
    arena.popTemp(); // missDirection
    arena.popTemp(); // colorBuffer
    arena.popTemp(); // barycentric
    // arena.popTemp(); // id
    // arena.popTemp(); // shadowOcclusion
    // arena.popTemp(); // shadowWeight
    // arena.popTemp(); // handlebuffer
}

static void mallocBuffers(
    BufferManager &buffers,
    ASArena &arena,
    int width,
    int height,
    Params &params,
    int handleSize,
    int worldSize
) {

    buffers.rayBufferSizeInBytes = width * height * sizeof(Ray) * 1;
    buffers.indexBufferSizeInBytes = width * height * sizeof(int) * 1;

    buffers.accelerationStructuresSizeInBytes = sizeof(AccelerationStructure) * handleSize;
    buffers.pathDataBufferSizeInBytes = width * height * sizeof(WavefrontPathData);
    buffers.scanBufferSizeInBytes = width * height * sizeof(int) * 2;
    buffers.colorBufferSizeInBytes = width * height * sizeof(float) * 3;
    buffers.transferOffsetSizeInBytes = sizeof(int) * (worldSize + 1);

    // buffers.inputDataBufferSizeInBytes = sizeof(float) * width * height * 5 * params.maxCount * params.shadowPathCount;
    // buffers.NNPathDataBufferSizeInBytes = sizeof(NNPathData) * width * height * params.maxCount * params.shadowPathCount;
    // buffers.predBufferSizeInBytes = sizeof(float) * width * height * params.maxCount * params.shadowPathCount * 2 * 2;
    buffers.inputDataBufferSizeInBytes = 0;
    buffers.NNPathDataBufferSizeInBytes = 0;
    buffers.predBufferSizeInBytes = 0;

    buffers.scanBufferNNSizeInBytes = sizeof(int) * width * height * params.maxCount * params.shadowPathCount;
    buffers.sceneOffsetSizeInBytes = sizeof(int) * (params.sceneSize + 1); // TODO
    buffers.shadowOcclusionCharTypeBufferSizeInBytes = sizeof(char) * width * height * params.maxCount * params.shadowPathCount;
    buffers.shadowOcclusionFloatTypeBufferSizeInBytes = sizeof(float) * width * height * params.maxCount * params.shadowPathCount;
    buffers.contributionBufferSizeInBytes = width * height * sizeof(float) * 3 * params.shadowPathCount;

    // new: wavefront



    params.transferPathDataBuffer = reinterpret_cast<WavefrontPathData *>(
        arena.pushTemp(buffers.pathDataBufferSizeInBytes)
    );
    params.pathDataBuffer = reinterpret_cast<WavefrontPathData *>(
        arena.pushTemp(buffers.pathDataBufferSizeInBytes * (1 + params.shadowPathCount))
    );

    params.accelerationStructures = reinterpret_cast<AccelerationStructure *>(
        arena.pushTemp(buffers.accelerationStructuresSizeInBytes)
    );

    // params.shadingPathDataBuffer = reinterpret_cast<WavefrontPathData *>(
    //     arena.pushTemp(buffers.pathDataBufferSizeInBytes)
    // );

    // params.colorBuffer = reinterpret_cast<float *>(
    //     arena.pushTemp(buffers.colorBufferSizeInBytes)
    // );

    params.scanResultBuffer = reinterpret_cast<int *>(
        arena.pushTemp(buffers.scanBufferSizeInBytes)
    );

    params.scanInBuffer = reinterpret_cast<int *>(
        arena.pushTemp(buffers.scanBufferSizeInBytes)
    );

    params.tempScanBuffer = reinterpret_cast<int *>(
        arena.pushTemp(buffers.scanBufferSizeInBytes)
    );

    params.transferCount = reinterpret_cast<int *>(
        arena.pushTemp(worldSize * sizeof(int))
    );

    params.transferOffset = reinterpret_cast<int *>(
        arena.pushTemp((worldSize + 1) * sizeof(int))
    );

    params.directLightingBuffer = reinterpret_cast<float *>(
        arena.pushTemp(buffers.colorBufferSizeInBytes * params.shadowPathCount)
    );

    params.envLightingBuffer = reinterpret_cast<float *>(
        arena.pushTemp(buffers.colorBufferSizeInBytes)
    );

    params.debugVisBuffer = reinterpret_cast<float *>( // TODO
        arena.pushTemp(buffers.colorBufferSizeInBytes)
    );

    params.scanResultNNBuffer = reinterpret_cast<int *>(
        arena.pushTemp(buffers.scanBufferNNSizeInBytes)
    );

    params.scanInNNBuffer = reinterpret_cast<int *>(
        arena.pushTemp(buffers.scanBufferNNSizeInBytes)
    );

    params.tempScanNNBuffer = reinterpret_cast<int *>(
        arena.pushTemp(buffers.scanBufferNNSizeInBytes)
    );

    params.sceneOffset = reinterpret_cast<int *>(
        arena.pushTemp(buffers.sceneOffsetSizeInBytes)      //TODO
    );

    // params.shadowOcclusionCharTypeBuffer = reinterpret_cast<char *>(
    //     arena.pushTemp(buffers.shadowOcclusionCharTypeBufferSizeInBytes)
    // );

    params.shadowOcclusionFloatTypeBuffer = reinterpret_cast<float *>(
        arena.pushTemp(buffers.shadowOcclusionFloatTypeBufferSizeInBytes)
    );

    params.contributionBuffer = reinterpret_cast<float *>(
        arena.pushTemp(buffers.contributionBufferSizeInBytes)
    );

    int numElem = width * height * params.maxCount * params.shadowPathCount;                // input number of elements
    int naux = 1024 << 1;
    int blks1 = int( numElem / naux ) + 1;
    int blks2 = int( blks1 / naux ) + 1;

    params.dev_auxArray1 = reinterpret_cast<int *>(
        arena.pushTemp(sizeof(int) * blks1)
    );

    params.dev_auxScan1 = reinterpret_cast<int *>(
        arena.pushTemp(sizeof(int) * blks1)
    );

    params.dev_auxArray2 = reinterpret_cast<int *>(
        arena.pushTemp(sizeof(int) * blks2)
    );

    params.dev_auxScan2 = reinterpret_cast<int *>(
        arena.pushTemp(sizeof(int) * blks2)
    );

    arena.printMemoryStatus();
}

static void deferredMallocBuffers(
    int maxPathSize,
    BufferManager &buffers,
    ASArena &arena,
    int width,
    int height,
    Params &params
) {
    arena.printMemoryStatus();

    int pixelSize = width * height;

    buffers.inputDataBufferSizeInBytes = sizeof(float) * pixelSize * 5 * params.maxCount * params.shadowPathCount;
    buffers.NNPathDataBufferSizeInBytes = sizeof(NNPathData) * pixelSize * params.maxCount * params.shadowPathCount;
    buffers.predBufferSizeInBytes = sizeof(float) * pixelSize * params.maxCount * params.shadowPathCount * 2 * 2;

    params.inputDataBuffer = reinterpret_cast<float *>(
        arena.pushTemp(buffers.inputDataBufferSizeInBytes)
    );

    params.packedInputDataBuffer = reinterpret_cast<float *>(
        arena.pushTemp(buffers.inputDataBufferSizeInBytes)
    );//TAG

    params.NNPathDataBuffer = reinterpret_cast<NNPathData *>(
        arena.pushTemp(buffers.NNPathDataBufferSizeInBytes)
    );//TAG

    params.packedNNPathDataBuffer = reinterpret_cast<NNPathData *>(
        arena.pushTemp(buffers.NNPathDataBufferSizeInBytes)
    );//TAG

    params.predBuffer = reinterpret_cast<float *>(
        arena.pushTemp(buffers.predBufferSizeInBytes)
    );//TAG


    arena.printMemoryStatus();
}

#if SEPARATEDNN
static void castShadowRaysDepthNN(
    BufferManager &buffers,
    std::map<PipelineType, OptixState> &optixStates,
    CUstream stream,
    int width,
    int height,
    Params &params,
    Timing &timing,
    std::vector<torch::jit::script::Module> modules,
    std::vector<torch::jit::script::Module> depth_modules,
    torch::Device device
) {
    std::vector<int> sceneOffsetCPU(params.sceneSize + 1);
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(sceneOffsetCPU.data()),
        params.sceneOffset,
        buffers.sceneOffsetSizeInBytes,
        cudaMemcpyDeviceToHost
    ));
    cudaMemcpy(sceneOffsetCPU.data(), params.sceneOffset, (params.sceneSize + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    torch::NoGradGuard no_grad;

    for(int i = 0; i < depth_model_name.size(); i++) {
        if(depth_model_name[i].compare("padding") == 0) continue;
        int hitAABBID = i + 1;
        if(sceneOffsetCPU[hitAABBID] == sceneOffsetCPU[hitAABBID - 1]) continue;

        std::vector<torch::jit::IValue> inputs;

        timing.start(TimedSection::NNTime);

        //Data Processing
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        const int startIndex = sceneOffsetCPU[hitAABBID - 1] * 5;
        const int dataSize = sceneOffsetCPU[hitAABBID] - sceneOffsetCPU[hitAABBID - 1];
        torch::Tensor dataGPU = torch::from_blob(params.packedInputDataBuffer + startIndex, {dataSize ,5}, options).clone();

        inputs.push_back(dataGPU);

        //Forward
        at::Tensor output = depth_modules[i].forward(inputs).toTensor();

        // auto outputCPU = output.to(at::kCPU);
        // std::cout << outputCPU << std::endl;

        cudaMemcpy(
            reinterpret_cast<void *>(params.predBuffer + sceneOffsetCPU[hitAABBID - 1]),
            output.data_ptr<float>(),
            output.numel() * sizeof(float),
            cudaMemcpyDeviceToDevice
        );

        torch::cuda::synchronize();
        timing.end(TimedSection::NNTime);
        // std::cout << "NNTime: " << timing.getMilliseconds(TimedSection::NNTime) << std::endl;
    }

    Depth_Buffer_Update(params, sceneOffsetCPU.back() + 1, width, height);
}
#endif

static void castShadowRaysNN(
    BufferManager &buffers,
    std::map<PipelineType, OptixState> &optixStates,
    CUstream stream,
    int width,
    int height,
    Params &params,
    Timing &timing,
    std::vector<torch::jit::script::Module> modules,
    std::vector<torch::jit::script::Module> depth_modules,
    torch::Device device
) {
    std::vector<int> sceneOffsetCPU(params.sceneSize + 1);
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(sceneOffsetCPU.data()),
        params.sceneOffset,
        buffers.sceneOffsetSizeInBytes,
        cudaMemcpyDeviceToHost
    ));
    cudaMemcpy(sceneOffsetCPU.data(), params.sceneOffset, (params.sceneSize + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    torch::NoGradGuard no_grad;

    #if SEPARATEDNN
    for(int i = 0; i < vis_model_name.size(); i++) {
        if(vis_model_name[i].compare("padding") == 0) continue;
        int hitAABBID = i + 1;
        if(sceneOffsetCPU[hitAABBID] == sceneOffsetCPU[hitAABBID - 1]) continue;

        std::vector<torch::jit::IValue> inputs;

        timing.start(TimedSection::NNTime);

        //Data Processing
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        const int startIndex = sceneOffsetCPU[hitAABBID - 1] * 5;
        const int dataSize = sceneOffsetCPU[hitAABBID] - sceneOffsetCPU[hitAABBID - 1];
        torch::Tensor dataGPU = torch::from_blob(params.packedInputDataBuffer + startIndex, {dataSize ,5}, options).clone();

        inputs.push_back(dataGPU);

        //Forward
        at::Tensor output = modules[i].forward(inputs).toTensor();

        // auto outputCPU = output.to(at::kCPU);
        // std::cout << outputCPU << std::endl;

        cudaMemcpy(
            reinterpret_cast<void *>(params.predBuffer + sceneOffsetCPU[hitAABBID - 1]),
            output.data_ptr<float>(),
            output.numel() * sizeof(float),
            cudaMemcpyDeviceToDevice
        );

        torch::cuda::synchronize();
        timing.end(TimedSection::NNTime);
        std::cout << "NNTime: " << timing.getMilliseconds(TimedSection::NNTime) << std::endl;
    }
    #else
    for(int i = 0; i < model_name.size(); i++) {
        if(model_name[i].compare("padding") == 0) continue;
        int hitAABBID = i + 1;
        if(sceneOffsetCPU[hitAABBID] == sceneOffsetCPU[hitAABBID - 1]) continue;

        std::vector<torch::jit::IValue> inputs;

        timing.start(TimedSection::NNTime);

        //Data Processing
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        const int startIndex = sceneOffsetCPU[hitAABBID - 1] * 5;
        const int dataSize = sceneOffsetCPU[hitAABBID] - sceneOffsetCPU[hitAABBID - 1];
        torch::Tensor dataGPU = torch::from_blob(params.packedInputDataBuffer + startIndex, {dataSize ,5}, options).clone();

        inputs.push_back(dataGPU);

        //Forward
        at::Tensor output = modules[i].forward(inputs).toTensor();

        // auto outputCPU = output.to(at::kCPU);
        // std::cout << outputCPU << std::endl;

        cudaMemcpy(
            reinterpret_cast<void *>(params.predBuffer + 2 * sceneOffsetCPU[hitAABBID - 1]),
            output.data_ptr<float>(),
            output.numel() * sizeof(float),
            cudaMemcpyDeviceToDevice
        );

        torch::cuda::synchronize();
        timing.end(TimedSection::NNTime);
        // std::cout << "NNTime: " << timing.getMilliseconds(TimedSection::NNTime) << std::endl;
    }
    #endif

    Frame_Buffer_Update(params, sceneOffsetCPU.back() + 1, width, height);
}

//===========================================================
//ray tracing
//===========================================================
static void runSample(
    int sample,
    int bounces,
    std::map<PipelineType, OptixState> &optixStates,
    SceneState &sceneState,
    BufferManager &buffers,
    std::vector<PtexTexture> &textures,
    CUstream stream,
    const int width,
    const int height,
    int spp,
    Params &params,
    CUdeviceptr d_params,
    std::vector<float> &outputImage,
    std::vector<float> &textureImage,
    Timing &timing,
    int &myid,
    int &comm_size,
    const std::string &exrFilename,
    int &frame,
    MpiBuffers &mpiBuffer,
    MPI_Comm &comm_world,
    std::vector<std::vector<AABB>> &aabbs,
    std::vector<torch::jit::script::Module> &modules,
    std::vector<torch::jit::script::Module> &depth_modules,
    torch::Device &device
) {
    //std::ofstream time_static("time_static.txt", std::ios::out);
    MPI_Barrier(comm_world); //同步
    timing.start(TimedSection::Sample);
    std::cout << "Sample #" << sample << std::endl;

    const int FLAG_COUT = myid;
    const int ROOT = 0; //MPI_ROOT
    int optix_width = width;
    int optix_height = height;
    int count = 0;
    int offset = 0;

    std::vector<float> visualizeValidPixel(width * height * 3, 0.f);

    params.worldID = myid;
    params.bounce = 0;
    params.sampleCount = sample;

    resetSampleBuffers(buffers, width, height, params);//buffer置为0，depth设为MAX
    // resetBounceBuffers(buffers, width, height, params);//buffer置为0，depth设为MAX
    CHECK_CUDA(cudaDeviceSynchronize());
    //Path Gen (masterNode)
    params.pathSize = width * height;
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_params),
        &params,
        sizeof(Params),
        cudaMemcpyHostToDevice
    ));

    if(myid == 0) {
        CHECK_OPTIX(optixLaunch(
            optixStates[PipelineType::PathGen].pipeline,
            stream,
            d_params,
            sizeof(Params),
            &optixStates[PipelineType::PathGen].sbt,
            optix_width,
            optix_height,
            /*depth=*/1
        ));

        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Path Trace
    for (int bounce = 0; bounce <= bounces ; bounce++) {

        resetNNBuffers(buffers, width, height, params);
        params.bounce = bounce;

        CHECK_CUDA(cudaMemcpy(
            reinterpret_cast<void *>(d_params),
            &params,
            sizeof(Params),
            cudaMemcpyHostToDevice
        ));

        int count = 0;
        while(true) {

            CHECK_OPTIX(optixLaunch(
                optixStates[PipelineType::TraRay].pipeline,
                stream,
                d_params,
                sizeof(Params),
                &optixStates[PipelineType::TraRay].sbt,
                optix_width,
                optix_height,
                /*depth=*/1
            ));

            CHECK_CUDA(cudaDeviceSynchronize());

            timing.start(TimedSection::Experiment_0);

            resetScanBuffers(buffers, width, height, params);

            Work_Efficient_Scan(params, width, height, comm_size);

            timing.end(TimedSection::Experiment_0);
            // std::cout << myid << "Work_Efficient_Scan: " << timing.getMilliseconds(TimedSection::Experiment_0) << std::endl;

            copyOutputBuffers(buffers, width, height, params, comm_size);

            std::vector<int> sendCount(comm_size);
            std::vector<int> recvCount(comm_size);
            std::vector<int> recvOffset(comm_size + 1, 0);


            MPI_Barrier(MPI_COMM_WORLD);
            timing.start(TimedSection::Experiment_1);


            for (int i = 0; i < comm_size; i++) {
                // std::cout << myid << "transferOffset[i]: " << buffers.output.transferOffset[i] << std::endl;
                // std::cout << myid << "transferOffset[i + 1]: " << buffers.output.transferOffset[i + 1] << std::endl;
                buffers.output.transferOffset[i + 1] *= sizeof(WavefrontPathData);
                sendCount[i] = buffers.output.transferOffset[i + 1] - buffers.output.transferOffset[i];
                if(i != myid) std::cout << myid << " To " << i << " sendCount: " << sendCount[i] / sizeof(WavefrontPathData) << std::endl;
            }

            MPI_Alltoall(sendCount.data(), 1, MPI_INT, recvCount.data(), 1, MPI_INT, MPI_COMM_WORLD);

            for (int i = 0; i < comm_size; i++) {
                recvOffset[i + 1] = recvOffset[i] + recvCount[i];
                // std::cout << myid << "recvCount: " << recvCount[i] / sizeof(WavefrontPathData) << std::endl;
            }

            resetBounceBuffers(buffers, width, height, params);//pathBuffer置为0
            MPI_Alltoallv(buffers.output.transferPathDataBuffer.data(), sendCount.data(), buffers.output.transferOffset.data(), MPI_BYTE, buffers.output.pathDataBuffer.data(), recvCount.data(), recvOffset.data(), MPI_BYTE, MPI_COMM_WORLD);

            timing.end(TimedSection::Experiment_1);
            std::cout << myid << "Transfer: " << timing.getMilliseconds(TimedSection::Experiment_1) << std::endl;

            CHECK_CUDA(cudaMemcpy(
                reinterpret_cast<void *>(params.pathDataBuffer),
                buffers.output.pathDataBuffer.data(),
                recvOffset.back(),
                cudaMemcpyHostToDevice
            ));

            if (buffers.output.transferOffset.back() == sendCount[myid] && recvOffset.back() == sendCount[myid]) {
                params.pathSize = recvOffset.back() / sizeof(WavefrontPathData);
                break;
            }
            params.pathSize = recvOffset.back() / sizeof(WavefrontPathData);
            // std::cout << "pathSize: " << params.pathSize << std::endl;
            CHECK_CUDA(cudaMemcpy(
                reinterpret_cast<void *>(d_params),
                &params,
                sizeof(Params),
                cudaMemcpyHostToDevice
            ));

            optix_height = (params.pathSize + optix_width - 1) / optix_width;
            count++;
        }

        std::cout << "shading pathSize: " << params.pathSize << std::endl;

        if(frame == 0 && sample == 0 && bounce == 0) deferredMallocBuffers(params.pathSize, buffers, sceneState.arena, width, height, params);

        params.shadowPathSize = params.shadowPathCount * params.pathSize;
        CHECK_CUDA(cudaMemcpy(
            reinterpret_cast<void *>(d_params),
            &params,
            sizeof(Params),
            cudaMemcpyHostToDevice
        ));

        CHECK_OPTIX(optixLaunch(
            optixStates[PipelineType::MainRay].pipeline,
            stream,
            d_params,
            sizeof(Params),
            &optixStates[PipelineType::MainRay].sbt,
            optix_width,
            optix_height,
            /*depth=*/1
        ));

        timing.start(TimedSection::MPI_Function_1);

        CHECK_OPTIX(optixLaunch(
            optixStates[PipelineType::ShadowRay].pipeline,
            stream,
            d_params,
            sizeof(Params),
            &optixStates[PipelineType::ShadowRay].sbt,
            optix_width,
            optix_height * params.shadowPathCount,
            /*depth=*/1
        ));


        CHECK_CUDA(cudaDeviceSynchronize());

        #if SEPARATEDNN
        timing.start(TimedSection::Experiment_2);
        resetScanNNBuffers(buffers, width, height, params);

        Work_Efficient_Scan_For_NN_HIT_INSIDE(params, width, height);

        castShadowRaysDepthNN(buffers, optixStates, stream, width, height, params, timing, modules, depth_modules, device);
        timing.end(TimedSection::Experiment_2);

        std::cout << "Depth TIME: " << timing.getMilliseconds(TimedSection::Experiment_2) << std::endl;
        #endif

        resetScanNNBuffers(buffers, width, height, params);

        Work_Efficient_Scan_For_NN(params, width, height);

        CHECK_CUDA(cudaDeviceSynchronize());



        castShadowRaysNN(buffers, optixStates, stream, width, height, params, timing, modules, depth_modules, device);

        timing.end(TimedSection::MPI_Function_1);

        std::cout << "NN VIS TIME: " << timing.getMilliseconds(TimedSection::MPI_Function_1) << std::endl;

        optix_height = (params.pathSize + optix_width) / optix_width;
        assert(params.pathSize <= width * height * 2);
        CHECK_CUDA(cudaMemcpy(
            reinterpret_cast<void *>(d_params),
            &params,
            sizeof(Params),
            cudaMemcpyHostToDevice
        ));

    }

    timing.end(TimedSection::Sample);
    //time_static.close();
}



void launch(
    RenderRequest renderRequest,
    std::map<PipelineType, OptixState> &optixStates,
    SceneState &sceneState,
    Cam cam,
    const std::string &exrFilename,
    int &myid,
    int &comm_size,
    int &frame,
    std::vector<float> &Image,
    std::vector<float> &normalImage,
    std::vector<float> &albedoImage,
    MPI_Comm &comm_world
) {
    Params params;
    BufferManager buffers;
    MpiBuffers mpiBuffer;

    CUdeviceptr d_params;
    CUstream stream;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(Params)));
    CHECK_CUDA(cudaStreamCreate(&stream));

    const int width = renderRequest.width;
    const int height = renderRequest.height;
    int geometriesAmount = 0 , trueGeometriesAmount = 0;
    params.maxCount = 3;
    params.shadowPathCount = 4;
    params.frameBufferSize = width * height;
    params.sceneSize = sceneState.geometries.size();

    std::vector<float> outputImage(width * height * 3, 0.f);
    std::vector<PtexTexture> textures;
    std::vector<OptixTraversableHandle> handles; // Handle aabbHandle originHandle
    std::vector<AccelerationStructure> accelerationStructures(sceneState.geometries.size());
    // std::vector<aabbRecord> AABBsInfo;

    #if MOANA
    for (const auto &filename : Textures::textureFilenames) {
        PtexTexture texture(MOANA_ROOT + std::string("/island/") + filename);
        textures.push_back(texture);
    }
    #else
    // sceneState.arena.printMemoryStatus();
    std::vector<cudaTextureObject_t> texObjs;
    stbi_set_flip_vertically_on_load(1);
    for (const auto &filename : Textures::textureFilenames) {
        int width, height, component;
        #if TEX_MOANA
        std::string load_filepath = MOANA_ROOT + std::string("/island/") + filename;
        #else
        std::string load_filepath = filename;
        #endif
        auto picture = stbi_loadf(load_filepath.c_str(), &width, &height, &component, 4);

        if (!picture) {
            // throw std::runtime_error("can not Load " + filename);
            std::cout << "can not Load " + filename << std::endl;
            texObjs.push_back(0);
            continue;
        }
        // std::cout << "Load " + filename << std::endl;
        component = 4;
        CUdeviceptr d_texture;
        const size_t bufferSizeInBytes = sizeof(float) * component * width * height;

        size_t pitch;
        CHECK_CUDA(cudaMallocPitch(
            reinterpret_cast<void **>(&d_texture),
            &pitch,//间距
            sizeof(float) * component * width,
            height
        ));

        CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_texture)));

        const size_t hostBufferSizeInBytes = sizeof(float) * component * width * height;
        const size_t deviceBufferSizeInBytes = pitch * height;

        d_texture = sceneState.arena.pushTemp(deviceBufferSizeInBytes);

        CHECK_CUDA(cudaMemcpy2D(
            reinterpret_cast<void *>(d_texture),
            pitch,
            picture,
            sizeof(float) * component * width,
            sizeof(float) * component * width,
            height,
            cudaMemcpyHostToDevice
        ));

        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr = reinterpret_cast<void *>(d_texture);
        resDesc.res.pitch2D.desc.x = 32;
        resDesc.res.pitch2D.desc.y = 32;
        resDesc.res.pitch2D.desc.z = 32;
        resDesc.res.pitch2D.desc.w = 32;
        resDesc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
        resDesc.res.pitch2D.width = width;
        resDesc.res.pitch2D.height = height;
        resDesc.res.pitch2D.pitchInBytes = pitch;

        cudaTextureDesc texDesc;//定义纹理
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;//越界情况下的访存模式
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;//滤波形式
        texDesc.readMode = cudaReadModeElementType;//读取模式-归一化float模式和原始数据模式
        texDesc.normalizedCoords = 1;//指定是否归一化坐标

        cudaTextureObject_t texObj = 0;
        CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

        texObjs.push_back(texObj);
        stbi_image_free(picture);
    }

    size_t texObjsSizeInBytes = sizeof(cudaTextureObject_t) * texObjs.size();
    cudaTextureObject_t* d_texObjs = reinterpret_cast<cudaTextureObject_t *>(sceneState.arena.pushTemp(texObjsSizeInBytes));
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_texObjs),
        texObjs.data(),
        texObjsSizeInBytes,
        cudaMemcpyHostToDevice
    ));
    params.albedoTextures = d_texObjs;
    // sceneState.arena.printMemoryStatus();

    #endif

    std::vector<Triangle> lights;
    #if BISTRO
    const Vec3 L = 10.f * Vec3(505.928150, 505.928150, 505.928150);
    const Vec3 L1 = 0.001f * Vec3(505.928150, 505.928150, 505.928150);
    const Vec3 L2 = 30.f * Vec3(505.928150, 505.928150, 505.928150);
    const Vec3 L3 = 10.f * Vec3(505.928150, 505.928150, 505.928150);

    std::vector<Vec3> pointsList = {
        Vec3(0.12184541672468185, 8.149029731750488, 0.31903180480003357),
        Vec3(0.13482901453971863, 8.29036808013916, 0.45987018942832947),
        Vec3(0.2922881841659546, 8.166015625, 0.3377407491207123),
        Vec3(0.3052719831466675, 8.307354927062988, 0.4785784184932709),

        Vec3(-4.677332878112793, 9.999991416931152, -6.165306091308594),
        Vec3(3.3226675987243652, 9.999991416931152, -6.165306091308594),
        Vec3(-4.677332878112793, 9.999991416931152, 13.834686279296875),
        Vec3(3.3226675987243652, 9.999991416931152, 13.834686279296875),

        Vec3(6.292665004730225, 4.862171173095703, 1.3878426551818848),
        Vec3(6.260426998138428, 4.918705940246582, 1.4343327283859253),
        Vec3(6.340075969696045, 4.868965148925781, 1.4374041557312012),
        Vec3(6.307837963104248, 4.925500869750977, 1.4838939905166626),

        Vec3(-7.037281513214111, 4.862171173095703, 3.9932265281677246),
        Vec3(-6.986903667449951, 4.91870641708374, 4.018969535827637),
        Vec3(-6.99449348449707, 4.8689656257629395, 3.9396233558654785),
        Vec3(-6.944116115570068, 4.925501346588135, 3.9653663635253906)
    };
    lights.push_back(Triangle(pointsList[0], pointsList[2], pointsList[3], L));
    lights.push_back(Triangle(pointsList[3], pointsList[1], pointsList[0], L));
    lights.push_back(Triangle(pointsList[3 + 4], pointsList[2 + 4], pointsList[0 + 4], L1));
    lights.push_back(Triangle(pointsList[0 + 4], pointsList[1 + 4], pointsList[3 + 4], L1));
    lights.push_back(Triangle(pointsList[0+8], pointsList[2+8], pointsList[3+8], L2));
    lights.push_back(Triangle(pointsList[3+8], pointsList[1+8], pointsList[0+8], L2));
    // lights.push_back(Triangle(pointsList[0+12], pointsList[2+12], pointsList[3+12], L3));
    // lights.push_back(Triangle(pointsList[0+12], pointsList[1+12], pointsList[3+12], L3));
    #elif SAN_MIGUEL
    const Vec3 L = 1.0f * Vec3(891.443777, 505.928150, 154.625939);

    std::vector<Vec3> pointsList = {
        Vec3(14779.412109375, 153398.8125, -4307.61865234375),
        Vec3(-2001.59326171875, 146156.09375, -12428.0302734375),
        Vec3(4770.7685546875, 157817.109375, 12434.7119140625),
        Vec3(-2001.59326171875, 146156.09375, -12428.0302734375)
    };
    lights.push_back(Triangle(pointsList[0], pointsList[2], pointsList[3], L));
    lights.push_back(Triangle(pointsList[3], pointsList[1], pointsList[0], L));
    #elif AIR_DROME
    const Vec3 L =Vec3(891.443777, 505.928150, 154.625939);

    std::vector<Vec3> pointsList = {
        Vec3(-2114.32861328125, 81.61964416503906, 140.63539123535156),
        Vec3(-2114.179443359375, 81.80250549316406, -59.34486389160156),
        Vec3(-1942.9859619140625, 442.6954345703125, 140.6354217529297),
        Vec3(-1942.8370361328125, 442.87811279296875, -59.34484100341797)
    };
    lights.push_back(Triangle(pointsList[0], pointsList[3], pointsList[2], L));
    lights.push_back(Triangle(pointsList[0], pointsList[1], pointsList[3], L));
    #else
    const Vec3 L = Vec3(891.443777, 505.928150, 154.625939);

    std::vector<Vec3> pointsList = {
        Vec3(101346.539, 202660.438, 189948.188),
        Vec3(106779.617, 187339.562, 201599.453),
        Vec3(83220.3828, 202660.438, 198400.547),
        Vec3(101346.539, 202660.438, 189948.188),
        Vec3(88653.4609, 187339.562, 210051.812),
        Vec3(83220.3828, 202660.438, 198400.547)
    };
    lights.push_back(Triangle(pointsList[0], pointsList[1], pointsList[2], L));
    lights.push_back(Triangle(pointsList[3], pointsList[4], pointsList[5], L));
    #endif

    size_t lightsSizeInBytes = sizeof(Triangle) * lights.size();
    Triangle* d_lights = reinterpret_cast<Triangle *>(sceneState.arena.pushTemp(lightsSizeInBytes));

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_lights),
        lights.data(),
        lightsSizeInBytes,
        cudaMemcpyHostToDevice
    ));
    params.directLights = d_lights;
    params.lightCount = lights.size();

    mallocBuffers(buffers, sceneState.arena, width, height, params, sceneState.geometries.size(), comm_size);

    for (const auto &[j, geometry] : enumerate(sceneState.geometries)) {

        AccelerationStructure tempAS;
        aabbRecord aabb;

        if(geometry.isProxy || GENERATE_DATA) {
            std::cout << geometry.aabbRecords.back() << std::endl;
            aabb.angle =  geometry.aabbRecords.back().angle;
            aabb.width =  geometry.aabbRecords.back().width;
            aabb.height = geometry.aabbRecords.back().height;
            aabb.m_maxX = geometry.aabbRecords.back().m_maxX;
            aabb.m_maxY = geometry.aabbRecords.back().m_maxY;
            aabb.m_maxZ = geometry.aabbRecords.back().m_maxZ;
            aabb.m_minX = geometry.aabbRecords.back().m_minX;
            aabb.m_minY = geometry.aabbRecords.back().m_minY;
            aabb.m_minZ = geometry.aabbRecords.back().m_minZ;
            aabb.m_max = Vec3(aabb.m_maxX, aabb.m_maxY, aabb.m_maxZ);
            aabb.m_min = Vec3(aabb.m_minX, aabb.m_minY, aabb.m_minZ);
            aabb.m_maxLength = (aabb.m_max - aabb.m_min).length();

            std::cout << " " << aabb.m_maxLength << std::endl;
        }
        tempAS.handle = geometry.handle;
        tempAS.originHandle = geometry.originHandle;
        tempAS.aabbHandle = geometry.aabbHandle;
        tempAS.isProxy = geometry.isProxy;
        tempAS.nodeID = geometry.nodeID;
        tempAS.AABBInfo = aabb;

        accelerationStructures[j] = tempAS;
    }
    std::cout << "accelerationStructuresSize = " << accelerationStructures.size() << std::endl;
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *> (params.accelerationStructures),
        accelerationStructures.data(),
        buffers.accelerationStructuresSizeInBytes,
        cudaMemcpyHostToDevice
    ));

    params.envLightTexture = sceneState.environmentState.textureObject;

    mallocMpiBuffer(mpiBuffer, comm_size, width, height);

    std::vector<float> textureImage(width * height * 3, 0.f);

    char node_name[MPI_MAX_PROCESSOR_NAME];
    int name_len, worldRank = 0, worldSize = 0, stride = 0;
    std::vector<std::vector<AABB>> aabbBuffer;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    // stride = worldSize / 2;

     #if SEPARATEDNN
    //======================================
    //Init
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        // std::cout << "CUDA available! Predicting on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else {
        std::cout << "Predicting on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    std::vector<torch::jit::script::Module> modules;
    for(int i = 0; i < vis_model_name.size(); i++) {
        if(vis_model_name[i].compare("padding") == 0) {
            modules.push_back(torch::jit::script::Module("padding"));
            continue;
        }
        std::string model_pb = "/home/moana/torchScript/" + vis_model_name[i];
        torch::jit::script::Module module = torch::jit::load(model_pb);
        module.to(device);
        modules.push_back(module);
    }
    std::vector<torch::jit::script::Module> depth_modules;
    for(int i = 0; i < depth_model_name.size(); i++) {
        if(depth_model_name[i].compare("padding") == 0) {
            depth_modules.push_back(torch::jit::script::Module("padding"));
            continue;
        }
        std::string model_pb = "/home/moana/torchScript/" + depth_model_name[i];
        torch::jit::script::Module module = torch::jit::load(model_pb);
        module.to(device);
        depth_modules.push_back(module);
    }
    //======================================
    #else
    //Init
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        // std::cout << "CUDA available! Predicting on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else {
        std::cout << "Predicting on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    std::vector<torch::jit::script::Module> modules;
    for(int i = 0; i < model_name.size(); i++) {
        if(model_name[i].compare("padding") == 0) {
            modules.push_back(torch::jit::script::Module("padding"));
            continue;
        }
        std::string model_pb = "/home/moana/torchScript/" + model_name[i];
        torch::jit::script::Module module = torch::jit::load(model_pb);
        module.to(device);
        modules.push_back(module);
    }
    std::vector<torch::jit::script::Module> depth_modules;
    //======================================
    #endif

    Scene scene(cam);
    Camera camera = scene.getCamera(width, height);

    for(int cframe = 0; cframe < 1; cframe++) {
    frame = cframe;
    #if BISTRO
    if(cframe < 48) {
        camera.m_origin += cframe * Vec3(0.001, 0.0, 0.0);
    }else{
        camera.m_origin = camera.m_origin - cframe * Vec3(0.001, 0.0, 0.0);
    }
    #elif CITY
    camera.m_origin += cframe * Vec3(0.001, 0.0005, 0.0);
    #elif AIR_DROME
    camera.m_origin += cframe * Vec3(-0.0001, 0.0, -0.0001);
    #else
    camera.m_origin += cframe * Vec3(0.001, 0.0, 0.0);
    #endif
    camera.updateTransformMatrix();

    params.camera = camera;
    resetFrameBuffers(buffers, width, height, params);//重置帧缓冲区
    const int worldSpp = renderRequest.spp;
    for (int sample = 0; sample < worldSpp; sample++) {
        Timing timing;//统计程序运行时长

        runSample(
            sample,
            renderRequest.bounces,
            optixStates,
            sceneState,
            buffers,
            textures,
            stream,
            width,
            height,
            worldSpp,
            params,
            d_params,
            outputImage,
            textureImage,
            timing,
            worldRank,
            worldSize,
            exrFilename,
            frame,
            mpiBuffer,
            comm_world,
            aabbBuffer,
            modules,
            depth_modules,
            device
        );

        std::cout << "   Sample timing:" << std::endl;
        std::cout << "    Total: " << timing.getMilliseconds(TimedSection::Sample) << std::endl;

    }

    copyOutputBuffersForColor(buffers, width, height, params);
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int pixelIndex = 3 * (row * width + col);
            mpiBuffer.tempImage[pixelIndex] =     (buffers.output.directLightingBuffer[pixelIndex + 0] + buffers.output.envLightingBuffer[pixelIndex + 0]) / float(worldSpp);
            mpiBuffer.tempImage[pixelIndex + 1] = (buffers.output.directLightingBuffer[pixelIndex + 1] + buffers.output.envLightingBuffer[pixelIndex + 1]) / float(worldSpp);
            mpiBuffer.tempImage[pixelIndex + 2] = (buffers.output.directLightingBuffer[pixelIndex + 2] + buffers.output.envLightingBuffer[pixelIndex + 2]) / float(worldSpp);
        }
    }

    MPI_Reduce(mpiBuffer.tempImage.data(), outputImage.data(), outputImage.size(), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (worldRank == 0) {
        Image::save(//To Do:Need Adjust Saving Path
            width,
            height,
            outputImage,
            std::to_string(cframe ) + exrFilename
        );}
    }

    CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_params)));
}

} }

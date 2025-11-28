#pragma once

#include <memory>
#include <vector>
#include <unordered_map>

#include "Buffer.hpp"
#include "Drawable.hpp"
#include "Mesh.hpp"
#include "BlendShapeData.hpp"
#include "AnimationPlayer.hpp"
#include "Graphics.hpp"
#include "Pipeline.hpp"
#include "Timer.hpp"
#include "GaussianAsset.hpp"
#include "NeuralCompensation.hpp"

namespace hrm
{
struct PrefixSumArgs
{
    bool inclusive { true };
    Buffer* srcBuffer { nullptr };
    Buffer* dstBuffer { nullptr };

    std::unique_ptr<Buffer> tempBuffer1;
    std::unique_ptr<Buffer> tempBuffer2;
};

struct SortArgs
{
    uint32_t count { 0u };
    uint32_t bitCount { 16u };
    Buffer* inputKeys { nullptr };
    Buffer* inputValues { nullptr };
    Buffer* dstKeys { nullptr };
    Buffer* dstValues { nullptr };

    // 8 uint for sort params, 3 * 4 uint for indirect draw args(SumPass, ReducePass, CopyPass), total 20 * sizeof(uint32_t)
    std::unique_ptr<Buffer> sortParamsBuffer;

    std::unique_ptr<Buffer> sortScratchBuffer;
    std::unique_ptr<Buffer> payloadScratchBuffer;
    std::unique_ptr<Buffer> scratchBuffer;
    std::unique_ptr<Buffer> reducedScratchBuffer;
};

class Renderer
{
public:
    Renderer(MTL::Device *device);
    ~Renderer() = default;
    
    void SetDrawable(void* drawableHandle);
    
    // update everything instead for draw
    void LogicalUpdate();
    void Draw();
    
    void SetCameraMatrix(uint32_t eyeIdx, const simd::float4x4& viewMat, const simd::float4x4& projMat);

    Mesh* GetMesh() const { return m_Mesh.get(); }

    void SetBlendShapeData(std::unique_ptr<BlendShapeData>&& blendShapeData);
    void SetBoneIndexMap(const std::unordered_map<std::string, uint32_t>& boneNameToIndex);
    void SetBonePositions(const std::vector<simd::float3>& bonePositions) { m_BonePositions = bonePositions; }
    void SetBoneQuaternions(const std::vector<simd::float4>& boneQuaternions) { m_BoneQuaternions = boneQuaternions; }

    simd::float4 GetBoneRotation(const std::string& boneName) { return m_BoneQuaternions[m_BoneNameToIndex[boneName]];};
    
    void SetBoneRotation(const std::string& boneName, const simd::float4& rotation) { m_BoneQuaternions[m_BoneNameToIndex[boneName]] = rotation; }
    void SetBonePosition(const std::string& boneName, const simd::float3& position) { m_BonePositions[m_BoneNameToIndex[boneName]] = position; }

    void SetBlendShapeWeight(uint32_t blendShapeIndex, float value) { m_BlendShapeWeights[blendShapeIndex] = value; }

    MTL::Device* GetDevice() const { return m_pDevice; }

    Graphics* GetGraphics() const { return m_Graphics.get(); }

    Buffer* GetPoseShadowCompensationBuffer() const { return m_PoseShadowCompensationBuffer; }

    std::unique_ptr<Buffer> CreateBuffer(uint32_t size, const std::string& name = "", MemoryType memoryType = MemoryType::Device);

private:

    void ApplyBlendShape();

    void CalculateBoneMatrices();

    void ApplyBlendShapeFrame(const BlendShapeFrame& frame, float weight, uint8_t* positions);

    void CreateAllPipelines();

    void EncodeSkinningPass(MTL::ComputeCommandEncoder* encoder);

    void EncodeBackfaceCullPass(MTL::ComputeCommandEncoder* encoder);

    void EncodeCalcProjDataPass(MTL::ComputeCommandEncoder* encoder);

    void EncodeGenVisiblePointsPass(MTL::ComputeCommandEncoder* encoder);

    void EncodeSortPass(MTL::ComputeCommandEncoder* encoder);

    void EncodePostprocessComputePass(MTL::ComputeCommandEncoder* encoder, uint32_t viewIdx);

    // for visionos
    void EncodePostprocessRenderPass(MTL::RenderCommandEncoder* encoder, uint32_t viewIdx);
    
    void EncodeMeshDepthRenderPass(MTL::RenderCommandEncoder* encoder, uint32_t viewIdx);

    void ComputePrefixSum(MTL::ComputeCommandEncoder* encoder);

    void LoadGaussianModel(const std::string& modelPath);

private:
    MTL::Device* m_pDevice { nullptr };
    std::unique_ptr<Drawable> m_Drawable { nullptr };
    std::unique_ptr<Graphics> m_Graphics { nullptr };
    
    simd::float4x4 m_ModelMatrix { matrix_identity_float4x4 };
    simd::float4x4 m_ViewMatrix[2] { matrix_identity_float4x4 };
    simd::float4x4 m_ProjMatrix[2] { matrix_identity_float4x4 };
    
    //------------------gaussian related assets------------------//
    std::unique_ptr<Mesh> m_Mesh;
    std::unique_ptr<AnimationPlayer> m_AnimationPlayer;
    // bones related data
    std::vector<simd::float4x4> m_BindPoses;
    std::unordered_map<std::string, uint32_t> m_BoneNameToIndex;
    std::vector<int> m_BoneParentIndices;
    std::vector<simd::float3> m_BonePositions;  // bone's local space positions
    std::vector<simd::float4> m_BoneQuaternions;  // bone's local space quaternions

    std::vector<simd::float4x4> m_BoneMatrices;    // cache it instead of create every frame
    std::vector<simd::float3> m_SkinnedPositions;  // cache it instead of create every frame
    std::vector<simd::float4> m_SkinnedQuaternions; // cache it instead of create every frame
    // blendshape related data
    std::unique_ptr<BlendShapeData> m_BlendShapeData;
    std::vector<float> m_BlendShapeWeights;
    Buffer* m_CachedBSVertexBuffer;
    std::unique_ptr<Buffer> m_SkinnedVertexBuffer;

    std::unique_ptr<GaussianAsset> m_GaussianAsset;
    uint32_t m_SHOrder { 3u };
    PrefixSumArgs m_PrefixSumArgs;
    SortArgs m_SortArgs;
    std::unique_ptr<Mesh> m_QuadMesh;
    //------------------runtime gpu buffers------------------//
    Buffer* m_PoseShadowCompensationBuffer;  // compensation data will be updated every frame, so it's should be allocated by buffer pool
    std::unique_ptr<Buffer> m_TriangleCullFlagBuffer;
    std::unique_ptr<Buffer> m_VisibleMasksBuffer;
    std::unique_ptr<Buffer> m_PointDistancesBuffer;
    std::unique_ptr<Buffer> m_SplatProjData0Buffer;
    std::unique_ptr<Buffer> m_SplatProjData1Buffer;
    std::unique_ptr<Buffer> m_VisibleMaskPrefixSumsBuffer;
    std::unique_ptr<Buffer> m_SplatSortKeysBuffer;
    std::unique_ptr<Buffer> m_IndirectDrawArgsBuffer;
    std::unique_ptr<Buffer> m_GPUSortDistancesBuffer;

    //------------------GPU compute pipeline states------------------//
    std::unique_ptr<ComputePipeline> m_SkinningPipeline;
    std::unique_ptr<ComputePipeline> m_BackfaceCullPipeline;
    std::unique_ptr<ComputePipeline> m_CalcProjDataPipeline;
    // prefix sum related pipelines
    std::unique_ptr<ComputePipeline> m_PrefixSumReducePipeline;
    std::unique_ptr<ComputePipeline> m_PrefixSumScanPipeline;
    std::unique_ptr<ComputePipeline> m_PrefixSumScanAddPipeline;
    std::unique_ptr<ComputePipeline> m_PrefixSumScanAddInclusivePipeline;
    // indirect draw related pipelines
    std::unique_ptr<ComputePipeline> m_GenVisiblePointsPipeline;
    std::unique_ptr<ComputePipeline> m_GenIndirectDrawArgsPipeline;
    // sort related pipelines
    std::unique_ptr<ComputePipeline> m_CalcDistancesPipeline;
    std::unique_ptr<ComputePipeline> m_SortCountPipeline;
    std::unique_ptr<ComputePipeline> m_SortReducePipeline;
    std::unique_ptr<ComputePipeline> m_SortScanPipeline;
    std::unique_ptr<ComputePipeline> m_SortScanAddPipeline;
    std::unique_ptr<ComputePipeline> m_SortScatterPipeline;
    // post process pipeline
    std::unique_ptr<RenderPipeline> m_PostprocessRenderPipeline; // for visionos
    std::unique_ptr<ComputePipeline> m_PostprocessComputePipeline; // for ios and macos
    std::unique_ptr<RenderPipeline> m_MeshDepthRenderPipeline; //for visionos
    
    //------------------GPU render pipeline state------------------//
    std::unique_ptr<RenderPipeline> m_GaussianRenderPipeline;

    std::unique_ptr<Timer> m_Timer;
    
    std::unique_ptr<NeuralCompensation> m_NeuralCompensation;
};

}  // namespace hrm

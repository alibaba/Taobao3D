#include <iostream>
#include <fstream>
#include <memory>

#include "json.hpp"

#include "Renderer.hpp"
#include "BlendShapeData.hpp"
#include "Buffer.hpp"
#include "Mesh.hpp"
#include "MetalHelper.hpp"
#include "AssetLoader.hpp"
#include "Logging.hpp"
#include "ShaderTypes.hpp"

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>

#define SORT_BIT_COUNT 16  // only support 16 bits for now
#define SORT_SUM_PASS_INDIRECT_BUFFER_OFFSET (8 * sizeof(uint32_t))
#define SORT_REDUCE_PASS_INDIRECT_BUFFER_OFFSET (12 * sizeof(uint32_t))
#define SORT_COPY_PASS_INDIRECT_BUFFER_OFFSET (16 * sizeof(uint32_t))

namespace
{

simd::float4x4 MakeProjectionMatrix(float n, float f, float fov, float aspect, bool flipY = false) {
    float fovRad = fov * M_PI / 180.0f;
    float t = n * std::tan(fovRad * .5f);
    float b = -t;
    float l = -t * aspect;
    float r = t * aspect;
    
    simd::float4x4 m = {
        simd_make_float4((2 * n) / (r - l), 0.0f, 0.0f, 0.0f),
        simd_make_float4(0.0f, -(2 * n) / (t - b), 0.0f, 0.0f),
        simd_make_float4(-(r + l) / (r - l), (t + b) / (t - b), (f + n) / (f - n), 1.0f),
        simd_make_float4(0.0f, 0.0f, (-2 * f * n) / (f - n), 0.0f)
    };
    if (flipY)
    {
        m.columns[0].y = -m.columns[0].y;
        m.columns[1].y = -m.columns[1].y;
        m.columns[2].y = -m.columns[2].y;
        m.columns[3].y = -m.columns[3].y;
    }
    return m;
}

simd::float4x4 MakeTransformMatrix(const simd::float3& position, const simd::float4& q)
{
    simd::float4x4 m(simd::quatf{q.x, q.y, q.z, q.w});
    m.columns[3] = simd::make_float4(position, 1.0f);
    return m;
}

simd::float4 MatrixToQuaternion(const simd::float4x4& M)
{
    simd::quatf q = simd::quatf(M);
    return q.vector;
}

simd::float4x4 RightToLeftHanded(const simd::float4x4& m)
{
    return simd::float4x4{
        simd_make_float4(m.columns[0].x, m.columns[0].y, -m.columns[0].z, m.columns[0].w),
        simd_make_float4(m.columns[1].x, m.columns[1].y, -m.columns[1].z, m.columns[1].w),
        simd_make_float4(-m.columns[2].x, -m.columns[2].y, m.columns[2].z, -m.columns[2].w),
        simd_make_float4(m.columns[3].x, m.columns[3].y, -m.columns[3].z, m.columns[3].w)
    };
}

simd::float4x4 AdjustProjectionMatrix(const simd::float4x4& projMat)
{
    simd::float4x4 mat = projMat;
    // zCompress: map ndc z [-1, 1] to [0, 1]
    mat.columns[0].z = (mat.columns[0].z + mat.columns[0].w) * 0.5f;
    mat.columns[1].z = (mat.columns[1].z + mat.columns[1].w) * 0.5f;
    mat.columns[2].z = (mat.columns[2].z + mat.columns[2].w) * 0.5f;
    mat.columns[3].z = (mat.columns[3].z + mat.columns[3].w) * 0.5f;

    // flipY
    mat.columns[0].y = -mat.columns[0].y;
    mat.columns[1].y = -mat.columns[1].y;
    mat.columns[2].y = -mat.columns[2].y;
    mat.columns[3].y = -mat.columns[3].y;
    return mat;
}

}

namespace hrm
{

// defined in AssetLoader.cpp
extern const std::string g_AssetBaseDir;

Renderer::Renderer(MTL::Device *device)
    : m_pDevice(device)
{
    m_Graphics = std::make_unique<Graphics>(this);

    m_AnimationPlayer = std::make_unique<AnimationPlayer>(this);

    LoadGaussianModel("hrm2-model-test");
    
    Buffer* vertexBuffer = m_Mesh->GetVertexBuffer();
    m_SkinnedVertexBuffer = CreateBuffer(vertexBuffer->GetSize(), "Skinned Vertex Buffer");
    m_SkinnedVertexBuffer->SetData(vertexBuffer->GetData(), vertexBuffer->GetSize());
    
    CreateAllPipelines();

    m_Timer = std::make_unique<Timer>();
    m_AnimationPlayer->Play();
    
    m_ModelMatrix.columns[0].x = -1.0f;
    m_ModelMatrix.columns[2].z = -1.0f;
}

void Renderer::SetDrawable(void* drawableHandle)
{
#ifdef PLATFORM_VISIONOS
    m_Drawable = std::make_unique<MetalVisionDrawable>(m_pDevice, (uint64_t)drawableHandle);
#else
    m_Drawable = std::make_unique<MetalDrawable>(m_pDevice, MetalHelper::GetMetalDrawable(drawableHandle));
#endif
}

void Renderer::Draw()
{
    if (!m_pDevice || !m_Drawable)
        return;
    
    // draw for multi view
    for (uint32_t viewIdx = 0, cnt = m_Drawable->GetViewCount(); viewIdx < cnt; ++viewIdx)
    {
        if (m_Drawable->GetTexture(viewIdx) == nullptr)
            continue;

        m_Graphics->BeginFrame();
        // compute passes
        MTL::ComputeCommandEncoder* computeEncoder = m_Graphics->CreateComputeCommandEncoder();

        EncodeSkinningPass(computeEncoder);
        EncodeBackfaceCullPass(computeEncoder);
        EncodeCalcProjDataPass(computeEncoder);
        EncodeGenVisiblePointsPass(computeEncoder);
        EncodeSortPass(computeEncoder);

        computeEncoder->endEncoding();

        // render pass
        MTL::RenderCommandEncoder* renderEncoder = m_Graphics->CreateGaussianRenderCommandEncoder(m_Drawable.get(), viewIdx);
        m_Graphics->SetPipelineStateToEncoder(renderEncoder, m_GaussianRenderPipeline.get());

        uint32_t offsetBytes = 0u;
        Buffer* uniformBuffer = m_Graphics->AllocatePerFrameConstantsBuffer(sizeof(RenderGaussianConstants), offsetBytes);
        RenderGaussianConstants* constants = (RenderGaussianConstants*)uniformBuffer->GetDataToModify(offsetBytes);
        constants->_CustomScreenParams = simd_make_float4(m_Drawable->GetLogicalSize(), 0.0f, 0.0f);
        renderEncoder->setVertexBuffer(uniformBuffer->GetHandle(), offsetBytes, 0);
        renderEncoder->setVertexBuffer(m_SplatProjData0Buffer->GetHandle(), 0, 1);
#if IS_STEREO_PIPELINE
        renderEncoder->setVertexBuffer(m_SplatProjData1Buffer->GetHandle(), 0, 2);
        renderEncoder->setVertexBuffer(m_SplatSortKeysBuffer->GetHandle(), 0, 3);
#else
        renderEncoder->setVertexBuffer(m_SplatSortKeysBuffer->GetHandle(), 0, 2);
#endif
        if (m_QuadMesh == nullptr)
        {
            m_QuadMesh = std::make_unique<Mesh>(m_Graphics.get());
            uint32_t indices[] = { 0, 1, 2, 1, 3, 2 };
            m_QuadMesh->SetIndexBufferData((uint8_t*)indices, sizeof(indices));
        }
#if ENABLE_INDIRECT
        Buffer* indirectBuffer = m_IndirectDrawArgsBuffer.get();
#else
        Buffer* indirectBuffer = nullptr;
#endif
        m_Graphics->DrawMesh(renderEncoder, m_QuadMesh.get(), m_GaussianAsset->GetSplatCount(), indirectBuffer);
        renderEncoder->endEncoding();
        
        // post process compute pass
#if PLATFORM_VISIONOS        
        renderEncoder = m_Graphics->CreatePostprocessRenderCommandEncoder(m_Drawable.get(), viewIdx);
        m_Graphics->SetPipelineStateToEncoder(renderEncoder, m_PostprocessRenderPipeline.get());
        EncodePostprocessRenderPass(renderEncoder, viewIdx);
        renderEncoder->endEncoding();

        renderEncoder = m_Graphics->CreateDrawDepthRenderCommandEncoder(m_Drawable.get(), viewIdx);
        m_Graphics->SetPipelineStateToEncoder(renderEncoder, m_MeshDepthRenderPipeline.get());
        EncodeMeshDepthRenderPass(renderEncoder, viewIdx);
        renderEncoder->endEncoding();
#endif
        
        m_Graphics->PresentDrawable(m_Drawable.get());
        m_Graphics->CommitCommandBuffer();
        m_Graphics->EndFrame();
    }
}

void Renderer::LogicalUpdate()
{    
#ifndef PLATFORM_VISIONOS
    m_ViewMatrix[0].columns[3] = simd_make_float4(0.0f, -1.2f, 2.5f, 1.0f);
    simd::float2 screenSize = simd_make_float2(1280, 720);
    if (m_Drawable != nullptr)
        screenSize = m_Drawable->GetLogicalSize();
    float fov = 60.0f;
    float aspect = screenSize.x / screenSize.y;
    float near = 0.1f;
    float far = 5.0f;
    m_ProjMatrix[0] = MakeProjectionMatrix(near, far, fov, aspect);
#else
    m_ModelMatrix.columns[3] = simd_make_float4(0.0f, -1.2f, 2.0f, 1.0f);
#endif
    
    m_Timer->Update();
    float deltaTime = m_Timer->GetDeltaTime();

    m_PoseShadowCompensationBuffer = m_Graphics->AllocatePerFramePoseShadowCompensationBuffer();
    
    m_AnimationPlayer->Update(deltaTime);
    
    m_NeuralCompensation->UpdateFrame();
    
    ApplyBlendShape();
}

void Renderer::SetCameraMatrix(uint32_t eyeIdx, const simd::float4x4& viewMat, const simd::float4x4& projMat)
{    
    m_ViewMatrix[eyeIdx] = RightToLeftHanded(viewMat);
    simd::float4x4 proj = RightToLeftHanded(projMat);
    // flipY
    proj.columns[0].y = -proj.columns[0].y;
    proj.columns[1].y = -proj.columns[1].y;
    proj.columns[2].y = -proj.columns[2].y;
    proj.columns[3].y = -proj.columns[3].y;
    // adjust projection
    proj.columns[2].z = 2.0f * proj.columns[2].z + 1;
    proj.columns[3].z = 2.0f * proj.columns[3].z;
    m_ProjMatrix[eyeIdx] = proj;
}

std::unique_ptr<Buffer> Renderer::CreateBuffer(uint32_t size, const std::string& name, MemoryType memoryType)
{
    std::unique_ptr<Buffer> buffer = std::make_unique<Buffer>(m_Graphics.get(), size, memoryType);
    buffer->SetName(name);
    return buffer;
}

void Renderer::SetBlendShapeData(std::unique_ptr<BlendShapeData>&& blendShapeData)
{
    m_BlendShapeData = std::move(blendShapeData);
}

void Renderer::CalculateBoneMatrices()
{
    if (m_BonePositions.empty())
    {
        LOG_ERROR("Bone positions is empty!");
        return;
    }

    int boneCount = (int)m_BonePositions.size();
    if (boneCount != m_BoneMatrices.size())
    {
        m_BoneMatrices.resize(boneCount);
        m_SkinnedPositions.resize(boneCount);
        m_SkinnedQuaternions.resize(boneCount);
    }

    // step 1: create local bone matrices
    for (int i = 0; i < boneCount; ++i)
    {
        m_BoneMatrices[i] = MakeTransformMatrix(m_BonePositions[i], m_BoneQuaternions[i]);
    }

    // step 2: calculate bone's Object Space matrices
    assert(!m_BoneParentIndices.empty());
    // dp with a little bit trick: i must greater than m_BoneParentIndices[i]
    for (int i = 0; i < boneCount; ++i)
    {
        int parentIdx = m_BoneParentIndices[i];
        assert(i > parentIdx);
        if (parentIdx != -1)
            m_BoneMatrices[i] = m_BoneMatrices[parentIdx] * m_BoneMatrices[i];
    }

    // step 3: calculate skinned positions and quaternions
    for (int i = 0; i < boneCount; ++i)
    {
        m_BoneMatrices[i] = m_BoneMatrices[i] * m_BindPoses[i];
        m_SkinnedPositions[i] = m_BoneMatrices[i].columns[3].xyz;
        m_SkinnedQuaternions[i] = MatrixToQuaternion(m_BoneMatrices[i]);
    }
}

void Renderer::ApplyBlendShape()
{
    m_CachedBSVertexBuffer = m_Graphics->AllocatePerFrameVertexBuffer();
    
    const uint8_t* inVertexBuffer = m_Mesh->GetVertexBuffer()->GetData();
    uint8_t* outVertexBuffer = m_CachedBSVertexBuffer->GetDataToModify();
    uint32_t size = (uint32_t)m_Mesh->GetVertexBuffer()->GetSize();

    assert(size == m_CachedBSVertexBuffer->GetSize() && size != 0);
    std::memcpy(outVertexBuffer, inVertexBuffer, size);
    
    if (m_BlendShapeData == nullptr)
        return;

    if (m_BlendShapeData->frames.size() != m_BlendShapeWeights.size())
    {
        LOG_ERROR("The size of weights is not equal the size of blendshapes' frames");
        return;
    }

    // find first position ptr
    uint8_t* positions = nullptr;
    for (const auto& layout : m_Mesh->GetVertexLayout())
    {
        switch (layout.attribute)
        {
        case VertexAttributeType::Position:
            positions = outVertexBuffer + layout.offsetBytes;
            break;
        default:
            break;
        }
    }

    assert(positions != nullptr);

    // apply blend shape to positions
    for (uint32_t idx = 0, sz = (uint32_t)m_BlendShapeWeights.size(); idx < sz; ++idx)
    {
        float weight = m_BlendShapeWeights[idx];
        const BlendShapeFrame& frame = m_BlendShapeData->frames[idx];
        ApplyBlendShapeFrame(frame, weight, positions);
    }
}

void Renderer::ApplyBlendShapeFrame(const BlendShapeFrame& frame, float weight, uint8_t* positions)
{
    // convert weight to relative weight
    weight = simd::clamp(weight, 0.0f, frame.weight);
    if (weight == 0.0f)
        return;
    float weightInPercent = weight / frame.weight;

    if (positions == nullptr)
    {
        LOG_ERROR("positions ptr is nullptr!");
        return;
    }

    uint32_t vertexStride = m_Mesh->GetVertexStride();
    for (uint32_t idx = 0, sz = (uint32_t)frame.positionIndices.size(); idx < sz; ++idx)
    {
        uint32_t index = frame.positionIndices[idx];
        const float* posOffset = &frame.positionsOffsets[idx * 3];
//        const simd::float3& posOffset = frame.positionsOffsets[idx];
        float* posValue = (float*)(positions + index * vertexStride);
        for (uint32_t j = 0; j < 3; ++j)
        {
            *(posValue + j) += weightInPercent *  (*(posOffset + j));
        }
    }
}

void Renderer::EncodeSkinningPass(MTL::ComputeCommandEncoder* encoder)
{
    CalculateBoneMatrices();

    encoder->pushDebugGroup(NS::String::string("Skinning Pass", NS::StringEncoding::UTF8StringEncoding));
    encoder->setComputePipelineState(m_SkinningPipeline->GetPipelineState());
    Buffer* skinningBuffer = m_Graphics->AllocatePerFrameSkinningBuffer();
    SkinningConstants* constants = (SkinningConstants*)skinningBuffer->GetDataToModify();

    uint32_t vertexCount = m_Mesh->GetVertexCount();
    uint32_t weightOffset = UINT32_MAX;
    uint32_t indexOffset = UINT32_MAX;
    for (const auto& layout : m_Mesh->GetVertexLayout())
    {
        switch (layout.attribute)
        {
        case VertexAttributeType::BoneWeight:
            weightOffset = layout.offsetBytes;
            break;
        case VertexAttributeType::BoneIndice:
            indexOffset = layout.offsetBytes;
            break;
        default:
            break;
        }
    }
    if (weightOffset == UINT32_MAX || indexOffset == UINT32_MAX)
    {
        encoder->popDebugGroup();
        LOG_ERROR("Bone weight or index offset is not found!");
        return;
    }

    constants->_MeshLayout = AsFloat4(m_Mesh->GetVertexStride(), weightOffset, indexOffset, vertexCount);
    for (uint32_t i = 0, sz = (uint32_t)m_SkinnedPositions.size(); i < sz; ++i)
    {
        constants->_BonePositions[i] = simd_make_float4(m_SkinnedPositions[i],  1.0f);
        constants->_BoneRotations[i] = m_SkinnedQuaternions[i];
    }
    encoder->setBuffer(skinningBuffer->GetHandle(), 0, 0);
    encoder->setBuffer(m_CachedBSVertexBuffer->GetHandle(), 0, 1);
    encoder->setBuffer(m_SkinnedVertexBuffer->GetHandle(), 0, 2);

    uint32_t groupCount = (vertexCount + GROUP_SIZE - 1) / GROUP_SIZE;
    encoder->dispatchThreadgroups(MTL::Size(groupCount, 1, 1), MTL::Size(GROUP_SIZE, 1, 1));
    encoder->insertDebugSignpost(NS::String::string("Skinning", NS::StringEncoding::UTF8StringEncoding));
    encoder->popDebugGroup();
}

void Renderer::EncodeBackfaceCullPass(MTL::ComputeCommandEncoder* encoder)
{
    if (m_GaussianAsset == nullptr)
    {
        LOG_ERROR("Gaussian asset is nullptr!");
        return;
    }

    encoder->pushDebugGroup(NS::String::string("Backface Cull Pass", NS::StringEncoding::UTF8StringEncoding));
    encoder->setComputePipelineState(m_BackfaceCullPipeline->GetPipelineState());
    
    /*
    constant type_Constants& Constants [[buffer(0)]],
    const device type_StructuredBuffer_int& _MeshIndice [[buffer(1)]],
    const device type_ByteAddressBuffer& _MeshVertice [[buffer(2)]],
    const device type_ByteAddressBuffer& _GSTriangleProp [[buffer(3)]],
    device type_RWStructuredBuffer_int& _TriangleCullFlag [[buffer(4)]],
    */
    uint32_t triangleCount = m_Mesh->GetTriangleCount();
    uint32_t offsetBytes = 0u;
    Buffer* constantsBuffer = m_Graphics->AllocatePerFrameConstantsBuffer(sizeof(BackfaceCullConstants), offsetBytes);
    BackfaceCullConstants* constants = (BackfaceCullConstants*)constantsBuffer->GetDataToModify(offsetBytes);
    constants->_MatrixWorldToObject = simd::inverse(m_ModelMatrix);
    constants->_CameraPosWS[0] = simd::make_float4(simd::inverse(m_ViewMatrix[0]).columns[3].xyz, 1.0f);
#ifdef IS_STEREO_PIPELINE
    constants->_CameraPosWS[1] = simd::make_float4(simd::inverse(m_ViewMatrix[1]).columns[3].xyz, 1.0f);
#endif
    constants->_MeshLayout = AsFloat4(m_Mesh->GetVertexStride(), triangleCount);
    encoder->setBuffer(constantsBuffer->GetHandle(), offsetBytes, 0);

    encoder->setBuffer(m_Mesh->GetIndexBuffer()->GetHandle(), 0, 1);
    encoder->setBuffer(m_SkinnedVertexBuffer->GetHandle(), 0, 2);
    encoder->setBuffer(m_GaussianAsset->GetFacePropData()->GetHandle(), 0, 3);
    if (m_TriangleCullFlagBuffer == nullptr)
    {
        m_TriangleCullFlagBuffer = CreateBuffer(triangleCount * sizeof(int), "Triangle Cull Flag Buffer");
    }
    encoder->setBuffer(m_TriangleCullFlagBuffer->GetHandle(), 0, 4);
    
    uint32_t groupCount = (triangleCount + GROUP_SIZE - 1) / GROUP_SIZE;
    encoder->dispatchThreadgroups(MTL::Size(groupCount, 1, 1), MTL::Size(GROUP_SIZE, 1, 1));
    encoder->insertDebugSignpost(NS::String::string("Backface Cull", NS::StringEncoding::UTF8StringEncoding));
    encoder->popDebugGroup();
}

void Renderer::EncodeCalcProjDataPass(MTL::ComputeCommandEncoder* encoder)
{
    if (m_GaussianAsset == nullptr)
    {
        LOG_ERROR("Gaussian asset is nullptr!");
        return;
    }

    encoder->pushDebugGroup(NS::String::string("Calc View Data Pass", NS::StringEncoding::UTF8StringEncoding));
    encoder->setComputePipelineState(m_CalcProjDataPipeline->GetPipelineState());
    /*
    constant type_Constants& Constants [[buffer(0)]],
    const device type_StructuredBuffer_int& _GSIndice [[buffer(1)]],
    const device type_StructuredBuffer_int& _TriangleCullFlag [[buffer(2)]],
    const device type_ByteAddressBuffer& _SplatPropData [[buffer(3)]],
    const device type_StructuredBuffer_SplatChunkInfo& _SplatChunks [[buffer(4)]],
    const device type_ByteAddressBuffer& _SplatPos [[buffer(5)]],
    const device type_ByteAddressBuffer& _SplatOther [[buffer(6)]],
    const device type_ByteAddressBuffer& _SplatSH [[buffer(7)]],
    const device type_StructuredBuffer_int& _MeshIndice [[buffer(8)]],
    const device type_ByteAddressBuffer& _MeshVertice [[buffer(9)]],
    const device type_StructuredBuffer_float& _PoseShadowCompensation [[buffer(10)]],
    device type_RWStructuredBuffer_uint& _VisibleMasks [[buffer(11)]],
    device type_RWStructuredBuffer_float& _PointDistances [[buffer(12)]],
    device type_RWStructuredBuffer_SplatProjData& _SplatProjData0 [[buffer(13)]],
    device type_RWStructuredBuffer_SplatProjData& _SplatProjData1 [[buffer(14)]],
    texture2d<float> _SplatColor [[texture(0)]],
    */
    uint32_t splatCount = m_GaussianAsset->GetSplatCount();
    uint32_t splatFormat = (uint32_t)m_GaussianAsset->GetPositionFormat() |
                           ((uint32_t)m_GaussianAsset->GetScaleFormat() << 8) |
                           ((uint32_t)m_GaussianAsset->GetSHFormat() << 16);
    uint32_t offsetBytes = 0u;
    Buffer* constantsBuffer = m_Graphics->AllocatePerFrameConstantsBuffer(sizeof(CalcProjDataConstants), offsetBytes);
    CalcProjDataConstants* constants = (CalcProjDataConstants*)constantsBuffer->GetDataToModify(offsetBytes);
    constants->_MatrixWorldToObject = simd::inverse(m_ModelMatrix);
    constants->_MatrixM = m_ModelMatrix;
    constants->_MatrixV[0] = m_ViewMatrix[0];
    constants->_MatrixP[0] = m_ProjMatrix[0];
    constants->_CameraPosWS[0] = simd::make_float4(simd::inverse(m_ViewMatrix[0]).columns[3].xyz, 1.0f);
    constants->_SplatProp = AsFloat4(splatCount, splatFormat, m_SHOrder, m_Mesh->GetVertexStride());
    constants->_VecScreenParams = simd_make_float4(m_Drawable->GetLogicalSize().x, 0, 0, 0);
#if IS_STEREO_PIPELINE
    constants->_MatrixV[1] = m_ViewMatrix[1];
    constants->_MatrixP[1] = m_ProjMatrix[1];
    constants->_CameraPosWS[1] = simd::make_float4(simd::inverse(m_ViewMatrix[1]).columns[3].xyz, 1.0f);
#endif
    encoder->setBuffer(constantsBuffer->GetHandle(), offsetBytes, 0);

    encoder->setBuffer(m_GaussianAsset->GetIdxData()->GetHandle(), 0, 1);
    if (m_TriangleCullFlagBuffer == nullptr)
    {
        m_TriangleCullFlagBuffer = CreateBuffer(m_Mesh->GetTriangleCount() * sizeof(uint32_t), "Triangle Cull Flag Buffer");
    }
    encoder->setBuffer(m_TriangleCullFlagBuffer->GetHandle(), 0, 2);
    encoder->setBuffer(m_GaussianAsset->GetGaussianPropData()->GetHandle(), 0, 3);  
    encoder->setBuffer(m_GaussianAsset->GetChunkData()->GetHandle(), 0, 4);
    encoder->setBuffer(m_GaussianAsset->GetPosData()->GetHandle(), 0, 5);
    encoder->setBuffer(m_GaussianAsset->GetOtherData()->GetHandle(), 0, 6);
    encoder->setBuffer(m_GaussianAsset->GetSHData()->GetHandle(), 0, 7);
    encoder->setBuffer(m_Mesh->GetIndexBuffer()->GetHandle(), 0, 8);
    encoder->setBuffer(m_SkinnedVertexBuffer->GetHandle(), 0, 9);
    encoder->setBuffer(m_PoseShadowCompensationBuffer->GetHandle(), 0, 10);
    if (m_VisibleMasksBuffer == nullptr)
    {
        m_VisibleMasksBuffer = CreateBuffer(splatCount * sizeof(uint32_t), "Visible Masks Buffer");
    }
    encoder->setBuffer(m_VisibleMasksBuffer->GetHandle(), 0, 11);
    if (m_PointDistancesBuffer == nullptr)
    {
        m_PointDistancesBuffer = CreateBuffer(splatCount * sizeof(float), "Point Distances Buffer");
    }
    encoder->setBuffer(m_PointDistancesBuffer->GetHandle(), 0, 12);
    if (m_SplatProjData0Buffer == nullptr)
    {
        m_SplatProjData0Buffer = CreateBuffer(splatCount * sizeof(SplatProjData), "Splat View Data 0 Buffer");
    }
    encoder->setBuffer(m_SplatProjData0Buffer->GetHandle(), 0, 13);
#if IS_STEREO_PIPELINE
    if (m_SplatProjData1Buffer == nullptr)
    {
        m_SplatProjData1Buffer = CreateBuffer(splatCount * sizeof(SplatProjData), "Splat View Data 1 Buffer");
    }
    encoder->setBuffer(m_SplatProjData1Buffer->GetHandle(), 0, 14);
#endif
    encoder->setTexture(m_GaussianAsset->GetColorTexture(), 0);
    
    uint32_t groupCount = (splatCount + GROUP_SIZE - 1) / GROUP_SIZE;
    encoder->dispatchThreadgroups(MTL::Size(groupCount, 1, 1), MTL::Size(GROUP_SIZE, 1, 1));
    encoder->insertDebugSignpost(NS::String::string("Calc View Data", NS::StringEncoding::UTF8StringEncoding));
    encoder->popDebugGroup();
}

void Renderer::EncodeGenVisiblePointsPass(MTL::ComputeCommandEncoder* encoder)
{
    if (m_GaussianAsset == nullptr)
    {
        LOG_ERROR("Gaussian asset is nullptr!");
        return;
    }

    // step 1: compute prefix sum
    encoder->pushDebugGroup(NS::String::string("Prefix Sum Pass", NS::StringEncoding::UTF8StringEncoding));
    ComputePrefixSum(encoder);
    encoder->popDebugGroup();

    // step 2: generate visible points
    encoder->pushDebugGroup(NS::String::string("Gen Visible Points Pass", NS::StringEncoding::UTF8StringEncoding));
    uint32_t splatCount = m_GaussianAsset->GetSplatCount();
    uint32_t sortCount = (splatCount + SORT_BLOCK_SIZE - 1) / SORT_BLOCK_SIZE * SORT_BLOCK_SIZE;
    encoder->setComputePipelineState(m_GenVisiblePointsPipeline->GetPipelineState());
    /*
    constant type_Constants& Constants [[buffer(0)]],
    const device type_StructuredBuffer_uint& _VisibleMaskPrefixSums [[buffer(1)]],
    device type_RWStructuredBuffer_uint& _VisiblePointList [[buffer(2)]],
    */
    uint32_t offsetBytes = 0u;
    Buffer* constantsBuffer = m_Graphics->AllocatePerFrameConstantsBuffer(sizeof(GenVisiblePointsConstants), offsetBytes);
    GenVisiblePointsConstants* visiblePointsConstants = (GenVisiblePointsConstants*)constantsBuffer->GetDataToModify(offsetBytes);
    visiblePointsConstants->_Count = AsFloat4(splatCount);
    encoder->setBuffer(constantsBuffer->GetHandle(), offsetBytes, 0);
    encoder->setBuffer(m_VisibleMaskPrefixSumsBuffer->GetHandle(), 0, 1);
    if (m_SplatSortKeysBuffer == nullptr)
    {
        m_SplatSortKeysBuffer = CreateBuffer(sortCount * sizeof(uint32_t), "Splat Sort Keys Buffer");
    }
    encoder->setBuffer(m_SplatSortKeysBuffer->GetHandle(), 0, 2);

    uint32_t groupCount = (splatCount + GROUP_SIZE - 1) / GROUP_SIZE;
    encoder->dispatchThreadgroups(MTL::Size(groupCount, 1, 1), MTL::Size(GROUP_SIZE, 1, 1));
    encoder->insertDebugSignpost(NS::String::string("Gen Visible Points", NS::StringEncoding::UTF8StringEncoding));
    encoder->popDebugGroup();

    // step 3: generate indirect draw args
    encoder->pushDebugGroup(NS::String::string("Gen Indirect Draw Args Pass", NS::StringEncoding::UTF8StringEncoding));
    encoder->setComputePipelineState(m_GenIndirectDrawArgsPipeline->GetPipelineState());
    /*
    constant type_Constants& Constants [[buffer(0)]],
    const device type_StructuredBuffer_uint& _VisibleMaskPrefixSums [[buffer(1)]],
    device type_RWStructuredBuffer_uint& _IndirectDrawArgs [[buffer(2)]],
    device type_RWStructuredBuffer_uint& _SortParams [[buffer(3)]],
    device type_RWStructuredBuffer_uint& _SortSumPassIndirectBuffer [[buffer(4)]],
    device type_RWStructuredBuffer_uint& _SortReducePassIndirectBuffer [[buffer(5)]],
    device type_RWStructuredBuffer_uint& _SortCopyPassIndirectBuffer [[buffer(6)]]
    */
    offsetBytes = 0u;
    constantsBuffer = m_Graphics->AllocatePerFrameConstantsBuffer(sizeof(GenIndirectDrawArgsConstants), offsetBytes);
    GenIndirectDrawArgsConstants* indirectDrawArgsConstants = (GenIndirectDrawArgsConstants*)constantsBuffer->GetDataToModify(offsetBytes);
    indirectDrawArgsConstants->_PointCount = AsFloat4(splatCount);
    encoder->setBuffer(constantsBuffer->GetHandle(), offsetBytes, 0);
    encoder->setBuffer(m_VisibleMaskPrefixSumsBuffer->GetHandle(), 0, 1);

    if (m_IndirectDrawArgsBuffer == nullptr)
    {
        m_IndirectDrawArgsBuffer = CreateBuffer(8 * sizeof(uint32_t), "Indirect Draw Args Buffer");
        m_GPUSortDistancesBuffer = CreateBuffer(sortCount * sizeof(uint32_t), "GPU Sort Distances Buffer");
        // init sort args
        m_SortArgs.count = splatCount;
        m_SortArgs.bitCount = SORT_BIT_COUNT;
        m_SortArgs.inputKeys = m_GPUSortDistancesBuffer.get();
        m_SortArgs.inputValues = m_SplatSortKeysBuffer.get();
        m_SortArgs.dstKeys = m_GPUSortDistancesBuffer.get();
        m_SortArgs.dstValues = m_SplatSortKeysBuffer.get();

        uint32_t blockSize = FFX_PARALLELSORT_ELEMENTS_PER_THREAD * FFX_PARALLELSORT_THREADGROUP_SIZE;
        uint32_t numBlocks = (splatCount + blockSize - 1) / blockSize;
        uint32_t numReducedBlocks = (numBlocks + blockSize - 1) / blockSize;
        uint32_t scratchBufferSize = FFX_PARALLELSORT_SORT_BIN_COUNT * numBlocks;
        uint32_t reduceScratchBufferSize = FFX_PARALLELSORT_SORT_BIN_COUNT * numReducedBlocks;

        m_SortArgs.sortParamsBuffer = CreateBuffer(20 * sizeof(uint32_t), "Sort Params Buffer");

        m_SortArgs.sortScratchBuffer = CreateBuffer( sortCount * sizeof(uint32_t), "Sort Scratch Buffer");
        m_SortArgs.payloadScratchBuffer = CreateBuffer( sortCount * sizeof(uint32_t), "Sort Payload Scratch Buffer");
        m_SortArgs.scratchBuffer = CreateBuffer( scratchBufferSize * sizeof(uint32_t), "Sort Scratch Buffer");
        m_SortArgs.reducedScratchBuffer = CreateBuffer( reduceScratchBufferSize * sizeof(uint32_t), "Sort Reduced Scratch Buffer");
    }
    encoder->setBuffer(m_IndirectDrawArgsBuffer->GetHandle(), 0, 2);
    encoder->setBuffer(m_SortArgs.sortParamsBuffer->GetHandle(), 0, 3);
#if ENABLE_INDIRECT
    encoder->setBuffer(m_SortArgs.sortParamsBuffer->GetHandle(), SORT_SUM_PASS_INDIRECT_BUFFER_OFFSET, 4);
    encoder->setBuffer(m_SortArgs.sortParamsBuffer->GetHandle(), SORT_REDUCE_PASS_INDIRECT_BUFFER_OFFSET, 5);
    encoder->setBuffer(m_SortArgs.sortParamsBuffer->GetHandle(), SORT_COPY_PASS_INDIRECT_BUFFER_OFFSET, 6);
#endif
    encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
    encoder->insertDebugSignpost(NS::String::string("Gen Indirect Draw Args", NS::StringEncoding::UTF8StringEncoding));
    encoder->popDebugGroup();
}

void Renderer::ComputePrefixSum(MTL::ComputeCommandEncoder* encoder)
{
    uint32_t splatCount = m_GaussianAsset->GetSplatCount();
    assert(splatCount < (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE));
    uint32_t blockNum1 = (splatCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint32_t blockNum2 = (blockNum1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (m_VisibleMaskPrefixSumsBuffer == nullptr)
    {
        // init prefix sum args
        m_VisibleMaskPrefixSumsBuffer = CreateBuffer(splatCount * sizeof(uint32_t), "Prefix Sums Buffer");
        m_PrefixSumArgs.srcBuffer = m_VisibleMasksBuffer.get();
        m_PrefixSumArgs.dstBuffer = m_VisibleMaskPrefixSumsBuffer.get();

        m_PrefixSumArgs.tempBuffer1 = CreateBuffer(blockNum1 * sizeof(uint32_t), "Prefix Sum Temp Buffer 1");
        m_PrefixSumArgs.tempBuffer2 = CreateBuffer(blockNum2 * sizeof(uint32_t), "Prefix Sum Temp Buffer 2");
        m_PrefixSumArgs.inclusive = true;
    }

    MTL::Buffer* srcBuffer = m_PrefixSumArgs.srcBuffer->GetHandle();
    MTL::Buffer* dstBuffer = m_PrefixSumArgs.dstBuffer->GetHandle();
    MTL::Buffer* blockSumBuffer1 = m_PrefixSumArgs.tempBuffer1->GetHandle();
    MTL::Buffer* blockSumBuffer2 = m_PrefixSumArgs.tempBuffer2->GetHandle();

    // step1: reduce 'srcBuffer' into blocks and compute 'blockSumBuffer1' (sum of elements in block)
    encoder->setComputePipelineState(m_PrefixSumReducePipeline->GetPipelineState());
    /*
    constant type_Constants& Constants [[buffer(0)]],
    const device type_StructuredBuffer_uint& _ReduceSrcValues [[buffer(1)]],
    device type_RWStructuredBuffer_uint& _ReduceDstValues [[buffer(2)]],
    */
    uint32_t offsetBytes = 0u;
    Buffer* constantsBuffer = m_Graphics->AllocatePerFrameConstantsBuffer(sizeof(PrefixSumReduceConstants), offsetBytes);
    PrefixSumReduceConstants* reduceConstants = (PrefixSumReduceConstants*)constantsBuffer->GetDataToModify(offsetBytes);
    reduceConstants->_Count = AsFloat4(splatCount);
    encoder->setBuffer(constantsBuffer->GetHandle(), offsetBytes, 0);
    encoder->setBuffer(srcBuffer, 0, 1);
    encoder->setBuffer(blockSumBuffer1, 0, 2);

    encoder->dispatchThreadgroups(MTL::Size(blockNum1, 1, 1), MTL::Size(THREAD_NUM_PER_GROUP, 1, 1));
    encoder->insertDebugSignpost(NS::String::string("Prefix Sum Reduce1", NS::StringEncoding::UTF8StringEncoding));

    // step2: reduce again and output 'blockSumBuffer2' (sum of elements in 'blockSumBuffer1')
    encoder->setComputePipelineState(m_PrefixSumReducePipeline->GetPipelineState());
    /*
    constant type_Constants& Constants [[buffer(0)]],
    const device type_StructuredBuffer_uint& _ReduceSrcValues [[buffer(1)]],
    device type_RWStructuredBuffer_uint& _ReduceDstValues [[buffer(2)]],
    */
    offsetBytes = 0u;
    constantsBuffer = m_Graphics->AllocatePerFrameConstantsBuffer(sizeof(PrefixSumReduceConstants), offsetBytes);
    reduceConstants = (PrefixSumReduceConstants*)constantsBuffer->GetDataToModify(offsetBytes);
    reduceConstants->_Count = AsFloat4(blockNum1);
    encoder->setBuffer(constantsBuffer->GetHandle(), offsetBytes, 0);
    encoder->setBuffer(blockSumBuffer1, 0, 1);
    encoder->setBuffer(blockSumBuffer2, 0, 2);

    encoder->dispatchThreadgroups(MTL::Size(blockNum2, 1, 1), MTL::Size(THREAD_NUM_PER_GROUP, 1, 1));
    encoder->insertDebugSignpost(NS::String::string("Prefix Sum Reduce2", NS::StringEncoding::UTF8StringEncoding));

    // step3: scan 'blockSumBuffer2' to compute prefix sum in global scope
    encoder->setComputePipelineState(m_PrefixSumScanPipeline->GetPipelineState());
    /*
    constant type_Constants& Constants [[buffer(0)]],
    const device type_StructuredBuffer_uint& _ScanSrcValues [[buffer(1)]],
    device type_RWStructuredBuffer_uint& _ScanDstValues [[buffer(2)]]
    */
    offsetBytes = 0u;
    constantsBuffer = m_Graphics->AllocatePerFrameConstantsBuffer(sizeof(PrefixSumScanConstants), offsetBytes);
    PrefixSumScanConstants* scanConstants = (PrefixSumScanConstants*)constantsBuffer->GetDataToModify(offsetBytes);
    scanConstants->_Count = AsFloat4(blockNum2);
    encoder->setBuffer(constantsBuffer->GetHandle(), offsetBytes, 0);
    encoder->setBuffer(blockSumBuffer2, 0, 1);
    encoder->setBuffer(blockSumBuffer2, 0, 2);

    encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(THREAD_NUM_PER_GROUP, 1, 1));
    encoder->insertDebugSignpost(NS::String::string("Prefix Sum Scan", NS::StringEncoding::UTF8StringEncoding));

    // step4: scan 'blockSumBuffer1' with 'blockSumBuffer2' to compute prefix sum in global scope
    encoder->setComputePipelineState(m_PrefixSumScanAddPipeline->GetPipelineState());
    /*
    constant type_Constants& Constants [[buffer(0)]],
    const device type_StructuredBuffer_uint& _ScanSrcValues [[buffer(1)]],
    const device type_StructuredBuffer_uint& _ScanAddSrcValues [[buffer(2)]],
    device type_RWStructuredBuffer_uint& _ScanDstValues [[buffer(3)]],
    */
    offsetBytes = 0u;
    constantsBuffer = m_Graphics->AllocatePerFrameConstantsBuffer(sizeof(PrefixSumScanAddConstants), offsetBytes);
    PrefixSumScanAddConstants* scanAddConstants = (PrefixSumScanAddConstants*)constantsBuffer->GetDataToModify(offsetBytes);
    scanAddConstants->_Count = AsFloat4(blockNum1);
    encoder->setBuffer(constantsBuffer->GetHandle(), offsetBytes, 0);
    encoder->setBuffer(blockSumBuffer1, 0, 1);
    encoder->setBuffer(blockSumBuffer2, 0, 2);
    encoder->setBuffer(blockSumBuffer1, 0, 3);

    encoder->dispatchThreadgroups(MTL::Size(blockNum2, 1, 1), MTL::Size(THREAD_NUM_PER_GROUP, 1, 1));
    encoder->insertDebugSignpost(NS::String::string("Prefix Sum ScanAdd", NS::StringEncoding::UTF8StringEncoding));

    // step5: scan 'srcBuffer' with 'blockSumBuffer1' to compute prefix sum in global scope
    auto pipelineState = m_PrefixSumArgs.inclusive ? m_PrefixSumScanAddInclusivePipeline->GetPipelineState() : m_PrefixSumScanAddPipeline->GetPipelineState();
    encoder->setComputePipelineState(pipelineState);
    /*
    constant type_Constants& Constants [[buffer(0)]],
    const device type_StructuredBuffer_uint& _ScanSrcValues [[buffer(1)]],
    const device type_StructuredBuffer_uint& _ScanAddSrcValues [[buffer(2)]],
    device type_RWStructuredBuffer_uint& _ScanDstValues [[buffer(3)]],
    */
    offsetBytes = 0u;
    constantsBuffer = m_Graphics->AllocatePerFrameConstantsBuffer(sizeof(PrefixSumScanAddInclusiveConstants), offsetBytes);
    PrefixSumScanAddInclusiveConstants* scanAddInclusiveConstants = (PrefixSumScanAddInclusiveConstants*)constantsBuffer->GetDataToModify(offsetBytes);
    scanAddInclusiveConstants->_Count = AsFloat4(splatCount);
    encoder->setBuffer(constantsBuffer->GetHandle(), offsetBytes, 0);
    encoder->setBuffer(srcBuffer, 0, 1);
    encoder->setBuffer(blockSumBuffer1, 0, 2);
    encoder->setBuffer(dstBuffer, 0, 3);

    encoder->dispatchThreadgroups(MTL::Size(blockNum1, 1, 1), MTL::Size(THREAD_NUM_PER_GROUP, 1, 1));
    encoder->insertDebugSignpost(NS::String::string("Prefix Sum ScanAddInclusive", NS::StringEncoding::UTF8StringEncoding));
}

void Renderer::EncodeSortPass(MTL::ComputeCommandEncoder* encoder)
{
    if (m_GaussianAsset == nullptr)
    {
        LOG_ERROR("Gaussian asset is nullptr!");
        return;
    }

    uint32_t splatCount = m_GaussianAsset->GetSplatCount();
    assert(m_SortArgs.inputKeys == m_SortArgs.dstKeys && m_SortArgs.inputValues == m_SortArgs.dstValues);
    // step 1: calculate sortable distances
    encoder->pushDebugGroup(NS::String::string("Calc Sortable Distances Pass", NS::StringEncoding::UTF8StringEncoding));
    encoder->setComputePipelineState(m_CalcDistancesPipeline->GetPipelineState());
    /*
    const device type_StructuredBuffer_uint& _SortParams [[buffer(0)]],
    const device type_StructuredBuffer_float& _PointDistances [[buffer(1)]],
    const device type_StructuredBuffer_uint& _SplatSortKeys [[buffer(2)]],
    device type_RWStructuredBuffer_uint& _SplatSortDistances [[buffer(3)]],
    */
    encoder->setBuffer(m_SortArgs.sortParamsBuffer->GetHandle(), 0, 0);
    encoder->setBuffer(m_PointDistancesBuffer->GetHandle(), 0, 1);
    encoder->setBuffer(m_SplatSortKeysBuffer->GetHandle(), 0, 2);
    encoder->setBuffer(m_GPUSortDistancesBuffer->GetHandle(), 0, 3);

    uint32_t groupCount = (splatCount + GROUP_SIZE - 1) / GROUP_SIZE;
    encoder->dispatchThreadgroups(MTL::Size(groupCount, 1, 1), MTL::Size(GROUP_SIZE, 1, 1));
    encoder->insertDebugSignpost(NS::String::string("Calc Sortable Distances", NS::StringEncoding::UTF8StringEncoding));
    encoder->popDebugGroup();

    // step 2: sort distances
    uint32_t maxIterCount = SORT_BIT_COUNT / FFX_PARALLELSORT_SORT_BITS_PER_PASS;
    Buffer* srcKeyBuffer = m_SortArgs.inputKeys;
    Buffer* srcPayloadBuffer = m_SortArgs.inputValues;
    Buffer* dstKeyBuffer = m_SortArgs.sortScratchBuffer.get();
    Buffer* dstPayloadBuffer = m_SortArgs.payloadScratchBuffer.get();
    uint32_t numBlocks = (splatCount + SORT_BLOCK_SIZE - 1) / SORT_BLOCK_SIZE;
    uint32_t numThreadGroupsToRun = FFX_PARALLELSORT_MAX_THREADGROUPS_TO_RUN;
    uint32_t blocksPerThreadGroup = numBlocks / numThreadGroupsToRun;
    uint32_t blocksWithAdditionalBlocks = numBlocks % numThreadGroupsToRun;
    if (numBlocks < numThreadGroupsToRun)
    {
        blocksPerThreadGroup = 1;
        numThreadGroupsToRun = numBlocks;
        blocksWithAdditionalBlocks = 0;
    }
    uint32_t numReducedThreadGroupsToRun = FFX_PARALLELSORT_SORT_BIN_COUNT * ((SORT_BLOCK_SIZE > numThreadGroupsToRun) ? 1 : (numThreadGroupsToRun + SORT_BLOCK_SIZE - 1) / SORT_BLOCK_SIZE);
    encoder->pushDebugGroup(NS::String::string("Sort Distances Passes", NS::StringEncoding::UTF8StringEncoding));
    for (uint32_t i = 0; i < maxIterCount; i++)
    {
        encoder->pushDebugGroup(NS::String::string(("Sort Iterate " + std::to_string(i + 1)).c_str(), NS::StringEncoding::UTF8StringEncoding));
        uint32_t shiftBits = i * FFX_PARALLELSORT_SORT_BITS_PER_PASS;
        simd::float4 packedParams1 = AsFloat4(0, 0, shiftBits);  // only shift bits is used

        // Sum Pass
        encoder->setComputePipelineState(m_SortCountPipeline->GetPipelineState());
        /*
        constant type_Constants& Constants [[buffer(0)]],
        const device type_StructuredBuffer_uint& sort_params [[buffer(1)]],
        device type_RWStructuredBuffer_uint& rw_source_keys [[buffer(2)]],
        device type_RWStructuredBuffer_uint& rw_sum_table [[buffer(3)]],
        */
        uint32_t offsetBytes = 0u;
        Buffer* constantsBuffer = m_Graphics->AllocatePerFrameConstantsBuffer(sizeof(SortCountConstants), offsetBytes);
        SortCountConstants* sortCountConstants = (SortCountConstants*)constantsBuffer->GetDataToModify(offsetBytes);
        sortCountConstants->_Packed_Params1 = packedParams1;

        encoder->setBuffer(constantsBuffer->GetHandle(), offsetBytes, 0);
        encoder->setBuffer(m_SortArgs.sortParamsBuffer->GetHandle(), 0, 1);
        encoder->setBuffer(srcKeyBuffer->GetHandle(), 0, 2);
        encoder->setBuffer(m_SortArgs.scratchBuffer->GetHandle(), 0, 3);
#if ENABLE_INDIRECT
        encoder->dispatchThreadgroups(m_SortArgs.sortParamsBuffer->GetHandle(), SORT_SUM_PASS_INDIRECT_BUFFER_OFFSET, MTL::Size(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1));
#else
        encoder->dispatchThreadgroups(MTL::Size(numThreadGroupsToRun, 1, 1), MTL::Size(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1));
#endif
        encoder->insertDebugSignpost(NS::String::string("Sort Distances Sum Pass", NS::StringEncoding::UTF8StringEncoding));

        // Reduce Pass
        encoder->setComputePipelineState(m_SortReducePipeline->GetPipelineState());
        /*
        const device type_StructuredBuffer_uint& sort_params [[buffer(0)]],
        device type_RWStructuredBuffer_uint& rw_sum_table [[buffer(1)]],
        device type_RWStructuredBuffer_uint& rw_reduce_table [[buffer(2)]],
        */
        encoder->setBuffer(m_SortArgs.sortParamsBuffer->GetHandle(), 0, 0);
        encoder->setBuffer(m_SortArgs.scratchBuffer->GetHandle(), 0, 1);
        encoder->setBuffer(m_SortArgs.reducedScratchBuffer->GetHandle(), 0, 2);
#if ENABLE_INDIRECT
        encoder->dispatchThreadgroups(m_SortArgs.sortParamsBuffer->GetHandle(), SORT_REDUCE_PASS_INDIRECT_BUFFER_OFFSET, MTL::Size(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1));
#else
        encoder->dispatchThreadgroups(MTL::Size(numReducedThreadGroupsToRun, 1, 1), MTL::Size(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1));
#endif
        encoder->insertDebugSignpost(NS::String::string("Sort Distances Reduce Pass", NS::StringEncoding::UTF8StringEncoding));

        // Scan Pass
        encoder->setComputePipelineState(m_SortScanPipeline->GetPipelineState());
        /*
        const device type_StructuredBuffer_uint& sort_params [[buffer(0)]],
        device type_RWStructuredBuffer_uint& rw_scan_source [[buffer(1)]],
        device type_RWStructuredBuffer_uint& rw_scan_dest [[buffer(2)]],
        */
        encoder->setBuffer(m_SortArgs.sortParamsBuffer->GetHandle(), 0, 0);
        encoder->setBuffer(m_SortArgs.reducedScratchBuffer->GetHandle(), 0, 1);
        encoder->setBuffer(m_SortArgs.reducedScratchBuffer->GetHandle(), 0, 2);
        encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1));
        encoder->insertDebugSignpost(NS::String::string("Sort Distances Scan Pass", NS::StringEncoding::UTF8StringEncoding));

        // Scan Add Pass
        encoder->setComputePipelineState(m_SortScanAddPipeline->GetPipelineState());
        /*
        const device type_StructuredBuffer_uint& sort_params [[buffer(0)]],
        device type_RWStructuredBuffer_uint& rw_scan_source [[buffer(1)]],
        device type_RWStructuredBuffer_uint& rw_scan_dest [[buffer(2)]],
        device type_RWStructuredBuffer_uint& rw_scan_scratch [[buffer(3)]],
        */
        encoder->setBuffer(m_SortArgs.sortParamsBuffer->GetHandle(), 0, 0);
        encoder->setBuffer(m_SortArgs.scratchBuffer->GetHandle(), 0, 1);
        encoder->setBuffer(m_SortArgs.scratchBuffer->GetHandle(), 0, 2);
        encoder->setBuffer(m_SortArgs.reducedScratchBuffer->GetHandle(), 0, 3);
#if ENABLE_INDIRECT
        encoder->dispatchThreadgroups(m_SortArgs.sortParamsBuffer->GetHandle(), SORT_REDUCE_PASS_INDIRECT_BUFFER_OFFSET, MTL::Size(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1));
#else
        encoder->dispatchThreadgroups(MTL::Size(numReducedThreadGroupsToRun, 1, 1), MTL::Size(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1));
#endif
        encoder->insertDebugSignpost(NS::String::string("Sort Distances Scan Add Pass", NS::StringEncoding::UTF8StringEncoding));

        // Scantter Pass
        encoder->setComputePipelineState(m_SortScatterPipeline->GetPipelineState());
        /*
        constant type_Constants& Constants [[buffer(0)]],
        const device type_StructuredBuffer_uint& sort_params [[buffer(1)]],
        device type_RWStructuredBuffer_uint& rw_source_keys [[buffer(2)]],
        device type_RWStructuredBuffer_uint& rw_dest_keys [[buffer(3)]],
        device type_RWStructuredBuffer_uint& rw_source_payloads [[buffer(4)]],
        device type_RWStructuredBuffer_uint& rw_dest_payloads [[buffer(5)]],
        device type_RWStructuredBuffer_uint& rw_sum_table [[buffer(6)]],
        */
        offsetBytes = 0u;
        constantsBuffer = m_Graphics->AllocatePerFrameConstantsBuffer(sizeof(SortScatterConstants), offsetBytes);
        SortScatterConstants* sortScatterConstants = (SortScatterConstants*)constantsBuffer->GetDataToModify(offsetBytes);
        sortScatterConstants->_Packed_Params1 = packedParams1;

        encoder->setBuffer(constantsBuffer->GetHandle(), offsetBytes, 0);
        encoder->setBuffer(m_SortArgs.sortParamsBuffer->GetHandle(), 0, 1);
        encoder->setBuffer(srcKeyBuffer->GetHandle(), 0, 2);
        encoder->setBuffer(dstKeyBuffer->GetHandle(), 0, 3);
        encoder->setBuffer(srcPayloadBuffer->GetHandle(), 0, 4);
        encoder->setBuffer(dstPayloadBuffer->GetHandle(), 0, 5);
        encoder->setBuffer(m_SortArgs.scratchBuffer->GetHandle(), 0, 6);
#if ENABLE_INDIRECT
        encoder->dispatchThreadgroups(m_SortArgs.sortParamsBuffer->GetHandle(), SORT_SUM_PASS_INDIRECT_BUFFER_OFFSET, MTL::Size(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1));
#else
        encoder->dispatchThreadgroups(MTL::Size(numThreadGroupsToRun, 1, 1), MTL::Size(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1));
#endif
        encoder->insertDebugSignpost(NS::String::string("Sort Distances Scatter Pass", NS::StringEncoding::UTF8StringEncoding));

        std::swap(srcKeyBuffer, dstKeyBuffer);
        std::swap(srcPayloadBuffer, dstPayloadBuffer);
        encoder->popDebugGroup();
    }
    encoder->popDebugGroup();
}

void Renderer::EncodePostprocessComputePass(MTL::ComputeCommandEncoder* encoder, uint32_t viewIdx)
{
    encoder->pushDebugGroup(NS::String::string("Postprocess Pass", NS::StringEncoding::UTF8StringEncoding));
    encoder->setComputePipelineState(m_PostprocessComputePipeline->GetPipelineState());
    /*
        constant type_Constants& Constants [[buffer(0)]],
        texture2d<float, access::read> _InTexture [[texture(0)]],
        texture2d<float, access::write> _OutTexture [[texture(1)]],
    */
    uint32_t offsetBytes = 0u;
    simd::float2 texSize = m_Drawable->GetLogicalSize();
    int pixelCount = std::floor(texSize.x * texSize.y + 0.5f);
    MTL::Texture* texture = m_Drawable->GetTexture(viewIdx);
    
    Buffer* constantsBuffer = m_Graphics->AllocatePerFrameConstantsBuffer(sizeof(PostprocessConstants), offsetBytes);
    PostprocessConstants* postprocessConstants = (PostprocessConstants*)constantsBuffer->GetDataToModify(offsetBytes);
    postprocessConstants->_Params = AsFloat4((int)texSize.x, (int)texSize.y, pixelCount);
    encoder->setBuffer(constantsBuffer->GetHandle(), offsetBytes, 0);
    encoder->setTexture(texture, 0);
    encoder->setTexture(texture, 1);

    uint32_t groupCount = (pixelCount + GROUP_SIZE - 1) / GROUP_SIZE;
    encoder->dispatchThreadgroups(MTL::Size(groupCount, 1, texture->arrayLength()), MTL::Size(GROUP_SIZE, 1, 1));
    encoder->insertDebugSignpost(NS::String::string("Postprocess", NS::StringEncoding::UTF8StringEncoding));
    encoder->popDebugGroup();
}

void Renderer::EncodePostprocessRenderPass(MTL::RenderCommandEncoder* encoder, uint32_t viewIdx)
{
    encoder->pushDebugGroup(NS::String::string("Postprocess Pass", NS::StringEncoding::UTF8StringEncoding));
    // encoder->setRenderPipelineState(m_PostprocessRenderPipeline->GetPipelineState());

    encoder->setFragmentTexture(m_Graphics->GetRenderTarget(), 0);
    encoder->setFragmentSamplerState(m_Graphics->GetOrCreateNearestSamplerState(), 0);

    m_Graphics->DrawMesh(encoder, m_QuadMesh.get());
    encoder->insertDebugSignpost(NS::String::string("Postprocess", NS::StringEncoding::UTF8StringEncoding));
    
    encoder->popDebugGroup();
}

void Renderer::EncodeMeshDepthRenderPass(MTL::RenderCommandEncoder* encoder, uint32_t viewIdx)
{
    encoder->pushDebugGroup(NS::String::string("Mesh Depth Pass", NS::StringEncoding::UTF8StringEncoding));

    uint32_t offsetBytes = 0u;
    Buffer* constantsBuffer = m_Graphics->AllocatePerFrameConstantsBuffer(sizeof(DepthOnlyConstants), offsetBytes);
    DepthOnlyConstants* depthOnlyConstants = (DepthOnlyConstants*)constantsBuffer->GetDataToModify(offsetBytes);
    depthOnlyConstants->_MVP[0] = AdjustProjectionMatrix(m_ProjMatrix[0]) * m_ViewMatrix[0] * m_ModelMatrix;
    int instanceCount = 1;
#if IS_STEREO_PIPELINE
    depthOnlyConstants->_MVP[1] = AdjustProjectionMatrix(m_ProjMatrix[1]) * m_ViewMatrix[1] * m_ModelMatrix;
    instanceCount *= 2;
#endif
    encoder->setVertexBuffer(constantsBuffer->GetHandle(), offsetBytes, 0);
    encoder->setVertexBuffer(m_SkinnedVertexBuffer->GetHandle(), 0, VERTEX_BUFFER_BINDING_OFFSET);
    encoder->drawIndexedPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle,
                                   m_Mesh->GetIndexCount(),
                                   MTL::IndexType::IndexTypeUInt32,
                                   m_Mesh->GetIndexBuffer()->GetHandle(),
                                   0,
                                   instanceCount);
    encoder->insertDebugSignpost(NS::String::string("Mesh Depth Pass", NS::StringEncoding::UTF8StringEncoding));
    
    encoder->popDebugGroup();
}

void Renderer::CreateAllPipelines()
{
    std::string stereoSuffix;
    std::string indirectSuffix;
#if IS_STEREO_PIPELINE
    stereoSuffix = "-Stereo";
#endif
#if ENABLE_INDIRECT
    indirectSuffix = "-Indirect";
#endif
    
    auto skinningShader = AssetLoader::LoadShader(m_pDevice, "shaders/Skinning.metal", "Skinning");
    m_SkinningPipeline = std::make_unique<ComputePipeline>(m_pDevice, std::move(skinningShader));
    
    auto backfaceCullShader = AssetLoader::LoadShader(m_pDevice, "shaders/CalcTriangleCullFlag" + stereoSuffix + ".metal", "CalcTriangleCullFlag");
    m_BackfaceCullPipeline = std::make_unique<ComputePipeline>(m_pDevice, std::move(backfaceCullShader));

    auto calcProjDataShader = AssetLoader::LoadShader(m_pDevice, "shaders/CalcProjData" + stereoSuffix + ".metal", "CalcProjData");
    m_CalcProjDataPipeline = std::make_unique<ComputePipeline>(m_pDevice, std::move(calcProjDataShader));

    auto prefixSumReduceShader = AssetLoader::LoadShader(m_pDevice, "shaders/PrefixSumReduce.metal", "PrefixSumReduce");
    m_PrefixSumReducePipeline = std::make_unique<ComputePipeline>(m_pDevice, std::move(prefixSumReduceShader));

    auto prefixSumScanShader = AssetLoader::LoadShader(m_pDevice, "shaders/PrefixSumScan.metal", "PrefixSumScan");
    m_PrefixSumScanPipeline = std::make_unique<ComputePipeline>(m_pDevice, std::move(prefixSumScanShader));

    auto prefixSumScanAddShader = AssetLoader::LoadShader(m_pDevice, "shaders/PrefixSumScanAdd.metal", "PrefixSumScanAdd");
    m_PrefixSumScanAddPipeline = std::make_unique<ComputePipeline>(m_pDevice, std::move(prefixSumScanAddShader));

    auto prefixSumScanAddInclusiveShader = AssetLoader::LoadShader(m_pDevice, "shaders/PrefixSumScanAddInclusive.metal", "PrefixSumScanAddInclusive");
    m_PrefixSumScanAddInclusivePipeline = std::make_unique<ComputePipeline>(m_pDevice, std::move(prefixSumScanAddInclusiveShader));

    auto genVisiblePointsShader = AssetLoader::LoadShader(m_pDevice, "shaders/GenVisblePointList.metal", "GenVisblePointList");
    m_GenVisiblePointsPipeline = std::make_unique<ComputePipeline>(m_pDevice, std::move(genVisiblePointsShader));

    auto genIndirectDrawArgsShader = AssetLoader::LoadShader(m_pDevice, "shaders/GenIndirectDrawArgs" + stereoSuffix + indirectSuffix + ".metal", "GenIndirectDrawArgs");
    m_GenIndirectDrawArgsPipeline = std::make_unique<ComputePipeline>(m_pDevice, std::move(genIndirectDrawArgsShader));

    auto calcDistancesShader = AssetLoader::LoadShader(m_pDevice, "shaders/CalcUint16Distances.metal", "CalcUint16Distances");
    m_CalcDistancesPipeline = std::make_unique<ComputePipeline>(m_pDevice, std::move(calcDistancesShader));

    auto sortCountShader = AssetLoader::LoadShader(m_pDevice, "shaders/FfxParallelSortCount.metal", "FfxParallelSortCount");
    m_SortCountPipeline = std::make_unique<ComputePipeline>(m_pDevice, std::move(sortCountShader));

    auto sortReduceShader = AssetLoader::LoadShader(m_pDevice, "shaders/FfxParallelSortReduce.metal", "FfxParallelSortReduce");
    m_SortReducePipeline = std::make_unique<ComputePipeline>(m_pDevice, std::move(sortReduceShader));

    auto sortScanShader = AssetLoader::LoadShader(m_pDevice, "shaders/FfxParallelSortScan.metal", "FfxParallelSortScan");
    m_SortScanPipeline = std::make_unique<ComputePipeline>(m_pDevice, std::move(sortScanShader));

    auto sortScanAddShader = AssetLoader::LoadShader(m_pDevice, "shaders/FfxParallelSortScanAdd.metal", "FfxParallelSortScanAdd");
    m_SortScanAddPipeline = std::make_unique<ComputePipeline>(m_pDevice, std::move(sortScanAddShader));

    auto sortScatterShader = AssetLoader::LoadShader(m_pDevice, "shaders/FfxParallelSortScatter.metal", "FfxParallelSortScatter");
    m_SortScatterPipeline = std::make_unique<ComputePipeline>(m_pDevice, std::move(sortScatterShader));

    auto gaussianRenderVertexShader = AssetLoader::LoadShader(m_pDevice, "shaders/RenderGaussianSplatsVert" + stereoSuffix + indirectSuffix + ".metal", "Vert");
    auto gaussianRenderFragmentShader = AssetLoader::LoadShader(m_pDevice, "shaders/RenderGaussianSplatsFrag.metal", "Frag");
    m_GaussianRenderPipeline = m_Graphics->CreateGaussianRenderPipeline(std::move(gaussianRenderVertexShader), std::move(gaussianRenderFragmentShader));

#ifdef PLATFORM_VISIONOS
    auto postprocessRenderVertexShader = AssetLoader::LoadShader(m_pDevice, "shaders/PostprocessVert" + stereoSuffix + ".metal", "Vert");
    auto postprocessRenderFragmentShader = AssetLoader::LoadShader(m_pDevice, "shaders/PostprocessFrag" + stereoSuffix + ".metal", "Frag");
    m_PostprocessRenderPipeline = m_Graphics->CreatePostprocessRenderPipeline(std::move(postprocessRenderVertexShader), std::move(postprocessRenderFragmentShader));

    auto depthRenderVertexShader = AssetLoader::LoadShader(m_pDevice, "shaders/DepthOnlyVert" + stereoSuffix + ".metal", "Vert");
    m_MeshDepthRenderPipeline = m_Graphics->CreteDepthRenderPipeline(std::move(depthRenderVertexShader));
#else
    auto postprocessShader = AssetLoader::LoadShader(m_pDevice, "shaders/Postprocess" + stereoSuffix + ".metal", "Postprocess");
    m_PostprocessComputePipeline = std::make_unique<ComputePipeline>(m_pDevice, std::move(postprocessShader));
#endif
}

void Renderer::LoadGaussianModel(const std::string& modelPath)
{
    const std::string fullPath = g_AssetBaseDir + modelPath;
    std::ifstream sceneDescFile(fullPath + "/scene_desc.json");
    if (!sceneDescFile.is_open())
    {
        LOG_ERROR("Failed to load scene desc file: %s", fullPath.c_str());
        return;
    }
    nlohmann::json sceneDesc = nlohmann::json::parse(sceneDescFile);
    
    const std::string& animationPath = sceneDesc["animation_paths"][0].get<std::string>();
    AssetLoader::LoadAnimationCurvesToPlayer(m_AnimationPlayer.get(), fullPath, animationPath);

    nlohmann::json rendererDesc = sceneDesc["renderer"];
    nlohmann::json meshDesc = rendererDesc["mesh_desc"];
    if (m_Mesh == nullptr)
        m_Mesh = std::make_unique<Mesh>(m_Graphics.get());
    // parse vertex layout
    std::vector<VertexLayout> vertexLayouts;
    for (const auto& layout : meshDesc["layouts"])
    {
        VertexAttributeType attributeType = (VertexAttributeType)layout["attribute"].get<uint32_t>();
        vertexLayouts.push_back({
            .attribute = attributeType,
            .offsetBytes = layout["offset"].get<uint32_t>(),
            .strideBytes = layout["stride"].get<uint32_t>(),
            .numElements = layout["num_elements"].get<uint32_t>(),
            .format = VertexLayout::ConvertToMTLVertexFormat(attributeType)
        });
    }
    m_Mesh->SetVertexLayout(std::move(vertexLayouts));
    const std::string& vertexBufferPath = meshDesc["mesh_vertex_buffer_path"].get<std::string>();
    const std::string& vertexBufferAbsPath = fullPath + "/" + vertexBufferPath;
    std::vector<uint8_t> vertexData = ReadBinaryFile(vertexBufferAbsPath);
    m_Mesh->SetVertexBufferData(vertexData.data(), (uint32_t)vertexData.size());
    
    const std::string& indexBufferPath = meshDesc["mesh_index_buffer_path"].get<std::string>();
    const std::string& indexBufferAbsPath = fullPath + "/" + indexBufferPath;
    std::vector<uint8_t> indexData = ReadBinaryFile(indexBufferAbsPath);
    m_Mesh->SetIndexBufferData(indexData.data(), (uint32_t)indexData.size());

    for (const auto& pose : rendererDesc["bind_poses"])
    {
        m_BindPoses.push_back(TryParseFloat4x4(pose));
    }
    for (const auto& boneTransform : rendererDesc["bone_transforms"])
    {
        m_BonePositions.push_back(TryParseFloat3(boneTransform["local_position"]));
        m_BoneQuaternions.push_back(TryParseFloat4(boneTransform["local_rotation"]));
    }
    uint32_t boneIdx = 0;
    for (const auto& boneName : rendererDesc["skeleton"])
    {
        m_BoneNameToIndex[boneName.get<std::string>()] = boneIdx++;
    }
    for (const auto& parentIdx : rendererDesc["skeleton_parent_id"])
    {
        m_BoneParentIndices.push_back(parentIdx.get<int>());
    }

    const nlohmann::json& blendShapeDesc = rendererDesc["blend_shape"];
    const std::string& blendShapePath = blendShapeDesc["blend_shape_path"].get<std::string>();
    const std::string& blendShapeAbsPath = fullPath + "/" + blendShapePath;
    std::vector<uint8_t> blendShapeData = ReadBinaryFile(blendShapeAbsPath);
    if (m_BlendShapeData == nullptr)
    {
        m_BlendShapeData = std::make_unique<BlendShapeData>();
    }
    auto& frames = m_BlendShapeData->frames;
    for (const auto& frameDesc : blendShapeDesc["blend_shape_buffer_desc"])
    {
        BlendShapeFrame frame;
        uint32_t offset = frameDesc["offset"].get<uint32_t>();
        uint32_t size = frameDesc["position_indice_size"].get<uint32_t>();
        uint32_t count = size / sizeof(uint32_t);
        frame.positionIndices.resize(count);
        std::memcpy(frame.positionIndices.data(), blendShapeData.data() + offset, size);
        
        offset += size;
        size = frameDesc["vertex_position_size"].get<uint32_t>();
        count = size / sizeof(float);
        frame.positionsOffsets.resize(count);
        std::memcpy(frame.positionsOffsets.data(), blendShapeData.data() + offset, size);
        frames.push_back(std::move(frame));
    }
    if (m_BlendShapeWeights.empty())
    {
        m_BlendShapeWeights.resize(frames.size(), 0.0f);
    }

    const std::string& gaussianPath = rendererDesc["gaussian_splat_data"].get<std::string>();

    m_GaussianAsset = AssetLoader::LoadGaussianAsset(this, fullPath, gaussianPath);
    
    auto readStringArrayFromJson = [](const nlohmann::json& json){
        std::vector<std::string> res;
        if (json.is_array())
        {
            for (const auto& element : json)
            {
                res.push_back(element.get<std::string>());
            }
        }
        return res;
    };
    
    if(rendererDesc.contains("neural_compensation"))
    {
        m_NeuralCompensation = std::make_unique<NeuralCompensation>(this);
        auto neuralCompJson = rendererDesc["neural_compensation"];
        if(neuralCompJson.contains("input_bone_names"))
        {
            m_NeuralCompensation->SetNetInputBoneNames(readStringArrayFromJson(neuralCompJson["input_bone_names"]));
        }
        if(neuralCompJson.contains("body_pose_deformer"))
        {
            auto posedeformer = neuralCompJson["body_pose_deformer"];
            
            std::string path = (fullPath.back() != '/' ? fullPath + '/' : fullPath) + posedeformer["path"].get<std::string>();
            std::ifstream file(path, std::ios::binary);
            std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
            
            MNN::BackendConfig backendConfig;
            backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_Normal;
            MNN::ScheduleConfig config;
            config.type = MNNForwardType::MNN_FORWARD_CPU;
            config.numThread = 4;
            config.backendConfig = &backendConfig;
            
            std::shared_ptr<MNN::Express::Executor::RuntimeManager> runtimeMgr(MNN::Express::Executor::RuntimeManager::createRuntimeManager(config), MNN::Express::Executor::RuntimeManager::destroy);
            
            auto _module = MNN::Express::Module::load(readStringArrayFromJson(posedeformer["inputs"]), readStringArrayFromJson(posedeformer["outputs"]), buffer.data(), buffer.size());
            
            m_NeuralCompensation->InitializeBodyPoseDeformModule(_module);
        }
        if(neuralCompJson.contains("pose_shadow_net"))
        {
            auto poseShadowNet = neuralCompJson["pose_shadow_net"];
            
            std::string path = (fullPath.back() != '/' ? fullPath + '/' : fullPath) + poseShadowNet["path"].get<std::string>();
            std::ifstream file(path, std::ios::binary);
            std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
            
            MNN::BackendConfig backendConfig;
            backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_Normal;
            MNN::ScheduleConfig config;
            config.type = MNNForwardType::MNN_FORWARD_CPU;
            config.numThread = 4;
            config.backendConfig = &backendConfig;
            
            std::shared_ptr<MNN::Express::Executor::RuntimeManager> runtimeMgr(MNN::Express::Executor::RuntimeManager::createRuntimeManager(config), MNN::Express::Executor::RuntimeManager::destroy);
            
            auto _module = MNN::Express::Module::load(readStringArrayFromJson(poseShadowNet["inputs"]), readStringArrayFromJson(poseShadowNet["outputs"]), buffer.data(), buffer.size());
            m_NeuralCompensation->InitializeBodyPoseShadowModule(_module);
        }
    }
    
    sceneDescFile.close();
}

}  // namespace hrm



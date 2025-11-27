#include "NeuralCompensation.hpp"
#include "Renderer.hpp"

#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>

#define NETWORK_INPUT_DIM 126

using namespace MNN::Express;

namespace hrm
{
simd::float3x3 MakeRotationMatrix(const simd::float4& q)
{
    float x = q.x * 2.0f;
    float y = q.y * 2.0f;
    float z = q.z * 2.0f;
    float xx = q.x * x;
    float yy = q.y * y;
    float zz = q.z * z;
    float xy = q.x * y;
    float xz = q.x * z;
    float yz = q.y * z;
    float wx = q.w * x;
    float wy = q.w * y;
    float wz = q.w * z;
    
    simd::float3x3 m;
    m.columns[0] = { 1.0f - (yy + zz),      xy + wz,      xz - wy };
    m.columns[1] = {      xy - wz,   1.0f - (xx + zz),      yz + wx };
    m.columns[2] = {      xz + wy,      yz - wx,   1.0f - (xx + yy) };
    return m;
}

simd::float4x4 MakeTransformMatrix(simd::float3 position, const simd::float4& q)
{
    simd::float4x4 m(simd::quatf{q.x, q.y, q.z, q.w});
    m.columns[3] = simd::make_float4(position, 1.0f);
    return m;
}

NeuralCompensation::~NeuralCompensation()
{
    if(m_BodyPoseDeformModule){
        MNN::Express::Module::destroy(m_BodyPoseDeformModule);
    }
    if(m_BodyPoseShadowModule){
        MNN::Express::Module::destroy(m_BodyPoseShadowModule);
    }
}

void NeuralCompensation::UpdateFrame()
{
    auto renderer = m_Renderer;
    if(renderer == nullptr)
        return;
    
    VARP input = MNN::Express::_Input({1, NETWORK_INPUT_DIM}, NCHW, halide_type_of<float>());
    float* inputPtr = input->writeMap<float>();
    assert(m_NetInputBoneNames.size() * 6 == NETWORK_INPUT_DIM);
    
    for (int i = 0; i < m_NetInputBoneNames.size(); i++)
    {
        int start = 6 * i;
        auto quat = renderer->GetBoneRotation(m_NetInputBoneNames[i]);
        auto rotationMat = MakeTransformMatrix(simd::make_float3(0, 0, 0),renderer->GetBoneRotation(m_NetInputBoneNames[i]));
        //transform rotation matrix to right handed in training period
        inputPtr[start] = rotationMat.columns[0][0];
        inputPtr[start + 1] = -rotationMat.columns[1][0];
        inputPtr[start + 2] = -rotationMat.columns[2][0];
        inputPtr[start + 3] = -rotationMat.columns[0][1];
        inputPtr[start + 4] = rotationMat.columns[1][1];
        inputPtr[start + 5] = rotationMat.columns[2][1];
    }
    
    if(m_BodyPoseDeformModule)
    {
        auto shapeOutput = m_BodyPoseDeformModule->onForward({input})[0];
        auto bsWeights = shapeOutput->readMap<float>();
        for(int i = 0; i < shapeOutput->getTensor()->elementSize(); i++)
        {
            renderer->SetBlendShapeWeight(i, std::max(std::min(bsWeights[i] * 100.0f, 100.0f), 0.0f));
        }
    }
    if(m_BodyPoseShadowModule)
    {
        auto shadowOutput = m_BodyPoseShadowModule->onForward({input})[0];
        auto shadowData = shadowOutput->readMap<float>();
        
        auto gpuBuffer = renderer->GetPoseShadowCompensationBuffer();
        renderer->GetPoseShadowCompensationBuffer()->SetData(reinterpret_cast<const uint8_t*>(shadowData) , shadowOutput->getTensor()->size());
    }
}
}

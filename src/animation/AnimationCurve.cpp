#include "AnimationCurve.hpp"

#include <algorithm>
#include <cassert>
#include <limits>

#include "Logging.hpp"

namespace
{
    float HermiteInterpolation(float v0, float dv0, float v1, float dv1, float t)
    {
        assert(std::abs(dv0) < 1e9f && std::abs(dv1) < 1e9f);
        float t2 = t * t;
        float t3 = t2 * t;
    
        float a = 2.0f * t3 - 3.0f * t2 + 1.0f;
        float b = t3 - 2.0f * t2 + t;
        float c = t3 - t2;
        float d = -2.0f * t3 + 3.0f * t2;
    
        return a * v0 + b * dv0 + c * dv1 + d * v1;
    }

    simd::float3 HermiteInterpolation(simd::float3 v0, simd::float3 dv0, simd::float3 v1, simd::float3 dv1, float t)
    {
        return simd_make_float3(
            HermiteInterpolation(v0.x, dv0.x, v1.x, dv1.x, t),
            HermiteInterpolation(v0.y, dv0.y, v1.y, dv1.y, t),
            HermiteInterpolation(v0.z, dv0.z, v1.z, dv1.z, t));
    }
    
    simd::float4 HermiteInterpolation(simd::float4 v0, simd::float4 dv0, simd::float4 v1, simd::float4 dv1, float t)
    {
        return simd_make_float4(
            HermiteInterpolation(v0.x, dv0.x, v1.x, dv1.x, t),
            HermiteInterpolation(v0.y, dv0.y, v1.y, dv1.y, t),
            HermiteInterpolation(v0.z, dv0.z, v1.z, dv1.z, t),
            HermiteInterpolation(v0.w, dv0.w, v1.w, dv1.w, t));
    }
}

namespace hrm
{

template<typename T>
AnimationCurve<T>::AnimationCurve(std::vector<KeyFrame<T>>&& keyFrames, const std::string& target, AttributeType attributeType)
: m_KeyFrames(std::move(keyFrames)), m_Target(target), m_AttributeType(attributeType)
{
    std::sort(m_KeyFrames.begin(), m_KeyFrames.end());
}

template<typename T>
void AnimationCurve<T>::SetKeyFrames(std::vector<KeyFrame<T>>&& keyFrames)
{
    m_KeyFrames = std::move(keyFrames);
    std::sort(m_KeyFrames.begin(), m_KeyFrames.end());
}

template<typename T>
void AnimationCurve<T>::SetKeyFrames(const std::vector<KeyFrame<T>>& keyFrames)
{
    m_KeyFrames = keyFrames;
    std::sort(m_KeyFrames.begin(), m_KeyFrames.end());
}

template<typename T>
const std::vector<KeyFrame<T>>& AnimationCurve<T>::GetKeyFrames() const
{
    return m_KeyFrames;
}

template<typename T>
std::vector<KeyFrame<T>>& AnimationCurve<T>::GetKeyFrames()
{
    return m_KeyFrames;
}

template<typename T>
void AnimationCurve<T>::SetInterpolationType(InterpolationType interpolationType)
{
    m_InterpolationType = interpolationType;
}

template<typename T>
InterpolationType AnimationCurve<T>::GetInterpolationType() const
{
    return m_InterpolationType;
}

template<typename T>
T AnimationCurve<T>::Evaluate(float time) const
{
    if (m_KeyFrames.empty())
    {
        LOG_ERROR("AnimationCurve is empty!");
        return 0.0f;
    }

    if (m_KeyFrames.size() == 1 || time <= m_KeyFrames[0].time)
    {
        return m_KeyFrames[0].value;
    }
    
    if (time >= m_KeyFrames.back().time)
    {
        return m_KeyFrames.back().value;
    }

    // binary search for the left and right key frames
    // assert l < t <= r
    uint32_t lIdx = 0, rIdx = (uint32_t)m_KeyFrames.size() - 1;
    while (lIdx + 1 < rIdx)
    {
        uint32_t mIdx = (lIdx + rIdx) / 2;
        if (m_KeyFrames[mIdx].time < time)
            lIdx = mIdx;
        else
            rIdx = mIdx;
    }
    
    const KeyFrame<T>& lKeyFrame = m_KeyFrames[lIdx];
    const KeyFrame<T>& rKeyFrame = m_KeyFrames[rIdx];
    float dt = rKeyFrame.time - lKeyFrame.time;
    if (dt < std::numeric_limits<float>::epsilon())
    {
        return lKeyFrame.value;
    }
    
    float t = (time - lKeyFrame.time) / dt;

    switch (m_InterpolationType)
    {
    case InterpolationType::Linear:
        return lKeyFrame.value + t * (rKeyFrame.value - lKeyFrame.value);
    case InterpolationType::Hermite:
    {
        T dl = lKeyFrame.outTangent * dt;
        T dr = rKeyFrame.inTangent * dt;
        return HermiteInterpolation(lKeyFrame.value, dl, rKeyFrame.value, dr, t);
    }
    default:
        LOG_ERROR("Unsupported interpolation type: %d", static_cast<int>(m_InterpolationType));
    }
    
    return T{};
}

template class AnimationCurve<float>;
template class AnimationCurve<simd::float3>;
template class AnimationCurve<simd::float4>;
} // namespace hrm

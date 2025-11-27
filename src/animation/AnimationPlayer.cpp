#include "AnimationPlayer.hpp"

#include "Logging.hpp"
#include "Renderer.hpp"

namespace hrm
{

void AnimationPlayer::AddAnimationCurves(std::vector<AnimationCurve<float>>&& animationCurves)
{
    for (const auto& animationCurve : animationCurves)
    {
        m_Duration = std::max(m_Duration, animationCurve.GetDuration());
    }
    m_FloatCurves.insert(m_FloatCurves.end(), animationCurves.begin(), animationCurves.end());
}

void AnimationPlayer::AddAnimationCurves(std::vector<AnimationCurve<simd::float3>>&& animationCurves)
{
    for (const auto& animationCurve : animationCurves)
    {
        m_Duration = std::max(m_Duration, animationCurve.GetDuration());
    }
    m_Float3Curves.insert(m_Float3Curves.end(), animationCurves.begin(), animationCurves.end());
}

void AnimationPlayer::AddAnimationCurves(std::vector<AnimationCurve<simd::float4>>&& animationCurves)
{
    for (const auto& animationCurve : animationCurves)
    {
        m_Duration = std::max(m_Duration, animationCurve.GetDuration());
    }
    m_Float4Curves.insert(m_Float4Curves.end(), animationCurves.begin(), animationCurves.end());
}

void AnimationPlayer::AddAnimationCurve(AnimationCurve<float>&& animationCurve)
{
    m_Duration = std::max(m_Duration, animationCurve.GetDuration());
    m_FloatCurves.push_back(std::move(animationCurve));
}

void AnimationPlayer::AddAnimationCurve(AnimationCurve<simd::float3>&& animationCurve)
{
    m_Duration = std::max(m_Duration, animationCurve.GetDuration());
    m_Float3Curves.push_back(std::move(animationCurve));
}

void AnimationPlayer::AddAnimationCurve(AnimationCurve<simd::float4>&& animationCurve)
{
    m_Duration = std::max(m_Duration, animationCurve.GetDuration());
    m_Float4Curves.push_back(std::move(animationCurve));
}

void AnimationPlayer::Stop()
{
    m_IsPlaying = false;
    m_CurrentTime = 0.0f;
}

void AnimationPlayer::Update(float deltaTime)
{
    if (!m_IsPlaying)
        return;

    m_CurrentTime += deltaTime * m_PlaySpeed;
    if (m_CurrentTime >= m_Duration)
    {
        switch (m_PlayMode)
        {
        case PlayMode::Once:
            m_IsPlaying = false;
            return;
        case PlayMode::Loop:
            m_CurrentTime = std::fmod(m_CurrentTime, m_Duration);
            break;
        default:
            LOG_ERROR("Unsupported play mode: %d", static_cast<int>(m_PlayMode));
            return;
        }
    }

    for (const auto& animationCurve : m_FloatCurves)
    {
        float value = animationCurve.Evaluate(m_CurrentTime);
        LOG_WARNING("Float animation curve is not implemented yet, there are no float targets actually");
    }
    for (const auto& animationCurve : m_Float3Curves)
    {
        simd::float3 value = animationCurve.Evaluate(m_CurrentTime);
        assert(m_pRenderer != nullptr);
        switch (animationCurve.GetAttributeType())
        {
        case AttributeType::Position:
            m_pRenderer->SetBonePosition(animationCurve.GetTarget(), value);
            break;
        default:
            LOG_ERROR("Unsupported attribute type: %d", static_cast<int>(animationCurve.GetAttributeType()));
            break;
        }
    }
    for (const auto& animationCurve : m_Float4Curves)
    {
        simd::float4 value = animationCurve.Evaluate(m_CurrentTime);
        assert(m_pRenderer != nullptr);
        switch (animationCurve.GetAttributeType())
        {
        case AttributeType::Rotation:
            m_pRenderer->SetBoneRotation(animationCurve.GetTarget(), value);
            break;
        default:
            LOG_ERROR("Unsupported attribute type: %d", static_cast<int>(animationCurve.GetAttributeType()));
            break;
        }
    }
}

} // namespace hrm

#pragma once

#include <vector>
#include <memory>

#include "AnimationCurve.hpp"

namespace hrm
{

class Renderer;

enum class PlayMode : uint8_t
{
    Once,
    Loop,
    Count
};

class AnimationPlayer
{
public:
    AnimationPlayer(Renderer* renderer) : m_pRenderer(renderer) {}
    ~AnimationPlayer() = default;

    void AddAnimationCurves(std::vector<AnimationCurve<float>>&& animationCurves);
    void AddAnimationCurves(std::vector<AnimationCurve<simd::float3>>&& animationCurves);
    void AddAnimationCurves(std::vector<AnimationCurve<simd::float4>>&& animationCurves);

    void AddAnimationCurve(AnimationCurve<float>&& animationCurve);
    void AddAnimationCurve(AnimationCurve<simd::float3>&& animationCurve);
    void AddAnimationCurve(AnimationCurve<simd::float4>&& animationCurve);

    void SetPlayMode(PlayMode playMode) { m_PlayMode = playMode; }
    PlayMode GetPlayMode() const { return m_PlayMode; }

    void SetPlaySpeed(float playSpeed) { m_PlaySpeed = playSpeed; }
    float GetPlaySpeed() const { return m_PlaySpeed; }

    float GetDuration() const { return m_Duration; }

    void Play() { m_IsPlaying = true; }
    void Pause() { m_IsPlaying = false; }
    void Stop();

    void Update(float deltaTime);

    void SetRenderer(Renderer* renderer) { m_pRenderer = renderer; }

private:
    std::vector<AnimationCurve<float>> m_FloatCurves;
    std::vector<AnimationCurve<simd::float3>> m_Float3Curves;
    std::vector<AnimationCurve<simd::float4>> m_Float4Curves;

    bool m_IsPlaying { false };
    PlayMode m_PlayMode { PlayMode::Loop };
    float m_Duration { 0.0f };
    float m_CurrentTime { 0.0f };
    float m_PlaySpeed { 1.0f };

    // use raw pointer instead of shared pointer to avoid circular dependency
    Renderer* m_pRenderer { nullptr };
};

} // namespace hrm

#pragma once

#include <vector>
#include <string>
#include <simd/simd.h>

namespace hrm
{

template<typename T>
struct KeyFrame
{
    float time { 0.0f };
    
    T value {};
    T inTangent {};
    T outTangent {};
    
    friend bool operator < (const KeyFrame& lhs, const KeyFrame& rhs)
    {
        return lhs.time < rhs.time;
    }
};

enum class InterpolationType : uint8_t
{
    Linear,
    Hermite,
    Count
};

enum class AttributeType : uint8_t
{
    Position,
    Rotation,
    BlendShapeWeight,
    Count
};

template<typename T>
class AnimationCurve
{
public:
    AnimationCurve() = default;
    AnimationCurve(std::vector<KeyFrame<T>>&& keyFrames, const std::string& target, AttributeType attributeType);
    ~AnimationCurve() = default;

    void SetTarget(const std::string& target) { m_Target = target; }
    const std::string& GetTarget() const { return m_Target; }

    void SetKeyFrames(std::vector<KeyFrame<T>>&& keyFrames);
    void SetKeyFrames(const std::vector<KeyFrame<T>>& keyFrames);
    const std::vector<KeyFrame<T>>& GetKeyFrames() const;
    std::vector<KeyFrame<T>>& GetKeyFrames();

    void SetInterpolationType(InterpolationType interpolationType);
    InterpolationType GetInterpolationType() const;

    void SetAttributeType(AttributeType attributeType) { m_AttributeType = attributeType; }
    AttributeType GetAttributeType() const { return m_AttributeType; }

    float GetDuration() const { return m_KeyFrames.empty() ? 0.0f : m_KeyFrames.back().time; }

    T Evaluate(float time) const;
    
private:
    std::vector<KeyFrame<T>> m_KeyFrames;
    InterpolationType m_InterpolationType { InterpolationType::Hermite };

    std::string   m_Target;  // bone's name actually
    AttributeType m_AttributeType { AttributeType::Count };
};
    
}

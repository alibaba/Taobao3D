#pragma once

#include <chrono>
#include "Logging.hpp"

#define DEBUG_STATS 1

namespace hrm
{

class Timer
{
public:
    Timer()
    : m_LastFrameTimePoint(std::chrono::steady_clock::now())
    {
    }
    
    void Update()
    {
        auto currentTimePoint = std::chrono::steady_clock::now();
        m_DeltaTime = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTimePoint - m_LastFrameTimePoint).count() / 1e9;
        m_LastFrameTimePoint = currentTimePoint;
        m_ElapsedTime += m_DeltaTime;
        
        m_FrameCount++;
        m_StatsTime += m_DeltaTime;

        if (m_StatsTime >= 1.0f)
        {
            m_FPS = m_FrameCount;
            m_FrameCount = 0;
            m_StatsTime = 0.0f;
#if DEBUG_STATS
            LOG_INFO("current fps: %.1f", m_FPS);
#endif
        }
    }
    
    float GetDeltaTime() const { return m_DeltaTime; }
    float GetElapsedTime() const { return m_ElapsedTime; }
    float GetFPS() const { return m_FPS; }
    
private:
    std::chrono::steady_clock::time_point m_LastFrameTimePoint;
    
    // statistical data
    float m_DeltaTime { 0.0f };
    float m_ElapsedTime { 0.0f };
    
    float m_FPS { 0.0f };
    float m_StatsTime { 0.0f };
    uint32_t m_FrameCount { 0u };
};

}

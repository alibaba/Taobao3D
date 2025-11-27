#pragma once

#include <vector>

#include <simd/simd.h>

namespace hrm
{

struct BlendShapeFrame
{
    float weight { 100.0f };

    // must be sparse blendshape and only modify position
    std::vector<uint32_t> positionIndices;
    // NOTE: simd::float3 have the same size and alignment as simd_float4.
    std::vector<float>    positionsOffsets;  // 3 offsets per index
};

struct BlendShapeData
{
    // every channel must has only one frame
    std::vector<BlendShapeFrame> frames;
};

} // namespace hrm

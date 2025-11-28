#pragma once

#include <TargetConditionals.h>
#include <simd/simd.h>

#define ENABLE_INDIRECT (!TARGET_OS_SIMULATOR)

#ifdef PLATFORM_VISIONOS
    #define IS_STEREO_PIPELINE (!TARGET_OS_SIMULATOR)
#else
    #define IS_STEREO_PIPELINE 0
#endif

#define GROUP_SIZE 256
#define MAX_BONES 200

// prefix sum argments
#define ELEMENT_NUM_PER_THREAD 4
#define THREAD_NUM_PER_GROUP 256
#define BLOCK_SIZE (THREAD_NUM_PER_GROUP * ELEMENT_NUM_PER_THREAD)


#define FFX_PARALLELSORT_ELEMENTS_PER_THREAD 4
#define FFX_PARALLELSORT_THREADGROUP_SIZE 128
#define FFX_PARALLELSORT_SORT_BITS_PER_PASS 4
#define FFX_PARALLELSORT_SORT_BIN_COUNT (1u << FFX_PARALLELSORT_SORT_BITS_PER_PASS)
#define FFX_PARALLELSORT_MAX_THREADGROUPS_TO_RUN 800
#define SORT_BLOCK_SIZE (FFX_PARALLELSORT_ELEMENTS_PER_THREAD * FFX_PARALLELSORT_THREADGROUP_SIZE)

namespace hrm
{

struct SkinningConstants
{
    simd::float4 _MeshLayout;
    simd::float4 _BonePositions[MAX_BONES];
    simd::float4 _BoneRotations[MAX_BONES];
};

struct BackfaceCullConstants
{
    simd::float4x4 _MatrixWorldToObject;
    simd::float4 _CameraPosWS[2];
    simd::float4 _MeshLayout;  // x: vertex stride, y: triangle count
};

struct CalcProjDataConstants
{
    simd::float4x4 _MatrixWorldToObject;
    simd::float4x4 _MatrixM;
    simd::float4x4 _MatrixV[2];
    simd::float4x4 _MatrixP[2];
    simd::float4 _CameraPosWS[2];
    simd::float4 _SplatProp;  // x: splat count, y: splat format, z: sh order, w: vertex stride
    simd::float4 _VecScreenParams; // x: screen width
};

// NOTE: only use sizeof SplatProjData, it's should be same as in shader
struct SplatProjData
{
    simd::float4 pos;
    simd::float4 color;
//    simd::float2 axis1;
//    simd::float2 axis2;
};

struct PrefixSumReduceConstants
{
    simd::float4 _Count;  // x: gaussian splat count
};

struct PrefixSumScanConstants
{
    simd::float4 _Count;  // x: gaussian splat count
};

struct PrefixSumScanAddConstants
{
    simd::float4 _Count;  // x: gaussian splat count
};

struct PrefixSumScanAddInclusiveConstants
{
    simd::float4 _Count;  // x: gaussian splat count
};

struct GenVisiblePointsConstants
{
    simd::float4 _Count;  // x: gaussian splat count
};

struct GenIndirectDrawArgsConstants
{
    simd::float4 _PointCount;  // x: gaussian splat count
};

struct SortCountConstants
{
    simd::float4 _Packed_Params1;  // uint numReduceThreadgroupPerBin, uint numScanValues, uint shift, uint padding
};

struct SortScatterConstants
{
    simd::float4 _Packed_Params1;  // uint numReduceThreadgroupPerBin, uint numScanValues, uint shift, uint padding
};

struct RenderGaussianConstants
{
    simd::float4 _CustomScreenParams;  // xy: screen size
};

struct PostprocessConstants
{
    simd::float4 _Params;  // x: texture width, y: texture height, z: pixel count, w: isFlipY
};

struct DepthOnlyConstants
{
    simd::float4x4 _MVP[2]; // model-view-projection matrix
};

//-----------------------helper functions-----------------------//
union IntFloatUnion
{
    int intValue;
    float floatValue;
};

inline float AsFloat(int intValue)
{
    IntFloatUnion unionValue;
    unionValue.intValue = intValue;
    return unionValue.floatValue;
}

inline simd::float4 AsFloat4(int val0, int val1 = 0, int val2 = 0, int val3 = 0)
{
    return simd_make_float4(AsFloat(val0), AsFloat(val1), AsFloat(val2), AsFloat(val3));
}

}

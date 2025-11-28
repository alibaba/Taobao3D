#include "common.hlsl"

#define FFX_PARALLELSORT_ELEMENTS_PER_THREAD 4
#define FFX_PARALLELSORT_THREADGROUP_SIZE 128
#define FFX_PARALLELSORT_SORT_BIN_COUNT 16
#define FFX_PARALLELSORT_MAX_THREADGROUPS_TO_RUN 800

cbuffer Constants : register(b0)
{
    float4 _PointCount;  // x: splat count
};

StructuredBuffer<uint> _VisibleMaskPrefixSums : register(t1);

RWStructuredBuffer<uint> _VisiblePointList : register(u2); // visble points to be rendered
RWStructuredBuffer<uint> _IndirectDrawArgs : register(u3); // contains 5 uint: index count per instance, instance count, start index, base vertex, base instance
RWStructuredBuffer<uint> _SortParams : register(u4); // uint[8] type
#ifdef ENABLE_INDIRECT
RWStructuredBuffer<uint> _SortSumPassIndirectBuffer : register(u5); // uint[3] type, corresponding to thread group size xyz
RWStructuredBuffer<uint> _SortReducePassIndirectBuffer : register(u6); // uint[3] type, corresponding to thread group size xyz
RWStructuredBuffer<uint> _SortCopyPassIndirectBuffer : register(u7); // uint[3] type, corresponding to thread group size xyz
#endif

uint GetPointCount() { return asuint(_PointCount.x); }

void FillSortArgs(uint pointCount)
{
	// compute sort params
	uint blockSize = FFX_PARALLELSORT_ELEMENTS_PER_THREAD * FFX_PARALLELSORT_THREADGROUP_SIZE;
	uint numBlocks = (pointCount + blockSize - 1) / blockSize;
	uint numThreadGroupsToRun = FFX_PARALLELSORT_MAX_THREADGROUPS_TO_RUN;
	uint numBlocksPerThreadGroup = numBlocks / numThreadGroupsToRun;
	uint numThreadGroupsWithAdditionalBlocks = numBlocks % numThreadGroupsToRun;
	if (numBlocks < numThreadGroupsToRun)
	{
		numBlocksPerThreadGroup = 1;
		numThreadGroupsToRun = numBlocks;
		numThreadGroupsWithAdditionalBlocks = 0;
	}

	uint numReduceThreadgroupPerBin = (numThreadGroupsToRun + blockSize - 1) / blockSize;
	uint numReducedThreadGroupsToRun = FFX_PARALLELSORT_SORT_BIN_COUNT * (numReduceThreadgroupPerBin);

	_SortParams[0] = pointCount;
	_SortParams[1] = numBlocksPerThreadGroup;
	_SortParams[2] = numThreadGroupsToRun;
	_SortParams[3] = numThreadGroupsWithAdditionalBlocks;
	_SortParams[4] = numReduceThreadgroupPerBin;
	_SortParams[5] = numReducedThreadGroupsToRun; // numScanValues
	_SortParams[6] = 0; // unused
	_SortParams[7] = 0; // unused

#ifdef ENABLE_INDIRECT
	_SortSumPassIndirectBuffer[0] = numThreadGroupsToRun;
	_SortSumPassIndirectBuffer[1] = 1;
	_SortSumPassIndirectBuffer[2] = 1;

	_SortReducePassIndirectBuffer[0] = numReduceThreadgroupPerBin * FFX_PARALLELSORT_SORT_BIN_COUNT;
	_SortReducePassIndirectBuffer[1] = 1;
	_SortReducePassIndirectBuffer[2] = 1;

	_SortCopyPassIndirectBuffer[0] = (pointCount + 255) / 256;
	_SortCopyPassIndirectBuffer[1] = 1;
	_SortCopyPassIndirectBuffer[2] = 1;
#endif
}

[numthreads(GROUP_SIZE, 1, 1)]
void GenVisblePointList(uint3 id : SV_DispatchThreadID)
{
	uint idx = id.x;
	if (idx >= GetPointCount())
		return;

	uint curSum = _VisibleMaskPrefixSums[idx];
	if (idx == 0)
	{
		if (curSum == 1)
		{
			_VisiblePointList[0] = 0; // first point is visible
		}
	}
	else
	{
		uint preSum = _VisibleMaskPrefixSums[idx - 1];
		if (preSum != curSum)
		{
			_VisiblePointList[curSum - 1] = idx;
		}
	}
}

[numthreads(1, 1, 1)]
void GenIndirectDrawArgs(uint3 id : SV_DispatchThreadID)
{
	uint visiblePointCount = _VisibleMaskPrefixSums[GetPointCount() - 1];
	_IndirectDrawArgs[0] = 6; // 6 indices per instance (quad)
#ifdef IS_STEREO_PIPELINE
	_IndirectDrawArgs[1] = visiblePointCount * 2;
#else
	_IndirectDrawArgs[1] = visiblePointCount;
#endif
	_IndirectDrawArgs[2] = 0;
	_IndirectDrawArgs[3] = 0;
	_IndirectDrawArgs[4] = 0;

	// set padding values zero
	_IndirectDrawArgs[5] = 0;
	_IndirectDrawArgs[6] = 0;
	_IndirectDrawArgs[7] = 0;

	// compute sort params
	FillSortArgs(visiblePointCount);
}
#define ELEMENT_NUM_PER_THREAD 4
#define THREAD_NUM_PER_GROUP 256
#define BLOCK_SIZE (THREAD_NUM_PER_GROUP * ELEMENT_NUM_PER_THREAD)

cbuffer Constants : register(b0)
{
    float4 _Count;  // x: gaussian splat count
};

StructuredBuffer<uint> _ReduceSrcValues : register(t1);
StructuredBuffer<uint> _ScanSrcValues : register(t2);
StructuredBuffer<uint> _ScanAddSrcValues : register(t3);

RWStructuredBuffer<uint> _ReduceDstValues : register(u4);
RWStructuredBuffer<uint> _ScanDstValues : register(u5);

uint GetCount() { return asuint(_Count.x); }

groupshared uint gs_TempSums[BLOCK_SIZE];

[numthreads(THREAD_NUM_PER_GROUP, 1, 1)]
void PrefixSumReduce(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
	// compute local sum (sum of elements processed by current thread)
	uint localSum = 0;
	uint baseIndex = BLOCK_SIZE * groupID;
	for (uint i = 0; i < ELEMENT_NUM_PER_THREAD; ++i)
	{
		uint localIndex = localID * ELEMENT_NUM_PER_THREAD + i;
		uint globalIndex = baseIndex + localIndex;
		localSum += (globalIndex < GetCount() ? _ReduceSrcValues[globalIndex] : 0);
	}

	// compute wave sum with local sum
	uint waveSum = WaveActiveSum(localSum);
	uint waveID = localID / WaveGetLaneCount();
	if (WaveIsFirstLane())
	{
		gs_TempSums[waveID] = waveSum;
	}

	// wait all waves in group
	GroupMemoryBarrierWithGroupSync();

	// compute group sum with wave sum
	if (!waveID)
	{
		// NOTE THREAD_NUM_PER_GROUP must be less than laneCount * laneCount, or groupSum is wrong
		uint blockSum = WaveActiveSum(localID < (THREAD_NUM_PER_GROUP / WaveGetLaneCount()) ? gs_TempSums[localID] : 0);
		if (localID == 0)
		{
			_ReduceDstValues[groupID] = blockSum;
		}
	}
}

groupshared uint gs_Values[BLOCK_SIZE];

void ComputeGroupPrefixSum(uint localID, uint groupID, uint elementCount, bool scanAdd, bool inclusivePrefixSum)
{
	// load element of current group to LDS
	uint baseIndex = BLOCK_SIZE * groupID;
	uint srcValues[ELEMENT_NUM_PER_THREAD];
	for (uint i = 0; i < ELEMENT_NUM_PER_THREAD; ++i)
	{
		uint localIndex = localID * ELEMENT_NUM_PER_THREAD + i;
		uint globalIndex = baseIndex + localIndex;
		uint value = globalIndex < elementCount ? _ScanSrcValues[globalIndex] : 0;
		srcValues[i] = value;
		gs_Values[localIndex] = value;
	}

	GroupMemoryBarrierWithGroupSync();

	// compute localSum (exclusive prefix sum of elements processed by current thread)
	uint localSum = 0;
	for (i = 0; i < ELEMENT_NUM_PER_THREAD; ++i)
	{
		uint localIndex = localID * ELEMENT_NUM_PER_THREAD + i;
		uint value = gs_Values[localIndex];
		gs_Values[localIndex] = localSum;
		localSum += value;
	}

	// compute prefix sum in wave
	uint waveSum = WavePrefixSum(localSum);
	uint waveID = localID / WaveGetLaneCount();
	uint laneID = WaveGetLaneIndex();
	// last element in wave store waveSum to LDS
	if (laneID == WaveGetLaneCount() - 1)
	{
		// WavePrefixSum returns exclusive prefix sum, so add 'localSum' to make it inclusive prefix sum
		gs_TempSums[waveID] = waveSum + localSum;
	}

	GroupMemoryBarrierWithGroupSync();

	// compute prefix sum for waveSums in LDS
	// waveCount must be less than sizeof(gs_TempSums)
	uint waveCount = THREAD_NUM_PER_GROUP / WaveGetLaneCount();
	if (localID < waveCount)
	{
		gs_TempSums[localID] = WavePrefixSum(gs_TempSums[localID]);
	}

	GroupMemoryBarrierWithGroupSync();

	uint groupSum = gs_TempSums[waveID] + waveSum;
	if (scanAdd)
	{
		// add prefix sum from up-level group
		groupSum += _ScanAddSrcValues[groupID];
	}

	// write back prefix sums in group scope to dst buffer
	for (i = 0; i < ELEMENT_NUM_PER_THREAD; ++i)
	{
		uint localIndex = localID * ELEMENT_NUM_PER_THREAD + i;
		uint globalIndex = baseIndex + localIndex;
		if (globalIndex < elementCount)
		{
			uint prefixSumInGroup = gs_Values[localIndex] + groupSum + (inclusivePrefixSum ? srcValues[i] : 0);
			_ScanDstValues[globalIndex] = prefixSumInGroup;
		}
	}
}

[numthreads(THREAD_NUM_PER_GROUP, 1, 1)]
void PrefixSumScan(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
	ComputeGroupPrefixSum(localID, groupID, GetCount(), false, false);
}

[numthreads(THREAD_NUM_PER_GROUP, 1, 1)]
void PrefixSumScanAdd(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
	ComputeGroupPrefixSum(localID, groupID, GetCount(), true, false);
}

[numthreads(THREAD_NUM_PER_GROUP, 1, 1)]
void PrefixSumScanAddInclusive(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
	ComputeGroupPrefixSum(localID, groupID, GetCount(), true, true);
}

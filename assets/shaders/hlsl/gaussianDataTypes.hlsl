#ifndef GAUSSIAN_DATA_TYPE
#define GAUSSIAN_DATA_TYPE

#define CHUNK_SIZE 256
#define TEXTURE_WIDTH 2048

#define VECTOR_FMT_32F 0
#define VECTOR_FMT_16 1
#define VECTOR_FMT_11 2
#define VECTOR_FMT_6 3

static const float SH_C1 = 0.4886025;
static const float SH_C2[] = { 1.0925484, -1.0925484, 0.3153916, -1.0925484, 0.5462742 };
static const float SH_C3[] = { -0.5900436, 2.8906114, -0.4570458, 0.3731763, -0.4570458, 1.4453057, -0.5900436 };

struct TriangleTransform
{
	float3x3 rotation;
    float3x3 jacobian;
	float3 origin;

	float scale_factor;
    float face_shadow;
};

struct SplatProjData
{
    float4 pos;
    half4 color;
    half2 axis1;
    half2 axis2;
};

struct SplatSHData
{
    half3 col, sh1, sh2, sh3, sh4, sh5, sh6, sh7, sh8, sh9, sh10, sh11, sh12, sh13, sh14, sh15;
};

struct SplatData
{
    float3 pos;
    float4 rot;
    float3 scale;
    half opacity;
    SplatSHData sh;
};

struct SplatChunkInfo
{
    uint colR, colG, colB, colA;
    float2 posX, posY, posZ;
    uint sclX, sclY, sclZ;
    uint shR, shG, shB;
};

#endif
#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct type_Constants
{
    float4x4 _MatrixWorldToObject;
    float4x4 _MatrixM;
    float4x4 _MatrixV[2];
    float4x4 _MatrixP[2];
    float4 _CameraPosWS[2];
    float4 _SplatProp;
    float4 _VecScreenParams;
};

struct type_StructuredBuffer_int
{
    int _m0[1];
};

struct type_ByteAddressBuffer
{
    uint _m0[1];
};

struct SplatChunkInfo
{
    uint colR;
    uint colG;
    uint colB;
    uint colA;
    float2 posX;
    float2 posY;
    float2 posZ;
    uint sclX;
    uint sclY;
    uint sclZ;
    uint shR;
    uint shG;
    uint shB;
};

struct type_StructuredBuffer_SplatChunkInfo
{
    SplatChunkInfo _m0[1];
};

struct type_StructuredBuffer_float
{
    float _m0[1];
};

struct type_RWStructuredBuffer_uint
{
    uint _m0[1];
};

struct type_RWStructuredBuffer_float
{
    float _m0[1];
};

struct SplatProjData
{
    float4 pos;
    half4 color;
    half2 axis1;
    half2 axis2;
};

struct type_RWStructuredBuffer_SplatProjData
{
    SplatProjData _m0[1];
};

constant float4 _3396 = {};
constant half4 _3397 = {};

kernel void CalcProjData(constant type_Constants& Constants [[buffer(0)]], const device type_StructuredBuffer_int& _GSIndice [[buffer(1)]], const device type_StructuredBuffer_int& _TriangleCullFlag [[buffer(2)]], const device type_ByteAddressBuffer& _SplatPropData [[buffer(3)]], const device type_StructuredBuffer_SplatChunkInfo& _SplatChunks [[buffer(4)]], const device type_ByteAddressBuffer& _SplatPos [[buffer(5)]], const device type_ByteAddressBuffer& _SplatOther [[buffer(6)]], const device type_ByteAddressBuffer& _SplatSH [[buffer(7)]], const device type_StructuredBuffer_int& _MeshIndice [[buffer(8)]], const device type_ByteAddressBuffer& _MeshVertice [[buffer(9)]], const device type_StructuredBuffer_float& _PoseShadowCompensation [[buffer(10)]], device type_RWStructuredBuffer_uint& _VisibleMasks [[buffer(11)]], device type_RWStructuredBuffer_float& _PointDistances [[buffer(12)]], device type_RWStructuredBuffer_SplatProjData& _SplatProjData0 [[buffer(13)]], texture2d<float> _SplatColor [[texture(0)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    do
    {
        if (gl_GlobalInvocationID.x >= as_type<uint>(Constants._SplatProp.x))
        {
            break;
        }
        int _270 = _GSIndice._m0[gl_GlobalInvocationID.x];
        uint _282 = (_SplatPropData._m0[(gl_GlobalInvocationID.x & 4294967292u) >> 2u] >> ((8u * (gl_GlobalInvocationID.x % 4u)) & 31u)) & 255u;
        if ((_282 != 2u) && (_282 != uint(_TriangleCullFlag._m0[uint(_270)] + 1)))
        {
            _VisibleMasks._m0[gl_GlobalInvocationID.x] = 0u;
            _SplatProjData0._m0[gl_GlobalInvocationID.x] = SplatProjData{ float4(0.0), half4(half(0.0)), half2(half(0.0)), half2(half(0.0)) };
            break;
        }
        uint _292 = gl_GlobalInvocationID.x / 256u;
        SplatChunkInfo _294 = _SplatChunks._m0[_292];
        uint _315 = as_type<uint>(Constants._SplatProp.y);
        uint _316 = _315 & 255u;
        bool _317 = _316 == 0u;
        uint _335;
        if (_317)
        {
            _335 = 12u;
        }
        else
        {
            uint _334;
            if (_316 == 1u)
            {
                _334 = 6u;
            }
            else
            {
                uint _333;
                if (_316 == 2u)
                {
                    _333 = 4u;
                }
                else
                {
                    _333 = (_316 == 3u) ? 2u : 0u;
                }
                _334 = _333;
            }
            _335 = _334;
        }
        uint _336 = gl_GlobalInvocationID.x * _335;
        uint _337 = _336 & 4294967292u;
        uint _338 = _337 >> 2u;
        float3 _474;
        if (_317)
        {
            uint _346 = (_337 + 4u) >> 2u;
            uint _350 = (_337 + 8u) >> 2u;
            uint _372;
            uint _373;
            uint _374;
            if (_336 != _337)
            {
                _372 = (_SplatPos._m0[_350] >> 16u) | ((_SplatPos._m0[(_337 + 12u) >> 2u] & 65535u) << 16u);
                _373 = (_SplatPos._m0[_346] >> 16u) | ((_SplatPos._m0[_350] & 65535u) << 16u);
                _374 = (_SplatPos._m0[_338] >> 16u) | ((_SplatPos._m0[_346] & 65535u) << 16u);
            }
            else
            {
                _372 = _SplatPos._m0[_350];
                _373 = _SplatPos._m0[_346];
                _374 = _SplatPos._m0[_338];
            }
            _474 = float3(as_type<float>(_374), as_type<float>(_373), as_type<float>(_372));
        }
        else
        {
            float3 _473;
            if (_316 == 1u)
            {
                uint _384 = (_337 + 4u) >> 2u;
                uint _395;
                uint _396;
                if (_336 != _337)
                {
                    _395 = _SplatPos._m0[_384] >> 16u;
                    _396 = (_SplatPos._m0[_338] >> 16u) | ((_SplatPos._m0[_384] & 65535u) << 16u);
                }
                else
                {
                    _395 = _SplatPos._m0[_384];
                    _396 = _SplatPos._m0[_338];
                }
                _473 = float3(float(_396 & 65535u) * 1.525902189314365386962890625e-05, float((_396 >> 16u) & 65535u) * 1.525902189314365386962890625e-05, float(_395 & 65535u) * 1.525902189314365386962890625e-05);
            }
            else
            {
                float3 _472;
                if (_316 == 2u)
                {
                    uint _430;
                    if (_336 != _337)
                    {
                        _430 = (_SplatPos._m0[_338] >> 16u) | ((_SplatPos._m0[(_337 + 4u) >> 2u] & 65535u) << 16u);
                    }
                    else
                    {
                        _430 = _SplatPos._m0[_338];
                    }
                    _472 = float3(half3(half(float(_430 & 2047u) * 0.000488519784994423389434814453125), half(float((_430 >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_430 >> 21u) & 2047u) * 0.000488519784994423389434814453125)));
                }
                else
                {
                    float3 _471;
                    if (_316 == 3u)
                    {
                        uint _454;
                        if (_336 != _337)
                        {
                            _454 = _SplatPos._m0[_338] >> 16u;
                        }
                        else
                        {
                            _454 = _SplatPos._m0[_338];
                        }
                        _471 = float3(half3(half(float(_454 & 63u) * 0.01587301678955554962158203125), half(float((_454 >> 6u) & 31u) * 0.0322580635547637939453125), half(float((_454 >> 11u) & 31u) * 0.0322580635547637939453125)));
                    }
                    else
                    {
                        _471 = float3(0.0);
                    }
                    _472 = _471;
                }
                _473 = _472;
            }
            _474 = _473;
        }
        float3 _475 = mix(float3(_294.posX.x, _294.posY.x, _294.posZ.x), float3(_294.posX.y, _294.posY.y, _294.posZ.y), _474);
        int _476 = 3 * _270;
        uint _477 = uint(_476);
        int _479 = _MeshIndice._m0[_477];
        uint _482 = uint(_476 + 1);
        int _484 = _MeshIndice._m0[_482];
        uint _487 = uint(_476 + 2);
        int _489 = _MeshIndice._m0[_487];
        float _491 = _475.x;
        float _493 = _475.z;
        float _495 = (1.0 - _491) - _493;
        int _500 = int(as_type<uint>(Constants._SplatProp.w));
        uint _503 = uint(_500 * _489) >> 2u;
        float3 _513 = as_type<float3>(uint3(_MeshVertice._m0[_503], _MeshVertice._m0[_503 + 1u], _MeshVertice._m0[_503 + 2u]));
        uint _516 = uint(_500 * _484) >> 2u;
        float3 _526 = as_type<float3>(uint3(_MeshVertice._m0[_516], _MeshVertice._m0[_516 + 1u], _MeshVertice._m0[_516 + 2u]));
        uint _529 = uint(_500 * _479) >> 2u;
        float3 _539 = as_type<float3>(uint3(_MeshVertice._m0[_529], _MeshVertice._m0[_529 + 1u], _MeshVertice._m0[_529 + 2u]));
        float3 _540 = _526 - _513;
        float3 _542 = cross(_540, _539 - _526);
        float _543 = length(_542);
        float3 _549 = _540 / float3(length(_540) + 9.9999999600419720025001879548654e-13);
        float3 _553 = _542 / float3(_543 + 9.9999999600419720025001879548654e-13);
        float3 _554 = cross(_549, _542);
        float _594 = _PoseShadowCompensation._m0[uint(_489)];
        float _597 = _PoseShadowCompensation._m0[uint(_484)];
        float _600 = _PoseShadowCompensation._m0[uint(_479)];
        float3 _605 = _539 - _513;
        float3 _606 = cross(_540, _605);
        float3 _612 = (_513 + (_606 / float3(sqrt(length(_606))))) - _513;
        float4 _641 = float4((((_513 * _491) + (_526 * _493)) + (_539 * _495)) - (_553 * (_475.y * sqrt(_543))), 1.0);
        float4 _642 = Constants._MatrixM * _641;
        float4 _651 = Constants._MatrixV[0] * float4(_642.xyz, 1.0);
        float _654 = _651.z;
        if (_654 <= 0.001000000047497451305389404296875)
        {
            _VisibleMasks._m0[gl_GlobalInvocationID.x] = 0u;
            _SplatProjData0._m0[gl_GlobalInvocationID.x] = SplatProjData{ float4(0.0), half4(half(0.0)), half2(half(0.0)), half2(half(0.0)) };
            break;
        }
        float4 _667 = Constants._MatrixP[0] * float4(_651.xyz, 1.0);
        float _669 = _667.w;
        if (!((((_669 > 0.0) && (abs(_667.x) <= _669)) && (abs(_667.y) <= _669)) && (abs(_667.z) <= _669)))
        {
            _VisibleMasks._m0[gl_GlobalInvocationID.x] = 0u;
            _SplatProjData0._m0[gl_GlobalInvocationID.x] = SplatProjData{ float4(0.0), half4(half(0.0)), half2(half(0.0)), half2(half(0.0)) };
            break;
        }
        uint _696 = as_type<uint>(Constants._SplatProp.z);
        uint _701 = ((gl_GlobalInvocationID.x & 255u) | ((gl_GlobalInvocationID.x & 254u) << 7u)) & 21845u;
        uint _704 = (_701 ^ (_701 >> 1u)) & 13107u;
        uint _707 = (_704 ^ (_704 >> 2u)) & 3855u;
        uint _711 = gl_GlobalInvocationID.x >> 8u;
        SplatChunkInfo _728 = _SplatChunks._m0[_292];
        half3 _801 = half3(half(float2(as_type<half2>(_728.shR)).x), half(float2(as_type<half2>(_728.shG)).x), half(float2(as_type<half2>(_728.shB)).x));
        half3 _814 = half3(half(float2(as_type<half2>(_728.shR >> 16u)).x), half(float2(as_type<half2>(_728.shG >> 16u)).x), half(float2(as_type<half2>(_728.shB >> 16u)).x));
        uint _822 = (_315 >> 8u) & 255u;
        uint _828 = (_315 >> 16u) & 255u;
        bool _829 = _822 == 0u;
        uint _847;
        if (_829)
        {
            _847 = 16u;
        }
        else
        {
            uint _846;
            if (_822 == 1u)
            {
                _846 = 10u;
            }
            else
            {
                uint _845;
                if (_822 == 2u)
                {
                    _845 = 8u;
                }
                else
                {
                    _845 = (_822 == 3u) ? 6u : 4u;
                }
                _846 = _845;
            }
            _847 = _846;
        }
        bool _848 = _828 > 3u;
        uint _852;
        if (_848)
        {
            _852 = _847 + 2u;
        }
        else
        {
            _852 = _847;
        }
        uint _853 = gl_GlobalInvocationID.x * _852;
        uint _854 = _853 & 4294967292u;
        uint _855 = _854 >> 2u;
        uint _869;
        if (_853 != _854)
        {
            _869 = (_SplatOther._m0[_855] >> 16u) | ((_SplatOther._m0[(_854 + 4u) >> 2u] & 65535u) << 16u);
        }
        else
        {
            _869 = _SplatOther._m0[_855];
        }
        float _883 = float((_869 >> 30u) & 3u);
        uint _890 = uint(rint(_883));
        float3 _893 = (float4(float(_869 & 1023u) * 0.000977517105638980865478515625, float((_869 >> 10u) & 1023u) * 0.000977517105638980865478515625, float((_869 >> 20u) & 1023u) * 0.000977517105638980865478515625, _883 * 0.3333333432674407958984375).xyz * 1.41421353816986083984375) - float3(0.707106769084930419921875);
        float4 _895 = float4(_893.x, _893.y, _893.z, _3396.w);
        float3 _896 = _893.xyz;
        _895.w = sqrt(1.0 - fast::clamp(dot(_896, _896), 0.0, 1.0));
        float4 _3410;
        if (_890 == 0u)
        {
            _3410 = _895.wxyz;
        }
        else
        {
            _3410 = _895;
        }
        float4 _3411;
        if (_890 == 1u)
        {
            _3411 = _3410.xwyz;
        }
        else
        {
            _3411 = _3410;
        }
        float4 _3412;
        if (_890 == 2u)
        {
            _3412 = _3411.xywz;
        }
        else
        {
            _3412 = _3411;
        }
        uint _921 = _853 + 4u;
        uint _922 = _921 & 4294967292u;
        uint _923 = _922 >> 2u;
        float3 _1059;
        if (_829)
        {
            uint _931 = (_922 + 4u) >> 2u;
            uint _935 = (_922 + 8u) >> 2u;
            uint _957;
            uint _958;
            uint _959;
            if (_921 != _922)
            {
                _957 = (_SplatOther._m0[_935] >> 16u) | ((_SplatOther._m0[(_922 + 12u) >> 2u] & 65535u) << 16u);
                _958 = (_SplatOther._m0[_931] >> 16u) | ((_SplatOther._m0[_935] & 65535u) << 16u);
                _959 = (_SplatOther._m0[_923] >> 16u) | ((_SplatOther._m0[_931] & 65535u) << 16u);
            }
            else
            {
                _957 = _SplatOther._m0[_935];
                _958 = _SplatOther._m0[_931];
                _959 = _SplatOther._m0[_923];
            }
            _1059 = float3(as_type<float>(_959), as_type<float>(_958), as_type<float>(_957));
        }
        else
        {
            float3 _1058;
            if (_822 == 1u)
            {
                uint _969 = (_922 + 4u) >> 2u;
                uint _980;
                uint _981;
                if (_921 != _922)
                {
                    _980 = _SplatOther._m0[_969] >> 16u;
                    _981 = (_SplatOther._m0[_923] >> 16u) | ((_SplatOther._m0[_969] & 65535u) << 16u);
                }
                else
                {
                    _980 = _SplatOther._m0[_969];
                    _981 = _SplatOther._m0[_923];
                }
                _1058 = float3(float(_981 & 65535u) * 1.525902189314365386962890625e-05, float((_981 >> 16u) & 65535u) * 1.525902189314365386962890625e-05, float(_980 & 65535u) * 1.525902189314365386962890625e-05);
            }
            else
            {
                float3 _1057;
                if (_822 == 2u)
                {
                    uint _1015;
                    if (_921 != _922)
                    {
                        _1015 = (_SplatOther._m0[_923] >> 16u) | ((_SplatOther._m0[(_922 + 4u) >> 2u] & 65535u) << 16u);
                    }
                    else
                    {
                        _1015 = _SplatOther._m0[_923];
                    }
                    _1057 = float3(half3(half(float(_1015 & 2047u) * 0.000488519784994423389434814453125), half(float((_1015 >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_1015 >> 21u) & 2047u) * 0.000488519784994423389434814453125)));
                }
                else
                {
                    float3 _1056;
                    if (_822 == 3u)
                    {
                        uint _1039;
                        if (_921 != _922)
                        {
                            _1039 = _SplatOther._m0[_923] >> 16u;
                        }
                        else
                        {
                            _1039 = _SplatOther._m0[_923];
                        }
                        _1056 = float3(half3(half(float(_1039 & 63u) * 0.01587301678955554962158203125), half(float((_1039 >> 6u) & 31u) * 0.0322580635547637939453125), half(float((_1039 >> 11u) & 31u) * 0.0322580635547637939453125)));
                    }
                    else
                    {
                        _1056 = float3(0.0);
                    }
                    _1057 = _1056;
                }
                _1058 = _1057;
            }
            _1059 = _1058;
        }
        float3 _1060 = mix(float3(half3(half(float2(as_type<half2>(_728.sclX)).x), half(float2(as_type<half2>(_728.sclY)).x), half(float2(as_type<half2>(_728.sclZ)).x))), float3(half3(half(float2(as_type<half2>(_728.sclX >> 16u)).x), half(float2(as_type<half2>(_728.sclY >> 16u)).x), half(float2(as_type<half2>(_728.sclZ >> 16u)).x))), _1059);
        float3 _1061 = _1060 * _1060;
        float3 _1062 = _1061 * _1061;
        float3 _1063 = _1062 * _1062;
        int3 _1064 = int3(uint3(((_711 % 128u) * 16u) + (_707 & 15u), ((_711 / 128u) * 16u) + (_707 >> 8u), 0u));
        half4 _1070 = mix(half4(half(float2(as_type<half2>(_728.colR)).x), half(float2(as_type<half2>(_728.colG)).x), half(float2(as_type<half2>(_728.colB)).x), half(float2(as_type<half2>(_728.colA)).x)), half4(half(float2(as_type<half2>(_728.colR >> 16u)).x), half(float2(as_type<half2>(_728.colG >> 16u)).x), half(float2(as_type<half2>(_728.colB >> 16u)).x), half(float2(as_type<half2>(_728.colA >> 16u)).x)), half4(_SplatColor.read(uint2(_1064.xy), _1064.z)));
        float _1075 = (float(_1070.w) - 0.5) * 0.5;
        half _1082 = half(fma(sqrt(abs(_1075)), float(int(sign(_1075))), 0.5));
        uint _1099;
        if (_848)
        {
            uint _1088 = (_853 + _852) - 2u;
            uint _1089 = _1088 & 4294967292u;
            uint _1090 = _1089 >> 2u;
            uint _1097;
            if (_1088 != _1089)
            {
                _1097 = _SplatOther._m0[_1090] >> 16u;
            }
            else
            {
                _1097 = _SplatOther._m0[_1090];
            }
            _1099 = _1097 & 65535u;
        }
        else
        {
            _1099 = gl_GlobalInvocationID.x;
        }
        bool _1100 = _828 == 0u;
        uint _1120;
        if (_1100)
        {
            _1120 = 192u;
        }
        else
        {
            uint _1119;
            if ((_828 == 1u) || _848)
            {
                _1119 = 96u;
            }
            else
            {
                uint _1118;
                if (_828 == 2u)
                {
                    _1118 = 60u;
                }
                else
                {
                    _1118 = (_828 == 3u) ? 32u : 0u;
                }
                _1119 = _1118;
            }
            _1120 = _1119;
        }
        uint _1121 = _1099 * _1120;
        uint _1122 = _1121 >> 2u;
        uint _1125 = _1122 + 1u;
        uint _1128 = _1122 + 2u;
        uint _1131 = _1122 + 3u;
        uint _1136 = (_1121 + 16u) >> 2u;
        uint _1139 = _1136 + 1u;
        uint _1142 = _1136 + 2u;
        uint _1145 = _1136 + 3u;
        half3 _3728;
        half3 _3735;
        half3 _3742;
        half3 _3750;
        half3 _3758;
        half3 _3766;
        half3 _3774;
        half3 _3782;
        half3 _3791;
        half3 _3800;
        half3 _3809;
        half3 _3818;
        half3 _3827;
        half3 _3836;
        half3 _3845;
        if (_1100)
        {
            uint _1154 = (_1121 + 32u) >> 2u;
            uint _1168 = (_1121 + 48u) >> 2u;
            uint _1182 = (_1121 + 64u) >> 2u;
            uint _1196 = (_1121 + 80u) >> 2u;
            uint _1210 = (_1121 + 96u) >> 2u;
            uint _1224 = (_1121 + 112u) >> 2u;
            uint _1238 = (_1121 + 128u) >> 2u;
            uint _1252 = (_1121 + 144u) >> 2u;
            uint _1266 = (_1121 + 160u) >> 2u;
            _3845 = half3(half(as_type<float>(_SplatSH._m0[_1266 + 2u])), half(as_type<float>(_SplatSH._m0[_1266 + 3u])), half(as_type<float>(_SplatSH._m0[(_1121 + 176u) >> 2u])));
            _3836 = half3(half(as_type<float>(_SplatSH._m0[_1252 + 3u])), half(as_type<float>(_SplatSH._m0[_1266])), half(as_type<float>(_SplatSH._m0[_1266 + 1u])));
            _3827 = half3(half(as_type<float>(_SplatSH._m0[_1252])), half(as_type<float>(_SplatSH._m0[_1252 + 1u])), half(as_type<float>(_SplatSH._m0[_1252 + 2u])));
            _3818 = half3(half(as_type<float>(_SplatSH._m0[_1238 + 1u])), half(as_type<float>(_SplatSH._m0[_1238 + 2u])), half(as_type<float>(_SplatSH._m0[_1238 + 3u])));
            _3809 = half3(half(as_type<float>(_SplatSH._m0[_1224 + 2u])), half(as_type<float>(_SplatSH._m0[_1224 + 3u])), half(as_type<float>(_SplatSH._m0[_1238])));
            _3800 = half3(half(as_type<float>(_SplatSH._m0[_1210 + 3u])), half(as_type<float>(_SplatSH._m0[_1224])), half(as_type<float>(_SplatSH._m0[_1224 + 1u])));
            _3791 = half3(half(as_type<float>(_SplatSH._m0[_1210])), half(as_type<float>(_SplatSH._m0[_1210 + 1u])), half(as_type<float>(_SplatSH._m0[_1210 + 2u])));
            _3782 = half3(half(as_type<float>(_SplatSH._m0[_1196 + 1u])), half(as_type<float>(_SplatSH._m0[_1196 + 2u])), half(as_type<float>(_SplatSH._m0[_1196 + 3u])));
            _3774 = half3(half(as_type<float>(_SplatSH._m0[_1182 + 2u])), half(as_type<float>(_SplatSH._m0[_1182 + 3u])), half(as_type<float>(_SplatSH._m0[_1196])));
            _3766 = half3(half(as_type<float>(_SplatSH._m0[_1168 + 3u])), half(as_type<float>(_SplatSH._m0[_1182])), half(as_type<float>(_SplatSH._m0[_1182 + 1u])));
            _3758 = half3(half(as_type<float>(_SplatSH._m0[_1168])), half(as_type<float>(_SplatSH._m0[_1168 + 1u])), half(as_type<float>(_SplatSH._m0[_1168 + 2u])));
            _3750 = half3(half(as_type<float>(_SplatSH._m0[_1154 + 1u])), half(as_type<float>(_SplatSH._m0[_1154 + 2u])), half(as_type<float>(_SplatSH._m0[_1154 + 3u])));
            _3742 = half3(half(as_type<float>(_SplatSH._m0[_1142])), half(as_type<float>(_SplatSH._m0[_1145])), half(as_type<float>(_SplatSH._m0[_1154])));
            _3735 = half3(half(as_type<float>(_SplatSH._m0[_1131])), half(as_type<float>(_SplatSH._m0[_1136])), half(as_type<float>(_SplatSH._m0[_1139])));
            _3728 = half3(half(as_type<float>(_SplatSH._m0[_1122])), half(as_type<float>(_SplatSH._m0[_1125])), half(as_type<float>(_SplatSH._m0[_1128])));
        }
        else
        {
            half3 _3729;
            half3 _3736;
            half3 _3743;
            half3 _3751;
            half3 _3759;
            half3 _3767;
            half3 _3775;
            half3 _3783;
            half3 _3792;
            half3 _3801;
            half3 _3810;
            half3 _3819;
            half3 _3828;
            half3 _3837;
            half3 _3846;
            if ((_828 == 1u) || _848)
            {
                uint _1513 = (_1121 + 32u) >> 2u;
                uint _1516 = _1513 + 1u;
                uint _1519 = _1513 + 2u;
                uint _1522 = _1513 + 3u;
                uint _1527 = (_1121 + 48u) >> 2u;
                uint _1530 = _1527 + 1u;
                uint _1533 = _1527 + 2u;
                uint _1536 = _1527 + 3u;
                uint _1541 = (_1121 + 64u) >> 2u;
                uint _1544 = _1541 + 1u;
                uint _1547 = _1541 + 2u;
                uint _1550 = _1541 + 3u;
                uint _1555 = (_1121 + 80u) >> 2u;
                uint _1558 = _1555 + 1u;
                _3846 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1558])).x), half(float2(as_type<half2>(_SplatSH._m0[_1558] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1555 + 2u])).x));
                _3837 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1550] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1555])).x), half(float2(as_type<half2>(_SplatSH._m0[_1555] >> 16u)).x));
                _3828 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1547])).x), half(float2(as_type<half2>(_SplatSH._m0[_1547] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1550])).x));
                _3819 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1541] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1544])).x), half(float2(as_type<half2>(_SplatSH._m0[_1544] >> 16u)).x));
                _3810 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1536])).x), half(float2(as_type<half2>(_SplatSH._m0[_1536] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1541])).x));
                _3801 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1530] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1533])).x), half(float2(as_type<half2>(_SplatSH._m0[_1533] >> 16u)).x));
                _3792 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1527])).x), half(float2(as_type<half2>(_SplatSH._m0[_1527] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1530])).x));
                _3783 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1519] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1522])).x), half(float2(as_type<half2>(_SplatSH._m0[_1522] >> 16u)).x));
                _3775 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1516])).x), half(float2(as_type<half2>(_SplatSH._m0[_1516] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1519])).x));
                _3767 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1145] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1513])).x), half(float2(as_type<half2>(_SplatSH._m0[_1513] >> 16u)).x));
                _3759 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1142])).x), half(float2(as_type<half2>(_SplatSH._m0[_1142] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1145])).x));
                _3751 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1136] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1139])).x), half(float2(as_type<half2>(_SplatSH._m0[_1139] >> 16u)).x));
                _3743 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1131])).x), half(float2(as_type<half2>(_SplatSH._m0[_1131] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1136])).x));
                _3736 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1125] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1128])).x), half(float2(as_type<half2>(_SplatSH._m0[_1128] >> 16u)).x));
                _3729 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1122])).x), half(float2(as_type<half2>(_SplatSH._m0[_1122] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1125])).x));
            }
            else
            {
                half3 _3730;
                half3 _3737;
                half3 _3744;
                half3 _3752;
                half3 _3760;
                half3 _3768;
                half3 _3776;
                half3 _3784;
                half3 _3793;
                half3 _3802;
                half3 _3811;
                half3 _3820;
                half3 _3829;
                half3 _3838;
                half3 _3847;
                if (_828 == 2u)
                {
                    uint _1862 = (_1121 + 32u) >> 2u;
                    uint _1865 = _1862 + 1u;
                    uint _1868 = _1862 + 2u;
                    uint _1871 = _1862 + 3u;
                    uint _1876 = (_1121 + 48u) >> 2u;
                    uint _1879 = _1876 + 1u;
                    uint _1882 = _1876 + 2u;
                    _3847 = half3(half(float(_SplatSH._m0[_1882] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1882] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1882] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _3838 = half3(half(float(_SplatSH._m0[_1879] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1879] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1879] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _3829 = half3(half(float(_SplatSH._m0[_1876] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1876] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1876] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _3820 = half3(half(float(_SplatSH._m0[_1871] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1871] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1871] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _3811 = half3(half(float(_SplatSH._m0[_1868] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1868] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1868] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _3802 = half3(half(float(_SplatSH._m0[_1865] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1865] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1865] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _3793 = half3(half(float(_SplatSH._m0[_1862] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1862] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1862] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _3784 = half3(half(float(_SplatSH._m0[_1145] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1145] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1145] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _3776 = half3(half(float(_SplatSH._m0[_1142] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1142] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1142] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _3768 = half3(half(float(_SplatSH._m0[_1139] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1139] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1139] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _3760 = half3(half(float(_SplatSH._m0[_1136] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1136] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1136] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _3752 = half3(half(float(_SplatSH._m0[_1131] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1131] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1131] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _3744 = half3(half(float(_SplatSH._m0[_1128] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1128] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1128] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _3737 = half3(half(float(_SplatSH._m0[_1125] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1125] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1125] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _3730 = half3(half(float(_SplatSH._m0[_1122] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1122] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1122] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                }
                else
                {
                    half3 _3731;
                    half3 _3738;
                    half3 _3745;
                    half3 _3753;
                    half3 _3761;
                    half3 _3769;
                    half3 _3777;
                    half3 _3785;
                    half3 _3794;
                    half3 _3803;
                    half3 _3812;
                    half3 _3821;
                    half3 _3830;
                    half3 _3839;
                    half3 _3848;
                    if (_828 == 3u)
                    {
                        half3 _3734;
                        half3 _3741;
                        half3 _3748;
                        if (_696 > 0u)
                        {
                            uint _2166 = _SplatSH._m0[_1122] >> 16u;
                            _3748 = half3(half(float(_SplatSH._m0[_1125] & 31u) * 0.0322580635547637939453125), half(float((_SplatSH._m0[_1125] >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_SplatSH._m0[_1125] >> 11u) & 31u) * 0.0322580635547637939453125));
                            _3741 = half3(half(float(_2166 & 31u) * 0.0322580635547637939453125), half(float((_2166 >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_2166 >> 11u) & 31u) * 0.0322580635547637939453125));
                            _3734 = half3(half(float(_SplatSH._m0[_1122] & 31u) * 0.0322580635547637939453125), half(float((_SplatSH._m0[_1122] >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_SplatSH._m0[_1122] >> 11u) & 31u) * 0.0322580635547637939453125));
                        }
                        else
                        {
                            _3748 = half3(half(0.0));
                            _3741 = half3(half(0.0));
                            _3734 = half3(half(0.0));
                        }
                        half3 _3755;
                        half3 _3763;
                        half3 _3771;
                        half3 _3779;
                        half3 _3787;
                        if (_696 > 1u)
                        {
                            uint _2204 = _SplatSH._m0[_1125] >> 16u;
                            uint _2239 = _SplatSH._m0[_1128] >> 16u;
                            uint _2274 = _SplatSH._m0[_1131] >> 16u;
                            _3787 = half3(half(float(_2274 & 31u) * 0.0322580635547637939453125), half(float((_2274 >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_2274 >> 11u) & 31u) * 0.0322580635547637939453125));
                            _3779 = half3(half(float(_SplatSH._m0[_1131] & 31u) * 0.0322580635547637939453125), half(float((_SplatSH._m0[_1131] >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_SplatSH._m0[_1131] >> 11u) & 31u) * 0.0322580635547637939453125));
                            _3771 = half3(half(float(_2239 & 31u) * 0.0322580635547637939453125), half(float((_2239 >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_2239 >> 11u) & 31u) * 0.0322580635547637939453125));
                            _3763 = half3(half(float(_SplatSH._m0[_1128] & 31u) * 0.0322580635547637939453125), half(float((_SplatSH._m0[_1128] >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_SplatSH._m0[_1128] >> 11u) & 31u) * 0.0322580635547637939453125));
                            _3755 = half3(half(float(_2204 & 31u) * 0.0322580635547637939453125), half(float((_2204 >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_2204 >> 11u) & 31u) * 0.0322580635547637939453125));
                        }
                        else
                        {
                            _3787 = half3(half(0.0));
                            _3779 = half3(half(0.0));
                            _3771 = half3(half(0.0));
                            _3763 = half3(half(0.0));
                            _3755 = half3(half(0.0));
                        }
                        half3 _3795;
                        half3 _3804;
                        half3 _3813;
                        half3 _3822;
                        half3 _3831;
                        half3 _3840;
                        half3 _3849;
                        if (_696 > 2u)
                        {
                            uint _2312 = _SplatSH._m0[_1136] >> 16u;
                            uint _2347 = _SplatSH._m0[_1139] >> 16u;
                            uint _2382 = _SplatSH._m0[_1142] >> 16u;
                            _3849 = half3(half(float(_SplatSH._m0[_1145] & 31u) * 0.0322580635547637939453125), half(float((_SplatSH._m0[_1145] >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_SplatSH._m0[_1145] >> 11u) & 31u) * 0.0322580635547637939453125));
                            _3840 = half3(half(float(_2382 & 31u) * 0.0322580635547637939453125), half(float((_2382 >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_2382 >> 11u) & 31u) * 0.0322580635547637939453125));
                            _3831 = half3(half(float(_SplatSH._m0[_1142] & 31u) * 0.0322580635547637939453125), half(float((_SplatSH._m0[_1142] >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_SplatSH._m0[_1142] >> 11u) & 31u) * 0.0322580635547637939453125));
                            _3822 = half3(half(float(_2347 & 31u) * 0.0322580635547637939453125), half(float((_2347 >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_2347 >> 11u) & 31u) * 0.0322580635547637939453125));
                            _3813 = half3(half(float(_SplatSH._m0[_1139] & 31u) * 0.0322580635547637939453125), half(float((_SplatSH._m0[_1139] >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_SplatSH._m0[_1139] >> 11u) & 31u) * 0.0322580635547637939453125));
                            _3804 = half3(half(float(_2312 & 31u) * 0.0322580635547637939453125), half(float((_2312 >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_2312 >> 11u) & 31u) * 0.0322580635547637939453125));
                            _3795 = half3(half(float(_SplatSH._m0[_1136] & 31u) * 0.0322580635547637939453125), half(float((_SplatSH._m0[_1136] >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_SplatSH._m0[_1136] >> 11u) & 31u) * 0.0322580635547637939453125));
                        }
                        else
                        {
                            _3849 = half3(half(0.0));
                            _3840 = half3(half(0.0));
                            _3831 = half3(half(0.0));
                            _3822 = half3(half(0.0));
                            _3813 = half3(half(0.0));
                            _3804 = half3(half(0.0));
                            _3795 = half3(half(0.0));
                        }
                        _3848 = _3849;
                        _3839 = _3840;
                        _3830 = _3831;
                        _3821 = _3822;
                        _3812 = _3813;
                        _3803 = _3804;
                        _3794 = _3795;
                        _3785 = _3787;
                        _3777 = _3779;
                        _3769 = _3771;
                        _3761 = _3763;
                        _3753 = _3755;
                        _3745 = _3748;
                        _3738 = _3741;
                        _3731 = _3734;
                    }
                    else
                    {
                        _3848 = half3(half(0.0));
                        _3839 = half3(half(0.0));
                        _3830 = half3(half(0.0));
                        _3821 = half3(half(0.0));
                        _3812 = half3(half(0.0));
                        _3803 = half3(half(0.0));
                        _3794 = half3(half(0.0));
                        _3785 = half3(half(0.0));
                        _3777 = half3(half(0.0));
                        _3769 = half3(half(0.0));
                        _3761 = half3(half(0.0));
                        _3753 = half3(half(0.0));
                        _3745 = half3(half(0.0));
                        _3738 = half3(half(0.0));
                        _3731 = half3(half(0.0));
                    }
                    _3847 = _3848;
                    _3838 = _3839;
                    _3829 = _3830;
                    _3820 = _3821;
                    _3811 = _3812;
                    _3802 = _3803;
                    _3793 = _3794;
                    _3784 = _3785;
                    _3776 = _3777;
                    _3768 = _3769;
                    _3760 = _3761;
                    _3752 = _3753;
                    _3744 = _3745;
                    _3737 = _3738;
                    _3730 = _3731;
                }
                _3846 = _3847;
                _3837 = _3838;
                _3828 = _3829;
                _3819 = _3820;
                _3810 = _3811;
                _3801 = _3802;
                _3792 = _3793;
                _3783 = _3784;
                _3775 = _3776;
                _3767 = _3768;
                _3759 = _3760;
                _3751 = _3752;
                _3743 = _3744;
                _3736 = _3737;
                _3729 = _3730;
            }
            _3845 = _3846;
            _3836 = _3837;
            _3827 = _3828;
            _3818 = _3819;
            _3809 = _3810;
            _3800 = _3801;
            _3791 = _3792;
            _3782 = _3783;
            _3774 = _3775;
            _3766 = _3767;
            _3758 = _3759;
            _3750 = _3751;
            _3742 = _3743;
            _3735 = _3736;
            _3728 = _3729;
        }
        half3 _3852;
        half3 _3854;
        half3 _3856;
        half3 _3858;
        half3 _3860;
        half3 _3862;
        half3 _3864;
        half3 _3866;
        half3 _3869;
        half3 _3872;
        half3 _3875;
        half3 _3878;
        half3 _3881;
        half3 _3885;
        half3 _3889;
        if ((_828 > 0u) && (_828 <= 3u))
        {
            half3 _3884;
            half3 _3888;
            half3 _3892;
            if (_696 > 0u)
            {
                _3892 = mix(_801, _814, _3728);
                _3888 = mix(_801, _814, _3735);
                _3884 = mix(_801, _814, _3742);
            }
            else
            {
                _3892 = _3728;
                _3888 = _3735;
                _3884 = _3742;
            }
            half3 _3868;
            half3 _3871;
            half3 _3874;
            half3 _3877;
            half3 _3880;
            if (_696 > 1u)
            {
                _3880 = mix(_801, _814, _3750);
                _3877 = mix(_801, _814, _3758);
                _3874 = mix(_801, _814, _3766);
                _3871 = mix(_801, _814, _3774);
                _3868 = mix(_801, _814, _3782);
            }
            else
            {
                _3880 = _3750;
                _3877 = _3758;
                _3874 = _3766;
                _3871 = _3774;
                _3868 = _3782;
            }
            half3 _3853;
            half3 _3855;
            half3 _3857;
            half3 _3859;
            half3 _3861;
            half3 _3863;
            half3 _3865;
            if (_696 > 2u)
            {
                _3865 = mix(_801, _814, _3791);
                _3863 = mix(_801, _814, _3800);
                _3861 = mix(_801, _814, _3809);
                _3859 = mix(_801, _814, _3818);
                _3857 = mix(_801, _814, _3827);
                _3855 = mix(_801, _814, _3836);
                _3853 = mix(_801, _814, _3845);
            }
            else
            {
                _3865 = _3791;
                _3863 = _3800;
                _3861 = _3809;
                _3859 = _3818;
                _3857 = _3827;
                _3855 = _3836;
                _3853 = _3845;
            }
            _3889 = _3892;
            _3885 = _3888;
            _3881 = _3884;
            _3878 = _3880;
            _3875 = _3877;
            _3872 = _3874;
            _3869 = _3871;
            _3866 = _3868;
            _3864 = _3865;
            _3862 = _3863;
            _3860 = _3861;
            _3858 = _3859;
            _3856 = _3857;
            _3854 = _3855;
            _3852 = _3853;
        }
        else
        {
            _3889 = _3728;
            _3885 = _3735;
            _3881 = _3742;
            _3878 = _3750;
            _3875 = _3758;
            _3872 = _3766;
            _3869 = _3774;
            _3866 = _3782;
            _3864 = _3791;
            _3862 = _3800;
            _3860 = _3809;
            _3858 = _3818;
            _3856 = _3827;
            _3854 = _3836;
            _3852 = _3845;
        }
        float _2476 = -_3412.x;
        float _2481 = -_3412.z;
        if (_1082 < half(0.0039215087890625))
        {
            _VisibleMasks._m0[gl_GlobalInvocationID.x] = 0u;
            _SplatProjData0._m0[gl_GlobalInvocationID.x] = SplatProjData{ float4(0.0), half4(half(0.0)), half2(half(0.0)), half2(half(0.0)) };
            break;
        }
        _VisibleMasks._m0[gl_GlobalInvocationID.x] = 1u;
        float _2511 = _2481 * _2481;
        float _2514 = _3412.w * _2481;
        float _2518 = _3412.w * _3412.y;
        float _2527 = _3412.w * _2476;
        float3x3 _2546 = (float3x3(float3(_1063.x, 0.0, 0.0), float3(0.0, _1063.y, 0.0), float3(0.0, 0.0, _1063.z)) * float3x3(float3(fma(-2.0, fma(_3412.y, _3412.y, _2511), 1.0), 2.0 * fma(_2476, _3412.y, -_2514), 2.0 * fma(_2476, _2481, _2518)), float3(2.0 * fma(_2476, _3412.y, _2514), fma(-2.0, fma(_2476, _2476, _2511), 1.0), 2.0 * fma(_3412.y, _2481, -_2527)), float3(2.0 * fma(_2476, _2481, -_2518), 2.0 * fma(_3412.y, _2481, _2527), fma(-2.0, fma(_2476, _2476, _3412.y * _3412.y), 1.0)))) * (float3x3(float3(0.0, 0.0, 1.0), float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0)) * float3x3(float3(_540.x, _605.x, _612.x), float3(_540.y, _605.y, _612.y), float3(_540.z, _605.z, _612.z)));
        float3x3 _2548 = transpose(_2546) * _2546;
        float3 _2555 = _2548[0];
        float _2559 = _2548[1].z;
        _PointDistances._m0[gl_GlobalInvocationID.x] = _654;
        float4x4 _2575 = Constants._MatrixV[0] * Constants._MatrixM;
        float4 _2585 = _2575 * _641;
        float _2602 = _2585.z;
        float _2613 = _2585.z;
        float _2626 = (Constants._VecScreenParams.x * Constants._MatrixP[0][0][0]) * 0.5;
        float _2628 = _2585.z;
        float _2629 = _2626 / _2628;
        float _2638 = _2628 * _2628;
        float3x3 _2663 = float3x3(float4(_2575[0][0], _2575[1][0], _2575[2][0], _2575[3][0]).xyz, float4(_2575[0][1], _2575[1][1], _2575[2][1], _2575[3][1]).xyz, float4(_2575[0][2], _2575[1][2], _2575[2][2], _2575[3][2]).xyz) * float3x3(float3(_2629, 0.0, (-(_2626 * (fast::clamp(_2585.x / _2602, (-1.2999999523162841796875) / Constants._MatrixP[0][0][0], 1.2999999523162841796875 / Constants._MatrixP[0][0][0]) * _2602))) / _2638), float3(0.0, _2629, (-(_2626 * (fast::clamp(_2585.y / _2613, (-1.2999999523162841796875) / Constants._MatrixP[0][0][0], 1.2999999523162841796875 / Constants._MatrixP[0][0][0]) * _2613))) / _2638), float3(0.0));
        float3x3 _2688 = (transpose(_2663) * float3x3(_2555, float3(_2555.y, _2548[1].y, _2559), float3(_2555.z, _2559, _2548[2].z))) * _2663;
        float _2690 = _2688[0].x;
        float _2692 = _2688[1].y;
        float _2694 = _2688[0].y;
        float _2698 = -(_2694 * _2694);
        float _2700 = precise::max(9.9999999747524270787835121154785e-07, fma(_2690, _2692, _2698));
        float _2703 = _2690 + 0.100000001490116119384765625;
        float _2714 = precise::max(9.9999999747524270787835121154785e-07, fma(_2703, _2692 + 0.100000001490116119384765625, _2698));
        float _2730 = _2688[1].y + 0.100000001490116119384765625;
        float _2734 = _2688[0].y;
        float _2745 = _2703 + _2730;
        float _2749 = length(float2((_2703 - _2730) * 0.5, _2734));
        float _2750 = fma(0.5, _2745, _2749);
        float2 _2756 = fast::normalize(float2(_2734, _2750 - _2703));
        float _2759 = -_2756.y;
        float2 _3384 = _2756;
        _3384.y = _2759;
        float3 _2789 = ((Constants._CameraPosWS[0].xyz - _642.xyz) * float3x3(float4(Constants._MatrixWorldToObject[0][0], Constants._MatrixWorldToObject[1][0], Constants._MatrixWorldToObject[2][0], Constants._MatrixWorldToObject[3][0]).xyz, float4(Constants._MatrixWorldToObject[0][1], Constants._MatrixWorldToObject[1][1], Constants._MatrixWorldToObject[2][1], Constants._MatrixWorldToObject[3][1]).xyz, float4(Constants._MatrixWorldToObject[0][2], Constants._MatrixWorldToObject[1][2], Constants._MatrixWorldToObject[2][2], Constants._MatrixWorldToObject[3][2]).xyz)) * transpose(transpose(float3x3(_549, _553, _554 / float3(length(_554) + 9.9999999600419720025001879548654e-13))));
        _2789.y = -_2789.y;
        int _2797 = int(_696);
        half3 _2798 = half3(fast::normalize(_2789)) * half(-1.0);
        half _2800 = _2798.x;
        half _2802 = _2798.y;
        half _2804 = _2798.z;
        half3 _2937;
        if (_2797 >= 1)
        {
            half3 _2818 = half3(float3(_1070.xyz) + (float3((((-_3889) * _2802) + (_3885 * _2804)) - (_3881 * _2800)) * 0.4886024892330169677734375));
            half3 _2936;
            if (_2797 >= 2)
            {
                half _2822 = _2800 * _2800;
                half _2823 = _2802 * _2802;
                half _2824 = _2804 * _2804;
                float _2829 = float(_2800 * _2802);
                half _2838 = -_2822;
                half _2840 = -_2802;
                half _2852 = -_2823;
                float _2854 = float(fma(_2800, _2800, _2852));
                half3 _2860 = half3(float3(_2818) + (((((float3(_3878) * (1.092548370361328125 * _2829)) + (float3(_3875) * ((-1.092548370361328125) * float(_2802 * _2804)))) + (float3(_3872) * (0.315391600131988525390625 * float(fma(_2840, _2802, fma(half(2.0), _2824, _2838)))))) + (float3(_3869) * ((-1.092548370361328125) * float(_2800 * _2804)))) + (float3(_3866) * (0.5462741851806640625 * _2854))));
                half3 _2935;
                if (_2797 >= 3)
                {
                    float _2865 = float(_2802);
                    float _2875 = float(_2804);
                    float _2886 = float(fma(_2840, _2802, fma(half(4.0), _2824, _2838)));
                    float _2903 = float(_2800);
                    _2935 = half3(float3(_2860) + (((((((float3(_3864) * (((-0.590043604373931884765625) * _2865) * float(fma(half(3.0), _2822, _2852)))) + (float3(_3862) * ((2.8906114101409912109375 * _2829) * _2875))) + (float3(_3860) * (((-0.4570457935333251953125) * _2865) * _2886))) + (float3(_3858) * ((0.3731763064861297607421875 * _2875) * float(fma(-half(3.0), _2823, fma(half(2.0), _2824, -(half(3.0) * _2822))))))) + (float3(_3856) * (((-0.4570457935333251953125) * _2903) * _2886))) + (float3(_3854) * ((1.44530570507049560546875 * _2875) * _2854))) + (float3(_3852) * (((-0.590043604373931884765625) * _2903) * float(fma(_2800, _2800, -(half(3.0) * _2823)))))));
                }
                else
                {
                    _2935 = _2860;
                }
                _2936 = _2935;
            }
            else
            {
                _2936 = _2818;
            }
            _2937 = _2936;
        }
        else
        {
            _2937 = _1070.xyz;
        }
        half3 _2943 = half3(float3(max(_2937, half3(half(0.0))).xyz) * (fma(_495, _600, fma(_491, _594, _493 * _597)) + 1.0));
        half4 _2944 = half4(_2943.x, _2943.y, _2943.z, _3397.w);
        _2944.w = half(precise::min(float(_1082) * (((_2700 <= 9.9999999747524270787835121154785e-07) || (_2714 <= 9.9999999747524270787835121154785e-07)) ? 0.0 : sqrt((_2700 / (_2714 + 9.9999999747524270787835121154785e-07)) + 9.9999999747524270787835121154785e-07)), 65000.0));
        _SplatProjData0._m0[gl_GlobalInvocationID.x] = SplatProjData{ _667, _2944, half2(_3384 * precise::min(sqrt(2.0 * _2750), 4096.0)), half2(float2(_2759, -_2756.x) * precise::min(sqrt(2.0 * precise::max(fma(0.5, _2745, -_2749), 0.100000001490116119384765625)), 4096.0)) };
        break;
    } while(false);
}


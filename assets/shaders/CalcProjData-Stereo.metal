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

constant float4 _3918 = {};
constant half4 _3919 = {};

kernel void CalcProjData(constant type_Constants& Constants [[buffer(0)]], const device type_StructuredBuffer_int& _GSIndice [[buffer(1)]], const device type_StructuredBuffer_int& _TriangleCullFlag [[buffer(2)]], const device type_ByteAddressBuffer& _SplatPropData [[buffer(3)]], const device type_StructuredBuffer_SplatChunkInfo& _SplatChunks [[buffer(4)]], const device type_ByteAddressBuffer& _SplatPos [[buffer(5)]], const device type_ByteAddressBuffer& _SplatOther [[buffer(6)]], const device type_ByteAddressBuffer& _SplatSH [[buffer(7)]], const device type_StructuredBuffer_int& _MeshIndice [[buffer(8)]], const device type_ByteAddressBuffer& _MeshVertice [[buffer(9)]], const device type_StructuredBuffer_float& _PoseShadowCompensation [[buffer(10)]], device type_RWStructuredBuffer_uint& _VisibleMasks [[buffer(11)]], device type_RWStructuredBuffer_float& _PointDistances [[buffer(12)]], device type_RWStructuredBuffer_SplatProjData& _SplatProjData0 [[buffer(13)]], device type_RWStructuredBuffer_SplatProjData& _SplatProjData1 [[buffer(14)]], texture2d<float> _SplatColor [[texture(0)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    do
    {
        if (gl_GlobalInvocationID.x >= as_type<uint>(Constants._SplatProp.x))
        {
            break;
        }
        int _284 = _GSIndice._m0[gl_GlobalInvocationID.x];
        uint _296 = (_SplatPropData._m0[(gl_GlobalInvocationID.x & 4294967292u) >> 2u] >> ((8u * (gl_GlobalInvocationID.x % 4u)) & 31u)) & 255u;
        if ((_296 != 2u) && (_296 != uint(_TriangleCullFlag._m0[uint(_284)] + 1)))
        {
            _VisibleMasks._m0[gl_GlobalInvocationID.x] = 0u;
            _SplatProjData0._m0[gl_GlobalInvocationID.x] = SplatProjData{ float4(0.0), half4(half(0.0)), half2(half(0.0)), half2(half(0.0)) };
            _SplatProjData1._m0[gl_GlobalInvocationID.x] = SplatProjData{ float4(0.0), half4(half(0.0)), half2(half(0.0)), half2(half(0.0)) };
            break;
        }
        uint _307 = gl_GlobalInvocationID.x / 256u;
        SplatChunkInfo _309 = _SplatChunks._m0[_307];
        uint _330 = as_type<uint>(Constants._SplatProp.y);
        uint _331 = _330 & 255u;
        bool _332 = _331 == 0u;
        uint _350;
        if (_332)
        {
            _350 = 12u;
        }
        else
        {
            uint _349;
            if (_331 == 1u)
            {
                _349 = 6u;
            }
            else
            {
                uint _348;
                if (_331 == 2u)
                {
                    _348 = 4u;
                }
                else
                {
                    _348 = (_331 == 3u) ? 2u : 0u;
                }
                _349 = _348;
            }
            _350 = _349;
        }
        uint _351 = gl_GlobalInvocationID.x * _350;
        uint _352 = _351 & 4294967292u;
        uint _353 = _352 >> 2u;
        float3 _489;
        if (_332)
        {
            uint _361 = (_352 + 4u) >> 2u;
            uint _365 = (_352 + 8u) >> 2u;
            uint _387;
            uint _388;
            uint _389;
            if (_351 != _352)
            {
                _387 = (_SplatPos._m0[_365] >> 16u) | ((_SplatPos._m0[(_352 + 12u) >> 2u] & 65535u) << 16u);
                _388 = (_SplatPos._m0[_361] >> 16u) | ((_SplatPos._m0[_365] & 65535u) << 16u);
                _389 = (_SplatPos._m0[_353] >> 16u) | ((_SplatPos._m0[_361] & 65535u) << 16u);
            }
            else
            {
                _387 = _SplatPos._m0[_365];
                _388 = _SplatPos._m0[_361];
                _389 = _SplatPos._m0[_353];
            }
            _489 = float3(as_type<float>(_389), as_type<float>(_388), as_type<float>(_387));
        }
        else
        {
            float3 _488;
            if (_331 == 1u)
            {
                uint _399 = (_352 + 4u) >> 2u;
                uint _410;
                uint _411;
                if (_351 != _352)
                {
                    _410 = _SplatPos._m0[_399] >> 16u;
                    _411 = (_SplatPos._m0[_353] >> 16u) | ((_SplatPos._m0[_399] & 65535u) << 16u);
                }
                else
                {
                    _410 = _SplatPos._m0[_399];
                    _411 = _SplatPos._m0[_353];
                }
                _488 = float3(float(_411 & 65535u) * 1.525902189314365386962890625e-05, float((_411 >> 16u) & 65535u) * 1.525902189314365386962890625e-05, float(_410 & 65535u) * 1.525902189314365386962890625e-05);
            }
            else
            {
                float3 _487;
                if (_331 == 2u)
                {
                    uint _445;
                    if (_351 != _352)
                    {
                        _445 = (_SplatPos._m0[_353] >> 16u) | ((_SplatPos._m0[(_352 + 4u) >> 2u] & 65535u) << 16u);
                    }
                    else
                    {
                        _445 = _SplatPos._m0[_353];
                    }
                    _487 = float3(half3(half(float(_445 & 2047u) * 0.000488519784994423389434814453125), half(float((_445 >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_445 >> 21u) & 2047u) * 0.000488519784994423389434814453125)));
                }
                else
                {
                    float3 _486;
                    if (_331 == 3u)
                    {
                        uint _469;
                        if (_351 != _352)
                        {
                            _469 = _SplatPos._m0[_353] >> 16u;
                        }
                        else
                        {
                            _469 = _SplatPos._m0[_353];
                        }
                        _486 = float3(half3(half(float(_469 & 63u) * 0.01587301678955554962158203125), half(float((_469 >> 6u) & 31u) * 0.0322580635547637939453125), half(float((_469 >> 11u) & 31u) * 0.0322580635547637939453125)));
                    }
                    else
                    {
                        _486 = float3(0.0);
                    }
                    _487 = _486;
                }
                _488 = _487;
            }
            _489 = _488;
        }
        float3 _490 = mix(float3(_309.posX.x, _309.posY.x, _309.posZ.x), float3(_309.posX.y, _309.posY.y, _309.posZ.y), _489);
        int _491 = 3 * _284;
        uint _492 = uint(_491);
        int _494 = _MeshIndice._m0[_492];
        uint _497 = uint(_491 + 1);
        int _499 = _MeshIndice._m0[_497];
        uint _502 = uint(_491 + 2);
        int _504 = _MeshIndice._m0[_502];
        float _506 = _490.x;
        float _508 = _490.z;
        float _510 = (1.0 - _506) - _508;
        int _515 = int(as_type<uint>(Constants._SplatProp.w));
        uint _518 = uint(_515 * _504) >> 2u;
        float3 _528 = as_type<float3>(uint3(_MeshVertice._m0[_518], _MeshVertice._m0[_518 + 1u], _MeshVertice._m0[_518 + 2u]));
        uint _531 = uint(_515 * _499) >> 2u;
        float3 _541 = as_type<float3>(uint3(_MeshVertice._m0[_531], _MeshVertice._m0[_531 + 1u], _MeshVertice._m0[_531 + 2u]));
        uint _544 = uint(_515 * _494) >> 2u;
        float3 _554 = as_type<float3>(uint3(_MeshVertice._m0[_544], _MeshVertice._m0[_544 + 1u], _MeshVertice._m0[_544 + 2u]));
        float3 _555 = _541 - _528;
        float3 _557 = cross(_555, _554 - _541);
        float _558 = length(_557);
        float3 _564 = _555 / float3(length(_555) + 9.9999999600419720025001879548654e-13);
        float3 _568 = _557 / float3(_558 + 9.9999999600419720025001879548654e-13);
        float3 _569 = cross(_564, _557);
        float _609 = _PoseShadowCompensation._m0[uint(_504)];
        float _612 = _PoseShadowCompensation._m0[uint(_499)];
        float _615 = _PoseShadowCompensation._m0[uint(_494)];
        float _619 = fma(_510, _615, fma(_506, _609, _508 * _612)) + 1.0;
        float3 _620 = _554 - _528;
        float3 _621 = cross(_555, _620);
        float3 _627 = (_528 + (_621 / float3(sqrt(length(_621))))) - _528;
        float4 _656 = float4((((_528 * _506) + (_541 * _508)) + (_554 * _510)) - (_568 * (_490.y * sqrt(_558))), 1.0);
        float4 _657 = Constants._MatrixM * _656;
        float3 _658 = _657.xyz;
        float4 _665 = float4(_657.xyz, 1.0);
        float4 _666 = Constants._MatrixV[0] * _665;
        float _669 = _666.z;
        if (_669 <= 0.001000000047497451305389404296875)
        {
            _VisibleMasks._m0[gl_GlobalInvocationID.x] = 0u;
            _SplatProjData0._m0[gl_GlobalInvocationID.x] = SplatProjData{ float4(0.0), half4(half(0.0)), half2(half(0.0)), half2(half(0.0)) };
            _SplatProjData1._m0[gl_GlobalInvocationID.x] = SplatProjData{ float4(0.0), half4(half(0.0)), half2(half(0.0)), half2(half(0.0)) };
            break;
        }
        float4 _683 = Constants._MatrixP[0] * float4(_666.xyz, 1.0);
        float _685 = _683.w;
        bool _703 = (((_685 > 0.0) && (abs(_683.x) <= _685)) && (abs(_683.y) <= _685)) && (abs(_683.z) <= _685);
        float4 _715 = (Constants._MatrixP[1] * Constants._MatrixV[1]) * _665;
        float _717 = _715.w;
        bool _735 = (((_717 > 0.0) && (abs(_715.x) <= _717)) && (abs(_715.y) <= _717)) && (abs(_715.z) <= _717);
        if ((!_703) && (!_735))
        {
            _VisibleMasks._m0[gl_GlobalInvocationID.x] = 0u;
            _SplatProjData0._m0[gl_GlobalInvocationID.x] = SplatProjData{ float4(0.0), half4(half(0.0)), half2(half(0.0)), half2(half(0.0)) };
            _SplatProjData1._m0[gl_GlobalInvocationID.x] = SplatProjData{ float4(0.0), half4(half(0.0)), half2(half(0.0)), half2(half(0.0)) };
            break;
        }
        uint _747 = as_type<uint>(Constants._SplatProp.z);
        uint _752 = ((gl_GlobalInvocationID.x & 255u) | ((gl_GlobalInvocationID.x & 254u) << 7u)) & 21845u;
        uint _755 = (_752 ^ (_752 >> 1u)) & 13107u;
        uint _758 = (_755 ^ (_755 >> 2u)) & 3855u;
        uint _762 = gl_GlobalInvocationID.x >> 8u;
        SplatChunkInfo _779 = _SplatChunks._m0[_307];
        half3 _852 = half3(half(float2(as_type<half2>(_779.shR)).x), half(float2(as_type<half2>(_779.shG)).x), half(float2(as_type<half2>(_779.shB)).x));
        half3 _865 = half3(half(float2(as_type<half2>(_779.shR >> 16u)).x), half(float2(as_type<half2>(_779.shG >> 16u)).x), half(float2(as_type<half2>(_779.shB >> 16u)).x));
        uint _873 = (_330 >> 8u) & 255u;
        uint _879 = (_330 >> 16u) & 255u;
        bool _880 = _873 == 0u;
        uint _898;
        if (_880)
        {
            _898 = 16u;
        }
        else
        {
            uint _897;
            if (_873 == 1u)
            {
                _897 = 10u;
            }
            else
            {
                uint _896;
                if (_873 == 2u)
                {
                    _896 = 8u;
                }
                else
                {
                    _896 = (_873 == 3u) ? 6u : 4u;
                }
                _897 = _896;
            }
            _898 = _897;
        }
        bool _899 = _879 > 3u;
        uint _903;
        if (_899)
        {
            _903 = _898 + 2u;
        }
        else
        {
            _903 = _898;
        }
        uint _904 = gl_GlobalInvocationID.x * _903;
        uint _905 = _904 & 4294967292u;
        uint _906 = _905 >> 2u;
        uint _920;
        if (_904 != _905)
        {
            _920 = (_SplatOther._m0[_906] >> 16u) | ((_SplatOther._m0[(_905 + 4u) >> 2u] & 65535u) << 16u);
        }
        else
        {
            _920 = _SplatOther._m0[_906];
        }
        float _934 = float((_920 >> 30u) & 3u);
        uint _941 = uint(rint(_934));
        float3 _944 = (float4(float(_920 & 1023u) * 0.000977517105638980865478515625, float((_920 >> 10u) & 1023u) * 0.000977517105638980865478515625, float((_920 >> 20u) & 1023u) * 0.000977517105638980865478515625, _934 * 0.3333333432674407958984375).xyz * 1.41421353816986083984375) - float3(0.707106769084930419921875);
        float4 _946 = float4(_944.x, _944.y, _944.z, _3918.w);
        float3 _947 = _944.xyz;
        _946.w = sqrt(1.0 - fast::clamp(dot(_947, _947), 0.0, 1.0));
        float4 _3932;
        if (_941 == 0u)
        {
            _3932 = _946.wxyz;
        }
        else
        {
            _3932 = _946;
        }
        float4 _3933;
        if (_941 == 1u)
        {
            _3933 = _3932.xwyz;
        }
        else
        {
            _3933 = _3932;
        }
        float4 _3934;
        if (_941 == 2u)
        {
            _3934 = _3933.xywz;
        }
        else
        {
            _3934 = _3933;
        }
        uint _972 = _904 + 4u;
        uint _973 = _972 & 4294967292u;
        uint _974 = _973 >> 2u;
        float3 _1110;
        if (_880)
        {
            uint _982 = (_973 + 4u) >> 2u;
            uint _986 = (_973 + 8u) >> 2u;
            uint _1008;
            uint _1009;
            uint _1010;
            if (_972 != _973)
            {
                _1008 = (_SplatOther._m0[_986] >> 16u) | ((_SplatOther._m0[(_973 + 12u) >> 2u] & 65535u) << 16u);
                _1009 = (_SplatOther._m0[_982] >> 16u) | ((_SplatOther._m0[_986] & 65535u) << 16u);
                _1010 = (_SplatOther._m0[_974] >> 16u) | ((_SplatOther._m0[_982] & 65535u) << 16u);
            }
            else
            {
                _1008 = _SplatOther._m0[_986];
                _1009 = _SplatOther._m0[_982];
                _1010 = _SplatOther._m0[_974];
            }
            _1110 = float3(as_type<float>(_1010), as_type<float>(_1009), as_type<float>(_1008));
        }
        else
        {
            float3 _1109;
            if (_873 == 1u)
            {
                uint _1020 = (_973 + 4u) >> 2u;
                uint _1031;
                uint _1032;
                if (_972 != _973)
                {
                    _1031 = _SplatOther._m0[_1020] >> 16u;
                    _1032 = (_SplatOther._m0[_974] >> 16u) | ((_SplatOther._m0[_1020] & 65535u) << 16u);
                }
                else
                {
                    _1031 = _SplatOther._m0[_1020];
                    _1032 = _SplatOther._m0[_974];
                }
                _1109 = float3(float(_1032 & 65535u) * 1.525902189314365386962890625e-05, float((_1032 >> 16u) & 65535u) * 1.525902189314365386962890625e-05, float(_1031 & 65535u) * 1.525902189314365386962890625e-05);
            }
            else
            {
                float3 _1108;
                if (_873 == 2u)
                {
                    uint _1066;
                    if (_972 != _973)
                    {
                        _1066 = (_SplatOther._m0[_974] >> 16u) | ((_SplatOther._m0[(_973 + 4u) >> 2u] & 65535u) << 16u);
                    }
                    else
                    {
                        _1066 = _SplatOther._m0[_974];
                    }
                    _1108 = float3(half3(half(float(_1066 & 2047u) * 0.000488519784994423389434814453125), half(float((_1066 >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_1066 >> 21u) & 2047u) * 0.000488519784994423389434814453125)));
                }
                else
                {
                    float3 _1107;
                    if (_873 == 3u)
                    {
                        uint _1090;
                        if (_972 != _973)
                        {
                            _1090 = _SplatOther._m0[_974] >> 16u;
                        }
                        else
                        {
                            _1090 = _SplatOther._m0[_974];
                        }
                        _1107 = float3(half3(half(float(_1090 & 63u) * 0.01587301678955554962158203125), half(float((_1090 >> 6u) & 31u) * 0.0322580635547637939453125), half(float((_1090 >> 11u) & 31u) * 0.0322580635547637939453125)));
                    }
                    else
                    {
                        _1107 = float3(0.0);
                    }
                    _1108 = _1107;
                }
                _1109 = _1108;
            }
            _1110 = _1109;
        }
        float3 _1111 = mix(float3(half3(half(float2(as_type<half2>(_779.sclX)).x), half(float2(as_type<half2>(_779.sclY)).x), half(float2(as_type<half2>(_779.sclZ)).x))), float3(half3(half(float2(as_type<half2>(_779.sclX >> 16u)).x), half(float2(as_type<half2>(_779.sclY >> 16u)).x), half(float2(as_type<half2>(_779.sclZ >> 16u)).x))), _1110);
        float3 _1112 = _1111 * _1111;
        float3 _1113 = _1112 * _1112;
        float3 _1114 = _1113 * _1113;
        int3 _1115 = int3(uint3(((_762 % 128u) * 16u) + (_758 & 15u), ((_762 / 128u) * 16u) + (_758 >> 8u), 0u));
        half4 _1121 = mix(half4(half(float2(as_type<half2>(_779.colR)).x), half(float2(as_type<half2>(_779.colG)).x), half(float2(as_type<half2>(_779.colB)).x), half(float2(as_type<half2>(_779.colA)).x)), half4(half(float2(as_type<half2>(_779.colR >> 16u)).x), half(float2(as_type<half2>(_779.colG >> 16u)).x), half(float2(as_type<half2>(_779.colB >> 16u)).x), half(float2(as_type<half2>(_779.colA >> 16u)).x)), half4(_SplatColor.read(uint2(_1115.xy), _1115.z)));
        float _1126 = (float(_1121.w) - 0.5) * 0.5;
        half _1133 = half(fma(sqrt(abs(_1126)), float(int(sign(_1126))), 0.5));
        uint _1150;
        if (_899)
        {
            uint _1139 = (_904 + _903) - 2u;
            uint _1140 = _1139 & 4294967292u;
            uint _1141 = _1140 >> 2u;
            uint _1148;
            if (_1139 != _1140)
            {
                _1148 = _SplatOther._m0[_1141] >> 16u;
            }
            else
            {
                _1148 = _SplatOther._m0[_1141];
            }
            _1150 = _1148 & 65535u;
        }
        else
        {
            _1150 = gl_GlobalInvocationID.x;
        }
        bool _1151 = _879 == 0u;
        uint _1171;
        if (_1151)
        {
            _1171 = 192u;
        }
        else
        {
            uint _1170;
            if ((_879 == 1u) || _899)
            {
                _1170 = 96u;
            }
            else
            {
                uint _1169;
                if (_879 == 2u)
                {
                    _1169 = 60u;
                }
                else
                {
                    _1169 = (_879 == 3u) ? 32u : 0u;
                }
                _1170 = _1169;
            }
            _1171 = _1170;
        }
        uint _1172 = _1150 * _1171;
        uint _1173 = _1172 >> 2u;
        uint _1176 = _1173 + 1u;
        uint _1179 = _1173 + 2u;
        uint _1182 = _1173 + 3u;
        uint _1187 = (_1172 + 16u) >> 2u;
        uint _1190 = _1187 + 1u;
        uint _1193 = _1187 + 2u;
        uint _1196 = _1187 + 3u;
        half3 _4250;
        half3 _4257;
        half3 _4264;
        half3 _4272;
        half3 _4280;
        half3 _4288;
        half3 _4296;
        half3 _4304;
        half3 _4313;
        half3 _4322;
        half3 _4331;
        half3 _4340;
        half3 _4349;
        half3 _4358;
        half3 _4367;
        if (_1151)
        {
            uint _1205 = (_1172 + 32u) >> 2u;
            uint _1219 = (_1172 + 48u) >> 2u;
            uint _1233 = (_1172 + 64u) >> 2u;
            uint _1247 = (_1172 + 80u) >> 2u;
            uint _1261 = (_1172 + 96u) >> 2u;
            uint _1275 = (_1172 + 112u) >> 2u;
            uint _1289 = (_1172 + 128u) >> 2u;
            uint _1303 = (_1172 + 144u) >> 2u;
            uint _1317 = (_1172 + 160u) >> 2u;
            _4367 = half3(half(as_type<float>(_SplatSH._m0[_1317 + 2u])), half(as_type<float>(_SplatSH._m0[_1317 + 3u])), half(as_type<float>(_SplatSH._m0[(_1172 + 176u) >> 2u])));
            _4358 = half3(half(as_type<float>(_SplatSH._m0[_1303 + 3u])), half(as_type<float>(_SplatSH._m0[_1317])), half(as_type<float>(_SplatSH._m0[_1317 + 1u])));
            _4349 = half3(half(as_type<float>(_SplatSH._m0[_1303])), half(as_type<float>(_SplatSH._m0[_1303 + 1u])), half(as_type<float>(_SplatSH._m0[_1303 + 2u])));
            _4340 = half3(half(as_type<float>(_SplatSH._m0[_1289 + 1u])), half(as_type<float>(_SplatSH._m0[_1289 + 2u])), half(as_type<float>(_SplatSH._m0[_1289 + 3u])));
            _4331 = half3(half(as_type<float>(_SplatSH._m0[_1275 + 2u])), half(as_type<float>(_SplatSH._m0[_1275 + 3u])), half(as_type<float>(_SplatSH._m0[_1289])));
            _4322 = half3(half(as_type<float>(_SplatSH._m0[_1261 + 3u])), half(as_type<float>(_SplatSH._m0[_1275])), half(as_type<float>(_SplatSH._m0[_1275 + 1u])));
            _4313 = half3(half(as_type<float>(_SplatSH._m0[_1261])), half(as_type<float>(_SplatSH._m0[_1261 + 1u])), half(as_type<float>(_SplatSH._m0[_1261 + 2u])));
            _4304 = half3(half(as_type<float>(_SplatSH._m0[_1247 + 1u])), half(as_type<float>(_SplatSH._m0[_1247 + 2u])), half(as_type<float>(_SplatSH._m0[_1247 + 3u])));
            _4296 = half3(half(as_type<float>(_SplatSH._m0[_1233 + 2u])), half(as_type<float>(_SplatSH._m0[_1233 + 3u])), half(as_type<float>(_SplatSH._m0[_1247])));
            _4288 = half3(half(as_type<float>(_SplatSH._m0[_1219 + 3u])), half(as_type<float>(_SplatSH._m0[_1233])), half(as_type<float>(_SplatSH._m0[_1233 + 1u])));
            _4280 = half3(half(as_type<float>(_SplatSH._m0[_1219])), half(as_type<float>(_SplatSH._m0[_1219 + 1u])), half(as_type<float>(_SplatSH._m0[_1219 + 2u])));
            _4272 = half3(half(as_type<float>(_SplatSH._m0[_1205 + 1u])), half(as_type<float>(_SplatSH._m0[_1205 + 2u])), half(as_type<float>(_SplatSH._m0[_1205 + 3u])));
            _4264 = half3(half(as_type<float>(_SplatSH._m0[_1193])), half(as_type<float>(_SplatSH._m0[_1196])), half(as_type<float>(_SplatSH._m0[_1205])));
            _4257 = half3(half(as_type<float>(_SplatSH._m0[_1182])), half(as_type<float>(_SplatSH._m0[_1187])), half(as_type<float>(_SplatSH._m0[_1190])));
            _4250 = half3(half(as_type<float>(_SplatSH._m0[_1173])), half(as_type<float>(_SplatSH._m0[_1176])), half(as_type<float>(_SplatSH._m0[_1179])));
        }
        else
        {
            half3 _4251;
            half3 _4258;
            half3 _4265;
            half3 _4273;
            half3 _4281;
            half3 _4289;
            half3 _4297;
            half3 _4305;
            half3 _4314;
            half3 _4323;
            half3 _4332;
            half3 _4341;
            half3 _4350;
            half3 _4359;
            half3 _4368;
            if ((_879 == 1u) || _899)
            {
                uint _1564 = (_1172 + 32u) >> 2u;
                uint _1567 = _1564 + 1u;
                uint _1570 = _1564 + 2u;
                uint _1573 = _1564 + 3u;
                uint _1578 = (_1172 + 48u) >> 2u;
                uint _1581 = _1578 + 1u;
                uint _1584 = _1578 + 2u;
                uint _1587 = _1578 + 3u;
                uint _1592 = (_1172 + 64u) >> 2u;
                uint _1595 = _1592 + 1u;
                uint _1598 = _1592 + 2u;
                uint _1601 = _1592 + 3u;
                uint _1606 = (_1172 + 80u) >> 2u;
                uint _1609 = _1606 + 1u;
                _4368 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1609])).x), half(float2(as_type<half2>(_SplatSH._m0[_1609] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1606 + 2u])).x));
                _4359 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1601] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1606])).x), half(float2(as_type<half2>(_SplatSH._m0[_1606] >> 16u)).x));
                _4350 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1598])).x), half(float2(as_type<half2>(_SplatSH._m0[_1598] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1601])).x));
                _4341 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1592] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1595])).x), half(float2(as_type<half2>(_SplatSH._m0[_1595] >> 16u)).x));
                _4332 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1587])).x), half(float2(as_type<half2>(_SplatSH._m0[_1587] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1592])).x));
                _4323 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1581] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1584])).x), half(float2(as_type<half2>(_SplatSH._m0[_1584] >> 16u)).x));
                _4314 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1578])).x), half(float2(as_type<half2>(_SplatSH._m0[_1578] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1581])).x));
                _4305 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1570] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1573])).x), half(float2(as_type<half2>(_SplatSH._m0[_1573] >> 16u)).x));
                _4297 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1567])).x), half(float2(as_type<half2>(_SplatSH._m0[_1567] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1570])).x));
                _4289 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1196] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1564])).x), half(float2(as_type<half2>(_SplatSH._m0[_1564] >> 16u)).x));
                _4281 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1193])).x), half(float2(as_type<half2>(_SplatSH._m0[_1193] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1196])).x));
                _4273 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1187] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1190])).x), half(float2(as_type<half2>(_SplatSH._m0[_1190] >> 16u)).x));
                _4265 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1182])).x), half(float2(as_type<half2>(_SplatSH._m0[_1182] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1187])).x));
                _4258 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1176] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1179])).x), half(float2(as_type<half2>(_SplatSH._m0[_1179] >> 16u)).x));
                _4251 = half3(half(float2(as_type<half2>(_SplatSH._m0[_1173])).x), half(float2(as_type<half2>(_SplatSH._m0[_1173] >> 16u)).x), half(float2(as_type<half2>(_SplatSH._m0[_1176])).x));
            }
            else
            {
                half3 _4252;
                half3 _4259;
                half3 _4266;
                half3 _4274;
                half3 _4282;
                half3 _4290;
                half3 _4298;
                half3 _4306;
                half3 _4315;
                half3 _4324;
                half3 _4333;
                half3 _4342;
                half3 _4351;
                half3 _4360;
                half3 _4369;
                if (_879 == 2u)
                {
                    uint _1913 = (_1172 + 32u) >> 2u;
                    uint _1916 = _1913 + 1u;
                    uint _1919 = _1913 + 2u;
                    uint _1922 = _1913 + 3u;
                    uint _1927 = (_1172 + 48u) >> 2u;
                    uint _1930 = _1927 + 1u;
                    uint _1933 = _1927 + 2u;
                    _4369 = half3(half(float(_SplatSH._m0[_1933] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1933] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1933] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _4360 = half3(half(float(_SplatSH._m0[_1930] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1930] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1930] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _4351 = half3(half(float(_SplatSH._m0[_1927] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1927] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1927] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _4342 = half3(half(float(_SplatSH._m0[_1922] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1922] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1922] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _4333 = half3(half(float(_SplatSH._m0[_1919] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1919] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1919] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _4324 = half3(half(float(_SplatSH._m0[_1916] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1916] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1916] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _4315 = half3(half(float(_SplatSH._m0[_1913] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1913] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1913] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _4306 = half3(half(float(_SplatSH._m0[_1196] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1196] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1196] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _4298 = half3(half(float(_SplatSH._m0[_1193] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1193] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1193] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _4290 = half3(half(float(_SplatSH._m0[_1190] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1190] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1190] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _4282 = half3(half(float(_SplatSH._m0[_1187] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1187] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1187] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _4274 = half3(half(float(_SplatSH._m0[_1182] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1182] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1182] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _4266 = half3(half(float(_SplatSH._m0[_1179] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1179] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1179] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _4259 = half3(half(float(_SplatSH._m0[_1176] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1176] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1176] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                    _4252 = half3(half(float(_SplatSH._m0[_1173] & 2047u) * 0.000488519784994423389434814453125), half(float((_SplatSH._m0[_1173] >> 11u) & 1023u) * 0.000977517105638980865478515625), half(float((_SplatSH._m0[_1173] >> 21u) & 2047u) * 0.000488519784994423389434814453125));
                }
                else
                {
                    half3 _4253;
                    half3 _4260;
                    half3 _4267;
                    half3 _4275;
                    half3 _4283;
                    half3 _4291;
                    half3 _4299;
                    half3 _4307;
                    half3 _4316;
                    half3 _4325;
                    half3 _4334;
                    half3 _4343;
                    half3 _4352;
                    half3 _4361;
                    half3 _4370;
                    if (_879 == 3u)
                    {
                        half3 _4256;
                        half3 _4263;
                        half3 _4270;
                        if (_747 > 0u)
                        {
                            uint _2217 = _SplatSH._m0[_1173] >> 16u;
                            _4270 = half3(half(float(_SplatSH._m0[_1176] & 31u) * 0.0322580635547637939453125), half(float((_SplatSH._m0[_1176] >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_SplatSH._m0[_1176] >> 11u) & 31u) * 0.0322580635547637939453125));
                            _4263 = half3(half(float(_2217 & 31u) * 0.0322580635547637939453125), half(float((_2217 >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_2217 >> 11u) & 31u) * 0.0322580635547637939453125));
                            _4256 = half3(half(float(_SplatSH._m0[_1173] & 31u) * 0.0322580635547637939453125), half(float((_SplatSH._m0[_1173] >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_SplatSH._m0[_1173] >> 11u) & 31u) * 0.0322580635547637939453125));
                        }
                        else
                        {
                            _4270 = half3(half(0.0));
                            _4263 = half3(half(0.0));
                            _4256 = half3(half(0.0));
                        }
                        half3 _4277;
                        half3 _4285;
                        half3 _4293;
                        half3 _4301;
                        half3 _4309;
                        if (_747 > 1u)
                        {
                            uint _2255 = _SplatSH._m0[_1176] >> 16u;
                            uint _2290 = _SplatSH._m0[_1179] >> 16u;
                            uint _2325 = _SplatSH._m0[_1182] >> 16u;
                            _4309 = half3(half(float(_2325 & 31u) * 0.0322580635547637939453125), half(float((_2325 >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_2325 >> 11u) & 31u) * 0.0322580635547637939453125));
                            _4301 = half3(half(float(_SplatSH._m0[_1182] & 31u) * 0.0322580635547637939453125), half(float((_SplatSH._m0[_1182] >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_SplatSH._m0[_1182] >> 11u) & 31u) * 0.0322580635547637939453125));
                            _4293 = half3(half(float(_2290 & 31u) * 0.0322580635547637939453125), half(float((_2290 >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_2290 >> 11u) & 31u) * 0.0322580635547637939453125));
                            _4285 = half3(half(float(_SplatSH._m0[_1179] & 31u) * 0.0322580635547637939453125), half(float((_SplatSH._m0[_1179] >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_SplatSH._m0[_1179] >> 11u) & 31u) * 0.0322580635547637939453125));
                            _4277 = half3(half(float(_2255 & 31u) * 0.0322580635547637939453125), half(float((_2255 >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_2255 >> 11u) & 31u) * 0.0322580635547637939453125));
                        }
                        else
                        {
                            _4309 = half3(half(0.0));
                            _4301 = half3(half(0.0));
                            _4293 = half3(half(0.0));
                            _4285 = half3(half(0.0));
                            _4277 = half3(half(0.0));
                        }
                        half3 _4317;
                        half3 _4326;
                        half3 _4335;
                        half3 _4344;
                        half3 _4353;
                        half3 _4362;
                        half3 _4371;
                        if (_747 > 2u)
                        {
                            uint _2363 = _SplatSH._m0[_1187] >> 16u;
                            uint _2398 = _SplatSH._m0[_1190] >> 16u;
                            uint _2433 = _SplatSH._m0[_1193] >> 16u;
                            _4371 = half3(half(float(_SplatSH._m0[_1196] & 31u) * 0.0322580635547637939453125), half(float((_SplatSH._m0[_1196] >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_SplatSH._m0[_1196] >> 11u) & 31u) * 0.0322580635547637939453125));
                            _4362 = half3(half(float(_2433 & 31u) * 0.0322580635547637939453125), half(float((_2433 >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_2433 >> 11u) & 31u) * 0.0322580635547637939453125));
                            _4353 = half3(half(float(_SplatSH._m0[_1193] & 31u) * 0.0322580635547637939453125), half(float((_SplatSH._m0[_1193] >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_SplatSH._m0[_1193] >> 11u) & 31u) * 0.0322580635547637939453125));
                            _4344 = half3(half(float(_2398 & 31u) * 0.0322580635547637939453125), half(float((_2398 >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_2398 >> 11u) & 31u) * 0.0322580635547637939453125));
                            _4335 = half3(half(float(_SplatSH._m0[_1190] & 31u) * 0.0322580635547637939453125), half(float((_SplatSH._m0[_1190] >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_SplatSH._m0[_1190] >> 11u) & 31u) * 0.0322580635547637939453125));
                            _4326 = half3(half(float(_2363 & 31u) * 0.0322580635547637939453125), half(float((_2363 >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_2363 >> 11u) & 31u) * 0.0322580635547637939453125));
                            _4317 = half3(half(float(_SplatSH._m0[_1187] & 31u) * 0.0322580635547637939453125), half(float((_SplatSH._m0[_1187] >> 5u) & 63u) * 0.01587301678955554962158203125), half(float((_SplatSH._m0[_1187] >> 11u) & 31u) * 0.0322580635547637939453125));
                        }
                        else
                        {
                            _4371 = half3(half(0.0));
                            _4362 = half3(half(0.0));
                            _4353 = half3(half(0.0));
                            _4344 = half3(half(0.0));
                            _4335 = half3(half(0.0));
                            _4326 = half3(half(0.0));
                            _4317 = half3(half(0.0));
                        }
                        _4370 = _4371;
                        _4361 = _4362;
                        _4352 = _4353;
                        _4343 = _4344;
                        _4334 = _4335;
                        _4325 = _4326;
                        _4316 = _4317;
                        _4307 = _4309;
                        _4299 = _4301;
                        _4291 = _4293;
                        _4283 = _4285;
                        _4275 = _4277;
                        _4267 = _4270;
                        _4260 = _4263;
                        _4253 = _4256;
                    }
                    else
                    {
                        _4370 = half3(half(0.0));
                        _4361 = half3(half(0.0));
                        _4352 = half3(half(0.0));
                        _4343 = half3(half(0.0));
                        _4334 = half3(half(0.0));
                        _4325 = half3(half(0.0));
                        _4316 = half3(half(0.0));
                        _4307 = half3(half(0.0));
                        _4299 = half3(half(0.0));
                        _4291 = half3(half(0.0));
                        _4283 = half3(half(0.0));
                        _4275 = half3(half(0.0));
                        _4267 = half3(half(0.0));
                        _4260 = half3(half(0.0));
                        _4253 = half3(half(0.0));
                    }
                    _4369 = _4370;
                    _4360 = _4361;
                    _4351 = _4352;
                    _4342 = _4343;
                    _4333 = _4334;
                    _4324 = _4325;
                    _4315 = _4316;
                    _4306 = _4307;
                    _4298 = _4299;
                    _4290 = _4291;
                    _4282 = _4283;
                    _4274 = _4275;
                    _4266 = _4267;
                    _4259 = _4260;
                    _4252 = _4253;
                }
                _4368 = _4369;
                _4359 = _4360;
                _4350 = _4351;
                _4341 = _4342;
                _4332 = _4333;
                _4323 = _4324;
                _4314 = _4315;
                _4305 = _4306;
                _4297 = _4298;
                _4289 = _4290;
                _4281 = _4282;
                _4273 = _4274;
                _4265 = _4266;
                _4258 = _4259;
                _4251 = _4252;
            }
            _4367 = _4368;
            _4358 = _4359;
            _4349 = _4350;
            _4340 = _4341;
            _4331 = _4332;
            _4322 = _4323;
            _4313 = _4314;
            _4304 = _4305;
            _4296 = _4297;
            _4288 = _4289;
            _4280 = _4281;
            _4272 = _4273;
            _4264 = _4265;
            _4257 = _4258;
            _4250 = _4251;
        }
        half3 _4374;
        half3 _4376;
        half3 _4378;
        half3 _4380;
        half3 _4382;
        half3 _4384;
        half3 _4386;
        half3 _4388;
        half3 _4391;
        half3 _4394;
        half3 _4397;
        half3 _4400;
        half3 _4403;
        half3 _4407;
        half3 _4411;
        if ((_879 > 0u) && (_879 <= 3u))
        {
            half3 _4406;
            half3 _4410;
            half3 _4414;
            if (_747 > 0u)
            {
                _4414 = mix(_852, _865, _4250);
                _4410 = mix(_852, _865, _4257);
                _4406 = mix(_852, _865, _4264);
            }
            else
            {
                _4414 = _4250;
                _4410 = _4257;
                _4406 = _4264;
            }
            half3 _4390;
            half3 _4393;
            half3 _4396;
            half3 _4399;
            half3 _4402;
            if (_747 > 1u)
            {
                _4402 = mix(_852, _865, _4272);
                _4399 = mix(_852, _865, _4280);
                _4396 = mix(_852, _865, _4288);
                _4393 = mix(_852, _865, _4296);
                _4390 = mix(_852, _865, _4304);
            }
            else
            {
                _4402 = _4272;
                _4399 = _4280;
                _4396 = _4288;
                _4393 = _4296;
                _4390 = _4304;
            }
            half3 _4375;
            half3 _4377;
            half3 _4379;
            half3 _4381;
            half3 _4383;
            half3 _4385;
            half3 _4387;
            if (_747 > 2u)
            {
                _4387 = mix(_852, _865, _4313);
                _4385 = mix(_852, _865, _4322);
                _4383 = mix(_852, _865, _4331);
                _4381 = mix(_852, _865, _4340);
                _4379 = mix(_852, _865, _4349);
                _4377 = mix(_852, _865, _4358);
                _4375 = mix(_852, _865, _4367);
            }
            else
            {
                _4387 = _4313;
                _4385 = _4322;
                _4383 = _4331;
                _4381 = _4340;
                _4379 = _4349;
                _4377 = _4358;
                _4375 = _4367;
            }
            _4411 = _4414;
            _4407 = _4410;
            _4403 = _4406;
            _4400 = _4402;
            _4397 = _4399;
            _4394 = _4396;
            _4391 = _4393;
            _4388 = _4390;
            _4386 = _4387;
            _4384 = _4385;
            _4382 = _4383;
            _4380 = _4381;
            _4378 = _4379;
            _4376 = _4377;
            _4374 = _4375;
        }
        else
        {
            _4411 = _4250;
            _4407 = _4257;
            _4403 = _4264;
            _4400 = _4272;
            _4397 = _4280;
            _4394 = _4288;
            _4391 = _4296;
            _4388 = _4304;
            _4386 = _4313;
            _4384 = _4322;
            _4382 = _4331;
            _4380 = _4340;
            _4378 = _4349;
            _4376 = _4358;
            _4374 = _4367;
        }
        float _2527 = -_3934.x;
        float _2532 = -_3934.z;
        if (_1133 < half(0.0039215087890625))
        {
            _VisibleMasks._m0[gl_GlobalInvocationID.x] = 0u;
            _SplatProjData0._m0[gl_GlobalInvocationID.x] = SplatProjData{ float4(0.0), half4(half(0.0)), half2(half(0.0)), half2(half(0.0)) };
            _SplatProjData1._m0[gl_GlobalInvocationID.x] = SplatProjData{ float4(0.0), half4(half(0.0)), half2(half(0.0)), half2(half(0.0)) };
            break;
        }
        _VisibleMasks._m0[gl_GlobalInvocationID.x] = 1u;
        float3x3 _2543 = transpose(transpose(float3x3(_564, _568, _569 / float3(length(_569) + 9.9999999600419720025001879548654e-13))));
        float _2563 = _2532 * _2532;
        float _2566 = _3934.w * _2532;
        float _2570 = _3934.w * _3934.y;
        float _2579 = _3934.w * _2527;
        float3x3 _2598 = (float3x3(float3(_1114.x, 0.0, 0.0), float3(0.0, _1114.y, 0.0), float3(0.0, 0.0, _1114.z)) * float3x3(float3(fma(-2.0, fma(_3934.y, _3934.y, _2563), 1.0), 2.0 * fma(_2527, _3934.y, -_2566), 2.0 * fma(_2527, _2532, _2570)), float3(2.0 * fma(_2527, _3934.y, _2566), fma(-2.0, fma(_2527, _2527, _2563), 1.0), 2.0 * fma(_3934.y, _2532, -_2579)), float3(2.0 * fma(_2527, _2532, -_2570), 2.0 * fma(_3934.y, _2532, _2579), fma(-2.0, fma(_2527, _2527, _3934.y * _3934.y), 1.0)))) * (float3x3(float3(0.0, 0.0, 1.0), float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0)) * float3x3(float3(_555.x, _620.x, _627.x), float3(_555.y, _620.y, _627.y), float3(_555.z, _620.z, _627.z)));
        float3x3 _2600 = transpose(_2598) * _2598;
        float3 _2607 = _2600[0];
        float _2609 = _2600[1].y;
        float _2611 = _2600[1].z;
        float _2613 = _2600[2].z;
        _PointDistances._m0[gl_GlobalInvocationID.x] = _669;
        if (_703)
        {
            float4x4 _2630 = Constants._MatrixV[0] * Constants._MatrixM;
            float4 _2640 = _2630 * _656;
            float _2657 = _2640.z;
            float _2668 = _2640.z;
            float _2681 = (Constants._VecScreenParams.x * Constants._MatrixP[0][0][0]) * 0.5;
            float _2683 = _2640.z;
            float _2684 = _2681 / _2683;
            float _2693 = _2683 * _2683;
            float3x3 _2718 = float3x3(float4(_2630[0][0], _2630[1][0], _2630[2][0], _2630[3][0]).xyz, float4(_2630[0][1], _2630[1][1], _2630[2][1], _2630[3][1]).xyz, float4(_2630[0][2], _2630[1][2], _2630[2][2], _2630[3][2]).xyz) * float3x3(float3(_2684, 0.0, (-(_2681 * (fast::clamp(_2640.x / _2657, (-1.2999999523162841796875) / Constants._MatrixP[0][0][0], 1.2999999523162841796875 / Constants._MatrixP[0][0][0]) * _2657))) / _2693), float3(0.0, _2684, (-(_2681 * (fast::clamp(_2640.y / _2668, (-1.2999999523162841796875) / Constants._MatrixP[0][0][0], 1.2999999523162841796875 / Constants._MatrixP[0][0][0]) * _2668))) / _2693), float3(0.0));
            float3x3 _2743 = (transpose(_2718) * float3x3(_2607, float3(_2607.y, _2609, _2611), float3(_2607.z, _2611, _2613))) * _2718;
            float _2745 = _2743[0].x;
            float _2747 = _2743[1].y;
            float _2749 = _2743[0].y;
            float _2753 = -(_2749 * _2749);
            float _2755 = precise::max(9.9999999747524270787835121154785e-07, fma(_2745, _2747, _2753));
            float _2758 = _2745 + 0.100000001490116119384765625;
            float _2769 = precise::max(9.9999999747524270787835121154785e-07, fma(_2758, _2747 + 0.100000001490116119384765625, _2753));
            float _2785 = _2743[1].y + 0.100000001490116119384765625;
            float _2789 = _2743[0].y;
            float _2800 = _2758 + _2785;
            float _2804 = length(float2((_2758 - _2785) * 0.5, _2789));
            float _2805 = fma(0.5, _2800, _2804);
            float2 _2811 = fast::normalize(float2(_2789, _2805 - _2758));
            float _2814 = -_2811.y;
            float2 _3839 = _2811;
            _3839.y = _2814;
            float3 _2844 = ((Constants._CameraPosWS[0].xyz - _658) * float3x3(float4(Constants._MatrixWorldToObject[0][0], Constants._MatrixWorldToObject[1][0], Constants._MatrixWorldToObject[2][0], Constants._MatrixWorldToObject[3][0]).xyz, float4(Constants._MatrixWorldToObject[0][1], Constants._MatrixWorldToObject[1][1], Constants._MatrixWorldToObject[2][1], Constants._MatrixWorldToObject[3][1]).xyz, float4(Constants._MatrixWorldToObject[0][2], Constants._MatrixWorldToObject[1][2], Constants._MatrixWorldToObject[2][2], Constants._MatrixWorldToObject[3][2]).xyz)) * _2543;
            _2844.y = -_2844.y;
            int _2852 = int(_747);
            half3 _2853 = half3(fast::normalize(_2844)) * half(-1.0);
            half _2855 = _2853.x;
            half _2857 = _2853.y;
            half _2859 = _2853.z;
            half3 _2992;
            if (_2852 >= 1)
            {
                half3 _2873 = half3(float3(_1121.xyz) + (float3((((-_4411) * _2857) + (_4407 * _2859)) - (_4403 * _2855)) * 0.4886024892330169677734375));
                half3 _2991;
                if (_2852 >= 2)
                {
                    half _2877 = _2855 * _2855;
                    half _2878 = _2857 * _2857;
                    half _2879 = _2859 * _2859;
                    float _2884 = float(_2855 * _2857);
                    half _2893 = -_2877;
                    half _2895 = -_2857;
                    half _2907 = -_2878;
                    float _2909 = float(fma(_2855, _2855, _2907));
                    half3 _2915 = half3(float3(_2873) + (((((float3(_4400) * (1.092548370361328125 * _2884)) + (float3(_4397) * ((-1.092548370361328125) * float(_2857 * _2859)))) + (float3(_4394) * (0.315391600131988525390625 * float(fma(_2895, _2857, fma(half(2.0), _2879, _2893)))))) + (float3(_4391) * ((-1.092548370361328125) * float(_2855 * _2859)))) + (float3(_4388) * (0.5462741851806640625 * _2909))));
                    half3 _2990;
                    if (_2852 >= 3)
                    {
                        float _2920 = float(_2857);
                        float _2930 = float(_2859);
                        float _2941 = float(fma(_2895, _2857, fma(half(4.0), _2879, _2893)));
                        float _2958 = float(_2855);
                        _2990 = half3(float3(_2915) + (((((((float3(_4386) * (((-0.590043604373931884765625) * _2920) * float(fma(half(3.0), _2877, _2907)))) + (float3(_4384) * ((2.8906114101409912109375 * _2884) * _2930))) + (float3(_4382) * (((-0.4570457935333251953125) * _2920) * _2941))) + (float3(_4380) * ((0.3731763064861297607421875 * _2930) * float(fma(-half(3.0), _2878, fma(half(2.0), _2879, -(half(3.0) * _2877))))))) + (float3(_4378) * (((-0.4570457935333251953125) * _2958) * _2941))) + (float3(_4376) * ((1.44530570507049560546875 * _2930) * _2909))) + (float3(_4374) * (((-0.590043604373931884765625) * _2958) * float(fma(_2855, _2855, -(half(3.0) * _2878)))))));
                    }
                    else
                    {
                        _2990 = _2915;
                    }
                    _2991 = _2990;
                }
                else
                {
                    _2991 = _2873;
                }
                _2992 = _2991;
            }
            else
            {
                _2992 = _1121.xyz;
            }
            half3 _2998 = half3(float3(max(_2992, half3(half(0.0))).xyz) * _619);
            half4 _2999 = half4(_2998.x, _2998.y, _2998.z, _3919.w);
            _2999.w = half(precise::min(float(_1133) * (((_2755 <= 9.9999999747524270787835121154785e-07) || (_2769 <= 9.9999999747524270787835121154785e-07)) ? 0.0 : sqrt((_2755 / (_2769 + 9.9999999747524270787835121154785e-07)) + 9.9999999747524270787835121154785e-07)), 65000.0));
            _SplatProjData0._m0[gl_GlobalInvocationID.x] = SplatProjData{ _683, _2999, half2(_3839 * precise::min(sqrt(2.0 * _2805), 4096.0)), half2(float2(_2814, -_2811.x) * precise::min(sqrt(2.0 * precise::max(fma(0.5, _2800, -_2804), 0.100000001490116119384765625)), 4096.0)) };
        }
        else
        {
            _SplatProjData0._m0[gl_GlobalInvocationID.x] = SplatProjData{ float4(0.0), half4(half(0.0)), half2(half(0.0)), half2(half(0.0)) };
        }
        if (_735)
        {
            float4x4 _3023 = Constants._MatrixV[1] * Constants._MatrixM;
            float4 _3033 = _3023 * _656;
            float _3050 = _3033.z;
            float _3061 = _3033.z;
            float _3074 = (Constants._VecScreenParams.x * Constants._MatrixP[1][0][0]) * 0.5;
            float _3076 = _3033.z;
            float _3077 = _3074 / _3076;
            float _3086 = _3076 * _3076;
            float3x3 _3111 = float3x3(float4(_3023[0][0], _3023[1][0], _3023[2][0], _3023[3][0]).xyz, float4(_3023[0][1], _3023[1][1], _3023[2][1], _3023[3][1]).xyz, float4(_3023[0][2], _3023[1][2], _3023[2][2], _3023[3][2]).xyz) * float3x3(float3(_3077, 0.0, (-(_3074 * (fast::clamp(_3033.x / _3050, (-1.2999999523162841796875) / Constants._MatrixP[1][0][0], 1.2999999523162841796875 / Constants._MatrixP[1][0][0]) * _3050))) / _3086), float3(0.0, _3077, (-(_3074 * (fast::clamp(_3033.y / _3061, (-1.2999999523162841796875) / Constants._MatrixP[1][0][0], 1.2999999523162841796875 / Constants._MatrixP[1][0][0]) * _3061))) / _3086), float3(0.0));
            float3x3 _3136 = (transpose(_3111) * float3x3(_2607, float3(_2607.y, _2609, _2611), float3(_2607.z, _2611, _2613))) * _3111;
            float _3138 = _3136[0].x;
            float _3140 = _3136[1].y;
            float _3142 = _3136[0].y;
            float _3146 = -(_3142 * _3142);
            float _3148 = precise::max(9.9999999747524270787835121154785e-07, fma(_3138, _3140, _3146));
            float _3151 = _3138 + 0.100000001490116119384765625;
            float _3162 = precise::max(9.9999999747524270787835121154785e-07, fma(_3151, _3140 + 0.100000001490116119384765625, _3146));
            float _3178 = _3136[1].y + 0.100000001490116119384765625;
            float _3182 = _3136[0].y;
            float _3193 = _3151 + _3178;
            float _3197 = length(float2((_3151 - _3178) * 0.5, _3182));
            float _3198 = fma(0.5, _3193, _3197);
            float2 _3204 = fast::normalize(float2(_3182, _3198 - _3151));
            float _3207 = -_3204.y;
            float2 _3906 = _3204;
            _3906.y = _3207;
            float3 _3237 = ((Constants._CameraPosWS[1].xyz - _658) * float3x3(float4(Constants._MatrixWorldToObject[0][0], Constants._MatrixWorldToObject[1][0], Constants._MatrixWorldToObject[2][0], Constants._MatrixWorldToObject[3][0]).xyz, float4(Constants._MatrixWorldToObject[0][1], Constants._MatrixWorldToObject[1][1], Constants._MatrixWorldToObject[2][1], Constants._MatrixWorldToObject[3][1]).xyz, float4(Constants._MatrixWorldToObject[0][2], Constants._MatrixWorldToObject[1][2], Constants._MatrixWorldToObject[2][2], Constants._MatrixWorldToObject[3][2]).xyz)) * _2543;
            _3237.y = -_3237.y;
            int _3245 = int(_747);
            half3 _3246 = half3(fast::normalize(_3237)) * half(-1.0);
            half _3248 = _3246.x;
            half _3250 = _3246.y;
            half _3252 = _3246.z;
            half3 _3385;
            if (_3245 >= 1)
            {
                half3 _3266 = half3(float3(_1121.xyz) + (float3((((-_4411) * _3250) + (_4407 * _3252)) - (_4403 * _3248)) * 0.4886024892330169677734375));
                half3 _3384;
                if (_3245 >= 2)
                {
                    half _3270 = _3248 * _3248;
                    half _3271 = _3250 * _3250;
                    half _3272 = _3252 * _3252;
                    float _3277 = float(_3248 * _3250);
                    half _3286 = -_3270;
                    half _3288 = -_3250;
                    half _3300 = -_3271;
                    float _3302 = float(fma(_3248, _3248, _3300));
                    half3 _3308 = half3(float3(_3266) + (((((float3(_4400) * (1.092548370361328125 * _3277)) + (float3(_4397) * ((-1.092548370361328125) * float(_3250 * _3252)))) + (float3(_4394) * (0.315391600131988525390625 * float(fma(_3288, _3250, fma(half(2.0), _3272, _3286)))))) + (float3(_4391) * ((-1.092548370361328125) * float(_3248 * _3252)))) + (float3(_4388) * (0.5462741851806640625 * _3302))));
                    half3 _3383;
                    if (_3245 >= 3)
                    {
                        float _3313 = float(_3250);
                        float _3323 = float(_3252);
                        float _3334 = float(fma(_3288, _3250, fma(half(4.0), _3272, _3286)));
                        float _3351 = float(_3248);
                        _3383 = half3(float3(_3308) + (((((((float3(_4386) * (((-0.590043604373931884765625) * _3313) * float(fma(half(3.0), _3270, _3300)))) + (float3(_4384) * ((2.8906114101409912109375 * _3277) * _3323))) + (float3(_4382) * (((-0.4570457935333251953125) * _3313) * _3334))) + (float3(_4380) * ((0.3731763064861297607421875 * _3323) * float(fma(-half(3.0), _3271, fma(half(2.0), _3272, -(half(3.0) * _3270))))))) + (float3(_4378) * (((-0.4570457935333251953125) * _3351) * _3334))) + (float3(_4376) * ((1.44530570507049560546875 * _3323) * _3302))) + (float3(_4374) * (((-0.590043604373931884765625) * _3351) * float(fma(_3248, _3248, -(half(3.0) * _3271)))))));
                    }
                    else
                    {
                        _3383 = _3308;
                    }
                    _3384 = _3383;
                }
                else
                {
                    _3384 = _3266;
                }
                _3385 = _3384;
            }
            else
            {
                _3385 = _1121.xyz;
            }
            half3 _3391 = half3(float3(max(_3385, half3(half(0.0))).xyz) * _619);
            half4 _3392 = half4(_3391.x, _3391.y, _3391.z, _3919.w);
            _3392.w = half(precise::min(float(_1133) * (((_3148 <= 9.9999999747524270787835121154785e-07) || (_3162 <= 9.9999999747524270787835121154785e-07)) ? 0.0 : sqrt((_3148 / (_3162 + 9.9999999747524270787835121154785e-07)) + 9.9999999747524270787835121154785e-07)), 65000.0));
            _SplatProjData1._m0[gl_GlobalInvocationID.x] = SplatProjData{ _715, _3392, half2(_3906 * precise::min(sqrt(2.0 * _3198), 4096.0)), half2(float2(_3207, -_3204.x) * precise::min(sqrt(2.0 * precise::max(fma(0.5, _3193, -_3197), 0.100000001490116119384765625)), 4096.0)) };
        }
        else
        {
            _SplatProjData1._m0[gl_GlobalInvocationID.x] = SplatProjData{ float4(0.0), half4(half(0.0)), half2(half(0.0)), half2(half(0.0)) };
        }
        break;
    } while(false);
}


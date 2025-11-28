#pragma once

#include <string>
#include <memory>

#include <Metal/Metal.hpp>
#include <simd/simd.h>
#include "json.hpp"

#include "GaussianAsset.hpp"

namespace hrm
{

class Mesh;
class Shader;
class Graphics;
class Renderer;
class AnimationPlayer;

simd::float3 TryParseFloat3(const nlohmann::json& json);
simd::float4 TryParseFloat4(const nlohmann::json& json);
simd::float4x4 TryParseFloat4x4(const nlohmann::json& json);

std::vector<uint8_t> ReadBinaryFile(const std::string& absPath);

class AssetLoader
{
public:
    static std::unique_ptr<Shader> LoadShader(MTL::Device* device, const std::string& shaderPath, const std::string& entryPoint);

    static std::unique_ptr<GaussianAsset> LoadGaussianAsset(Renderer* pRenderer, const std::string& packPath, const std::string& filePath);

    static void LoadAnimationCurvesToPlayer(AnimationPlayer* pPlayer, const std::string& packPath, const std::string& filePath);
};

} // namespace hrm

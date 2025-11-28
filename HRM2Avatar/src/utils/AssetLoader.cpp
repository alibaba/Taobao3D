#include "AssetLoader.hpp"

#include <unordered_map>
#include <string>
#include <chrono>
#include <fstream>

#include "Mesh.hpp"
#include "Logging.hpp"
#include "MetalHelper.hpp"
#include "Shader.hpp"
#include "Buffer.hpp"
#include "AnimationPlayer.hpp"

namespace
{
template<typename T>
void AppendDataToBuffer(std::vector<uint8_t>& buffer, const T& value) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
    buffer.insert(buffer.end(), bytes, bytes + sizeof(T));
}
}

namespace hrm
{

extern const std::string g_AssetBaseDir = GetBundleAssetsPath();

simd::float3 TryParseFloat3(const nlohmann::json& json)
{
    return simd_make_float3(json[0].get<float>(), json[1].get<float>(), json[2].get<float>());
}

simd::float4 TryParseFloat4(const nlohmann::json& json)
{
    return simd_make_float4(json[0].get<float>(), json[1].get<float>(), json[2].get<float>(), json[3].get<float>());
}

simd::float4x4 TryParseFloat4x4(const nlohmann::json& json)
{
    return simd::transpose(simd::float4x4{
        simd_make_float4(json[0].get<float>(), json[1].get<float>(), json[2].get<float>(), json[3].get<float>()),
        simd_make_float4(json[4].get<float>(), json[5].get<float>(), json[6].get<float>(), json[7].get<float>()),
        simd_make_float4(json[8].get<float>(), json[9].get<float>(), json[10].get<float>(), json[11].get<float>()),
        simd_make_float4(json[12].get<float>(), json[13].get<float>(), json[14].get<float>(), json[15].get<float>())
    });
}

std::vector<uint8_t> ReadBinaryFile(const std::string& absPath)
{
    std::ifstream file(absPath, std::ios::binary);
    if (!file.is_open())
    {
        LOG_ERROR("Failed to read binary file: %s", absPath.c_str());
        return {};
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size);
    file.close();
    return data;
}

std::unique_ptr<Shader> AssetLoader::LoadShader(MTL::Device* device, const std::string& shaderPath, const std::string& entryPoint)
{
    const auto tStart = std::chrono::steady_clock::now();

    const std::string fullPath = g_AssetBaseDir + shaderPath;

    std::ifstream file(fullPath);
    if (!file.is_open())
    {
        LOG_ERROR("Failed to load shader: %s", fullPath.c_str());
        return nullptr;
    }
    
    ShaderDesc desc;
    desc.sourceCode = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    desc.entryPoint = entryPoint;
    desc.name = shaderPath;

    auto shader = std::make_unique<Shader>(device, desc);
    if (shader == nullptr)
    {
        LOG_ERROR("Failed to load shader: %s", fullPath.c_str());
        return nullptr;
    }

    const auto tEnd = std::chrono::steady_clock::now();
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count();
    LOG_INFO("Loaded shader %s in %d ms", fullPath.c_str(), elapsedMs);

    return shader;
}

std::unique_ptr<GaussianAsset> AssetLoader::LoadGaussianAsset(Renderer* pRenderer, const std::string& packPath, const std::string& filePath)
{
    std::string pachDir = packPath.back() != '/' ? packPath + '/' : packPath;
    const std::string fullPath = pachDir + filePath;
    std::ifstream gaussianFile(fullPath);
    if (!gaussianFile.is_open())
    {
        LOG_ERROR("Failed to load gaussian asset: %s", fullPath.c_str());
        return nullptr;
    }
    nlohmann::json gaussianDesc = nlohmann::json::parse(gaussianFile);

    auto gaussianAsset = std::make_unique<GaussianAsset>(pRenderer);
    gaussianAsset->SetSplatCount(gaussianDesc["splat_count"].get<uint32_t>());
    gaussianAsset->SetPositionFormat((VectorFormat)gaussianDesc["pos_format"].get<uint32_t>());
    gaussianAsset->SetScaleFormat((VectorFormat)gaussianDesc["scale_format"].get<uint32_t>());
    gaussianAsset->SetSHFormat((SHFormat)gaussianDesc["sh_format"].get<uint32_t>());
    gaussianAsset->SetColorTextureSize(gaussianDesc["color_width"].get<uint32_t>(), gaussianDesc["color_height"].get<uint32_t>());

    const std::string& dataPath = gaussianDesc["data"].get<std::string>();
    const std::vector<uint8_t> data = ReadBinaryFile(pachDir + dataPath);

    uint32_t offset = gaussianDesc["pos_data"]["offset"].get<uint32_t>();
    uint32_t size = gaussianDesc["pos_data"]["size"].get<uint32_t>();
    gaussianAsset->SetPosData(data.data() + offset, size);

    offset = gaussianDesc["color_data"]["offset"].get<uint32_t>();
    size = gaussianDesc["color_data"]["size"].get<uint32_t>();
    gaussianAsset->SetColorData(data.data() + offset, size);
    
    offset = gaussianDesc["other_data"]["offset"].get<uint32_t>();
    size = gaussianDesc["other_data"]["size"].get<uint32_t>();
    gaussianAsset->SetOtherData(data.data() + offset, size);

    offset = gaussianDesc["sh_data"]["offset"].get<uint32_t>();
    size = gaussianDesc["sh_data"]["size"].get<uint32_t>();
    gaussianAsset->SetSHData(data.data() + offset, size);

    offset = gaussianDesc["chunk_data"]["offset"].get<uint32_t>();
    size = gaussianDesc["chunk_data"]["size"].get<uint32_t>();
    gaussianAsset->SetChunkData(data.data() + offset, size);

    offset = gaussianDesc["idx_data"]["offset"].get<uint32_t>();
    size = gaussianDesc["idx_data"]["size"].get<uint32_t>();
    gaussianAsset->SetIdxData(data.data() + offset, size);

    offset = gaussianDesc["gaussian_prop_data"]["offset"].get<uint32_t>();
    size = gaussianDesc["gaussian_prop_data"]["size"].get<uint32_t>();
    gaussianAsset->SetGaussianPropData(data.data() + offset, size);

    offset = gaussianDesc["face_property_data"]["offset"].get<uint32_t>();
    size = gaussianDesc["face_property_data"]["size"].get<uint32_t>();
    gaussianAsset->SetFacePropData(data.data() + offset, size);

    gaussianFile.close();

    return gaussianAsset;
}

void AssetLoader::LoadAnimationCurvesToPlayer(AnimationPlayer* pPlayer, const std::string& packPath, const std::string& filePath)
{
    std::string pachDir = packPath.back() != '/' ? packPath + '/' : packPath;
    const std::string fullPath = pachDir + filePath;
    std::ifstream animationFile(fullPath);
    if (!animationFile.is_open())
    {
        LOG_ERROR("Failed to load animation curves: %s", fullPath.c_str());
        return;
    }
    nlohmann::json animationDesc = nlohmann::json::parse(animationFile);

    std::vector<KeyFrame<float>> floatKeyFrames;
    std::vector<KeyFrame<simd::float3>> float3KeyFrames;
    std::vector<KeyFrame<simd::float4>> float4KeyFrames;
    for (const auto& curveDesc : animationDesc)
    {
        const std::string& target = curveDesc["target"].get<std::string>();
        const std::string& type = curveDesc["type"].get<std::string>();
        const nlohmann::json& keyFrames = curveDesc["keyframes"];
        const nlohmann::json& times = keyFrames["times"];
        const nlohmann::json& values = keyFrames["values"];
        const nlohmann::json& inTangents = keyFrames["inTangents"];
        const nlohmann::json& outTangents = keyFrames["outTangents"];
        uint32_t numFrames = (uint32_t)times.size();
        if (type == "local_rotation")
        {
            AnimationCurve<simd::float4> float4Curve;
            float4Curve.SetTarget(target);
            float4Curve.SetAttributeType(AttributeType::Rotation);

            float4KeyFrames.resize(numFrames);
            for (uint32_t i = 0; i < numFrames; ++i)
            {
                float4KeyFrames[i].time = times[i].get<float>();
                float4KeyFrames[i].value = TryParseFloat4(values[i]);
                float4KeyFrames[i].inTangent = TryParseFloat4(inTangents[i]);
                float4KeyFrames[i].outTangent = TryParseFloat4(outTangents[i]);
            }

            float4Curve.SetKeyFrames(float4KeyFrames);
            pPlayer->AddAnimationCurve(std::move(float4Curve));
        }
        else if (type == "local_position")
        {
            AnimationCurve<simd::float3> float3Curve;
            float3Curve.SetTarget(target);
            float3Curve.SetAttributeType(AttributeType::Position);
            
            float3KeyFrames.resize(numFrames);
            for (uint32_t i = 0; i < numFrames; ++i)
            {
                float3KeyFrames[i].time = times[i].get<float>();
                float3KeyFrames[i].value = TryParseFloat3(values[i]);
                float3KeyFrames[i].inTangent = TryParseFloat3(inTangents[i]);
                float3KeyFrames[i].outTangent = TryParseFloat3(outTangents[i]);
            }

            float3Curve.SetKeyFrames(float3KeyFrames);
            pPlayer->AddAnimationCurve(std::move(float3Curve));
        }
        else if (type == "blend_shape_weight")
        {
            AnimationCurve<float> floatCurve;
            floatCurve.SetTarget(target);
            floatCurve.SetAttributeType(AttributeType::BlendShapeWeight);

            floatKeyFrames.resize(numFrames);
            for (uint32_t i = 0; i < numFrames; ++i)
            {
                floatKeyFrames[i].time = times[i].get<float>();
                floatKeyFrames[i].value = values[i].get<float>();
                floatKeyFrames[i].inTangent = inTangents[i].get<float>();
                floatKeyFrames[i].outTangent = outTangents[i].get<float>();
            }

            floatCurve.SetKeyFrames(floatKeyFrames);
            pPlayer->AddAnimationCurve(std::move(floatCurve));
        }
        else
        {
            LOG_ERROR("Unsupported animation curve type: %s", type.c_str());
            continue;
        }
    }
    animationFile.close();
}
} // namespace hrm

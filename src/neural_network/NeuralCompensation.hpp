#pragma once

#include <memory>
#include <vector>
#include <string>

namespace MNN{
    class Interpreter;
    class Session;
    struct BackendConfig;
namespace Express{
    class Module;
    class Executor;
}
}

namespace hrm
{
class Renderer;

class MNNContext
{
    std::shared_ptr<MNN::Express::Module> m_Module;
};

class NeuralCompensation
{
public:
    NeuralCompensation(Renderer* r) {m_Renderer = r;};
    ~NeuralCompensation();
    void InitializeBodyPoseDeformModule(MNN::Express::Module* ptr) {m_BodyPoseDeformModule = ptr;};
    void InitializeBodyPoseShadowModule(MNN::Express::Module* ptr) {m_BodyPoseShadowModule = ptr;};
    void SetNetInputBoneNames(std::vector<std::string>&& inputBoneNames){m_NetInputBoneNames = inputBoneNames;};
    void UpdateFrame();
    
private:
    MNN::Express::Module* m_BodyPoseDeformModule;
    MNN::Express::Module* m_BodyPoseShadowModule;
    
    Renderer* m_Renderer;
    std::vector<std::string> m_NetInputBoneNames;
};
}

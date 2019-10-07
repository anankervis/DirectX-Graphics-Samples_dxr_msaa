//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#define NOMINMAX

#include "d3d12.h"
#include "d3d12video.h"
#include <d3d12.h>
#include "dxgi1_3.h"
#include "GameCore.h"
#include "GraphicsCore.h"
#include "CameraController.h"
#include "BufferManager.h"
#include "Camera.h"
#include "Model.h"
#include "GpuBuffer.h"
#include "CommandContext.h"
#include "SamplerManager.h"
#include "TemporalEffects.h"
#include "MotionBlur.h"
#include "DepthOfField.h"
#include "PostEffects.h"
#include "SSAO.h"
#include "FXAA.h"
#include "SystemTime.h"
#include "TextRenderer.h"
#include "ShadowCamera.h"
#include "ParticleEffectManager.h"
#include "GameInput.h"

#include <atlbase.h>
#include "DXSampleHelper.h"

#include "CompiledShaders/ModelViewerVS.h"
#include "CompiledShaders/ModelViewerPS.h"
#include "CompiledShaders/BeamsLib.h"
#include "CompiledShaders/BeamsShadeQuads.h"
#include "CompiledShaders/RaysLib.h"

#include "Shaders/RayCommon.h"
#include "Shaders/Shading.h"

#include <ShellScalingAPI.h>
#pragma comment(lib, "Shcore.lib")

#define ALIGN(alignment, num) ((((num) + alignment - 1) / alignment) * alignment)

using namespace GameCore;
using namespace Math;
using namespace Graphics;

BoolVar enableMsaa("Application/Raytracing/enableMsaa", true);

CComPtr<ID3D12Device5> g_pRaytracingDevice;

ByteAddressBuffer          g_shadeConstantBuffer;
ByteAddressBuffer          g_dynamicConstantBuffer;

D3D12_GPU_DESCRIPTOR_HANDLE g_GpuSceneMaterialSrvs[27];
D3D12_CPU_DESCRIPTOR_HANDLE g_SceneMeshInfo;
D3D12_CPU_DESCRIPTOR_HANDLE g_SceneIndices;

D3D12_GPU_DESCRIPTOR_HANDLE g_OutputUAV;
D3D12_GPU_DESCRIPTOR_HANDLE g_DepthAndNormalsTable;
D3D12_GPU_DESCRIPTOR_HANDLE g_SceneSrvs;

// we only need a single bottom-level BVH
constexpr uint32_t bvhBottomCount = 1;

struct BVH
{
    CComPtr<ID3D12Resource> top;
    CComPtr<ID3D12Resource> bottom;
};
BVH g_bvhTriangles;
BVH g_bvhAABBs;

DynamicCB           g_dynamicCb;
CComPtr<ID3D12RootSignature> g_GlobalRaytracingRootSignature;
CComPtr<ID3D12RootSignature> g_LocalRaytracingRootSignature;

RootSignature g_BeamPostRootSig;
ComputePSO g_BeamShadeQuadsPSO;

enum class RenderMode
{
    raster = 0,
    rays,
    beams,

    count,
};
const char* renderModeStr[] =
{
    "Raster",
    "Rays",
    "Beams",
};
EnumVar renderMode("Application/Raytracing/RenderMode", int(RenderMode::beams), int(RenderMode::count), renderModeStr);

const static UINT MaxRayRecursion = 1;

const static UINT c_NumCameraPositions = 5;

struct RaytracingDispatchRayInputs
{
    RaytracingDispatchRayInputs() {}
    RaytracingDispatchRayInputs(
        ID3D12Device5 &device,
        ID3D12StateObject *pPSO,
        void *pHitGroupShaderTable,
        UINT HitGroupStride,
        UINT HitGroupTableSize,
        LPCWSTR rayGenExportName,
        LPCWSTR missExportName) : m_pPSO(pPSO)
    {
        const UINT shaderTableSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
        ID3D12StateObjectProperties* stateObjectProperties = nullptr;
        ThrowIfFailed(pPSO->QueryInterface(IID_PPV_ARGS(&stateObjectProperties)));
        void *pRayGenShaderData = stateObjectProperties->GetShaderIdentifier(rayGenExportName);
        void *pMissShaderData = stateObjectProperties->GetShaderIdentifier(missExportName);

        m_HitGroupStride = HitGroupStride;

        // MiniEngine requires that all initial data be aligned to 16 bytes
        UINT alignment = 16;
        std::vector<BYTE> alignedShaderTableData(shaderTableSize + alignment - 1);
        BYTE *pAlignedShaderTableData = alignedShaderTableData.data() + ((UINT64)alignedShaderTableData.data() % alignment);
        memcpy(pAlignedShaderTableData, pRayGenShaderData, shaderTableSize);
        m_RayGenShaderTable.Create(L"Ray Gen Shader Table", 1, shaderTableSize, alignedShaderTableData.data());
        
        memcpy(pAlignedShaderTableData, pMissShaderData, shaderTableSize);
        m_MissShaderTable.Create(L"Miss Shader Table", 1, shaderTableSize, alignedShaderTableData.data());
        
        m_HitShaderTable.Create(L"Hit Shader Table", 1, HitGroupTableSize, pHitGroupShaderTable);
    }

    D3D12_DISPATCH_RAYS_DESC GetDispatchRayDesc(UINT DispatchWidth, UINT DispatchHeight)
    {
        D3D12_DISPATCH_RAYS_DESC dispatchRaysDesc = {};

        dispatchRaysDesc.RayGenerationShaderRecord.StartAddress = m_RayGenShaderTable.GetGpuVirtualAddress();
        dispatchRaysDesc.RayGenerationShaderRecord.SizeInBytes = m_RayGenShaderTable.GetBufferSize();
        dispatchRaysDesc.HitGroupTable.StartAddress = m_HitShaderTable.GetGpuVirtualAddress();
        dispatchRaysDesc.HitGroupTable.SizeInBytes = m_HitShaderTable.GetBufferSize();
        dispatchRaysDesc.HitGroupTable.StrideInBytes = m_HitGroupStride;
        dispatchRaysDesc.MissShaderTable.StartAddress = m_MissShaderTable.GetGpuVirtualAddress();
        dispatchRaysDesc.MissShaderTable.SizeInBytes = m_MissShaderTable.GetBufferSize();
        dispatchRaysDesc.MissShaderTable.StrideInBytes = dispatchRaysDesc.MissShaderTable.SizeInBytes; // Only one entry
        dispatchRaysDesc.Width = DispatchWidth;
        dispatchRaysDesc.Height = DispatchHeight;
        dispatchRaysDesc.Depth = 1;
        return dispatchRaysDesc;
    }

    UINT m_HitGroupStride;
    CComPtr<ID3D12StateObject> m_pPSO;
    ByteAddressBuffer   m_RayGenShaderTable;
    ByteAddressBuffer   m_MissShaderTable;
    ByteAddressBuffer   m_HitShaderTable;
};

RaytracingDispatchRayInputs g_RaytracingInputs_Ray;
RaytracingDispatchRayInputs g_RaytracingInputs_Beam;

struct MaterialRootConstant
{
    UINT MaterialID;
};

D3D12_CPU_DESCRIPTOR_HANDLE g_bvh_attributeSrvs[34];

class DxrMsaaDemo : public GameCore::IGameApp
{
public:

    DxrMsaaDemo() {}

    virtual void Startup() override;
    virtual void Cleanup() override;

    virtual void Update( float deltaT ) override;
    virtual void RenderScene() override;
    virtual void RenderUI(class GraphicsContext&) override;
    virtual void Raytrace(class GraphicsContext&);

    void SetCameraToPredefinedPosition(int cameraPosition);

private:

    void createBvh(const Model &model, BVH &bvh, bool triangles);

    void RenderObjects( GraphicsContext& Context, const Matrix4& ViewProjMat);
    void RaytraceDiffuse(GraphicsContext& context, const Math::Camera& camera, ColorBuffer& colorTarget);
    void RaytraceDiffuseBeams(GraphicsContext& context, const Math::Camera& camera, ColorBuffer& colorTarget);

    Camera m_Camera;
    std::auto_ptr<CameraController> m_CameraController;
    Matrix4 m_ViewProjMatrix;
    D3D12_VIEWPORT m_MainViewport;
    D3D12_RECT m_MainScissor;

    RootSignature m_RootSig;
    GraphicsPSO m_ModelPSO;

    D3D12_CPU_DESCRIPTOR_HANDLE m_DefaultSampler;

    Model m_Model;

    Vector3 m_SunDirection;

    struct CameraPosition
    {
        Vector3 position;
        float heading;
        float pitch;
    };

    CameraPosition m_CameraPosArray[c_NumCameraPositions];
    UINT m_CameraPosArrayCurrentPosition;
};

// Returns bool whether the device supports DirectX Raytracing tier.
inline bool IsDirectXRaytracingSupported(IDXGIAdapter1* adapter)
{
    ComPtr<ID3D12Device> testDevice;
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 featureSupportData = {};

    return SUCCEEDED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&testDevice)))
        && SUCCEEDED(testDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &featureSupportData, sizeof(featureSupportData)))
        && featureSupportData.RaytracingTier != D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
}

int wmain(int argc, wchar_t** argv)
{
    // disable scaling of the output window
    SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);

#if _DEBUG
    CComPtr<ID3D12Debug> debugInterface;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugInterface))))
    {
        debugInterface->EnableDebugLayer();
    }
#endif

    CComPtr<ID3D12Device> pDevice;
    CComPtr<IDXGIAdapter1> pAdapter;
    CComPtr<IDXGIFactory2> pFactory;
    CreateDXGIFactory2(0, IID_PPV_ARGS(&pFactory));
    bool validDeviceFound = false;
    for (uint32_t Idx = 0; !validDeviceFound && DXGI_ERROR_NOT_FOUND != pFactory->EnumAdapters1(Idx, &pAdapter); ++Idx)
    {
        DXGI_ADAPTER_DESC1 desc;
        pAdapter->GetDesc1(&desc);

        if (IsDirectXRaytracingSupported(pAdapter))
        {
            validDeviceFound = true;
        }
        pAdapter = nullptr;
    }

    s_EnableVSync.Decrement();
    TargetResolution = k1080p;
    g_DisplayWidth = 1920;
    g_DisplayHeight = 1080;
    GameCore::RunApplication(DxrMsaaDemo(), L"DxrMsaaDemo"); 
    return 0;
}

ExpVar m_SunLightIntensity("Application/Lighting/Sun Light Intensity", 4.0f, 0.0f, 16.0f, 0.1f);
ExpVar m_AmbientIntensity("Application/Lighting/Ambient Intensity", 0.1f, -16.0f, 16.0f, 0.1f);
NumVar m_SunOrientation("Application/Lighting/Sun Orientation", -0.5f, -100.0f, 100.0f, 0.1f );
NumVar m_SunInclination("Application/Lighting/Sun Inclination", 0.75f, 0.0f, 1.0f, 0.01f );

class DescriptorHeapStack
{
public:
    DescriptorHeapStack(ID3D12Device &device, UINT numDescriptors, D3D12_DESCRIPTOR_HEAP_TYPE type, UINT NodeMask) :
        m_device(device)
    {
        D3D12_DESCRIPTOR_HEAP_DESC desc = {};
        desc.NumDescriptors = numDescriptors;
        desc.Type = type;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        desc.NodeMask = NodeMask;
        device.CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_pDescriptorHeap));

        m_descriptorSize = device.GetDescriptorHandleIncrementSize(type);
        m_descriptorHeapCpuBase = m_pDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
    }

    ID3D12DescriptorHeap &GetDescriptorHeap() { return *m_pDescriptorHeap; }

    void AllocateDescriptor(_Out_ D3D12_CPU_DESCRIPTOR_HANDLE &cpuHandle, _Out_ UINT &descriptorHeapIndex)
    {
        descriptorHeapIndex = m_descriptorsAllocated;
        cpuHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(m_descriptorHeapCpuBase, descriptorHeapIndex, m_descriptorSize);
        m_descriptorsAllocated++;
    }

    UINT AllocateBufferSrv(_In_ ID3D12Resource &resource)
    {
        UINT descriptorHeapIndex;
        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle;
        AllocateDescriptor(cpuHandle, descriptorHeapIndex);
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.NumElements = (UINT)(resource.GetDesc().Width / sizeof(UINT32));
        srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

        m_device.CreateShaderResourceView(&resource, &srvDesc, cpuHandle);
        return descriptorHeapIndex;
    }

    UINT AllocateBufferUav(_In_ ID3D12Resource &resource)
    {
        UINT descriptorHeapIndex;
        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle;
        AllocateDescriptor(cpuHandle, descriptorHeapIndex);
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.NumElements = (UINT)(resource.GetDesc().Width / sizeof(UINT32));
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;

        m_device.CreateUnorderedAccessView(&resource, nullptr, &uavDesc, cpuHandle);
        return descriptorHeapIndex;
    }

    D3D12_GPU_DESCRIPTOR_HANDLE GetGpuHandle(UINT descriptorIndex)
    {
        return CD3DX12_GPU_DESCRIPTOR_HANDLE(m_pDescriptorHeap->GetGPUDescriptorHandleForHeapStart(), descriptorIndex, m_descriptorSize);
    }
private:
    ID3D12Device & m_device;
    CComPtr<ID3D12DescriptorHeap> m_pDescriptorHeap;
    UINT m_descriptorsAllocated = 0;
    UINT m_descriptorSize;
    D3D12_CPU_DESCRIPTOR_HANDLE m_descriptorHeapCpuBase;
};

std::unique_ptr<DescriptorHeapStack> g_pRaytracingDescriptorHeap;

StructuredBuffer    g_hitShaderMeshInfoBuffer;

static
void InitializeSceneInfo(
    const Model& model)
{
    //
    // Mesh info
    //
    std::vector<RayTraceMeshInfo>   meshInfoData(model.m_Header.meshCount);
    for (UINT i=0; i < model.m_Header.meshCount; ++i)
    {
        meshInfoData[i].m_indexOffsetBytes = model.m_pMesh[i].indexDataByteOffset;
        meshInfoData[i].m_uvAttributeOffsetBytes = model.m_pMesh[i].vertexDataByteOffset + model.m_pMesh[i].attrib[Model::attrib_texcoord0].offset;
        meshInfoData[i].m_normalAttributeOffsetBytes = model.m_pMesh[i].vertexDataByteOffset + model.m_pMesh[i].attrib[Model::attrib_normal].offset;
        meshInfoData[i].m_positionAttributeOffsetBytes = model.m_pMesh[i].vertexDataByteOffset + model.m_pMesh[i].attrib[Model::attrib_position].offset;
        meshInfoData[i].m_tangentAttributeOffsetBytes = model.m_pMesh[i].vertexDataByteOffset + model.m_pMesh[i].attrib[Model::attrib_tangent].offset;
        meshInfoData[i].m_bitangentAttributeOffsetBytes = model.m_pMesh[i].vertexDataByteOffset + model.m_pMesh[i].attrib[Model::attrib_bitangent].offset;
        meshInfoData[i].m_attributeStrideBytes = model.m_pMesh[i].vertexStride;
        meshInfoData[i].m_materialInstanceId = model.m_pMesh[i].materialIndex;
        ASSERT(meshInfoData[i].m_materialInstanceId < 27);
    }

    g_hitShaderMeshInfoBuffer.Create(L"RayTraceMeshInfo",
        (UINT)meshInfoData.size(),
        sizeof(meshInfoData[0]),
        meshInfoData.data());

    g_SceneIndices = model.m_IndexBuffer.GetSRV();
    g_SceneMeshInfo = g_hitShaderMeshInfoBuffer.GetSRV();
}

static
void InitializeViews(const Model& model)
{
    D3D12_CPU_DESCRIPTOR_HANDLE uavHandle;
    UINT uavDescriptorIndex;
    g_pRaytracingDescriptorHeap->AllocateDescriptor(uavHandle, uavDescriptorIndex);
    Graphics::g_Device->CopyDescriptorsSimple(1, uavHandle, g_SceneColorBuffer.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    g_OutputUAV = g_pRaytracingDescriptorHeap->GetGpuHandle(uavDescriptorIndex);

    {
        D3D12_CPU_DESCRIPTOR_HANDLE srvHandle;
        UINT srvDescriptorIndex;
        g_pRaytracingDescriptorHeap->AllocateDescriptor(srvHandle, srvDescriptorIndex);
        Graphics::g_Device->CopyDescriptorsSimple(1, srvHandle, g_SceneDepthBuffer.GetDepthSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        g_DepthAndNormalsTable = g_pRaytracingDescriptorHeap->GetGpuHandle(srvDescriptorIndex);
    }

    {
        D3D12_CPU_DESCRIPTOR_HANDLE srvHandle;
        UINT srvDescriptorIndex;
        g_pRaytracingDescriptorHeap->AllocateDescriptor(srvHandle, srvDescriptorIndex);
        Graphics::g_Device->CopyDescriptorsSimple(1, srvHandle, g_SceneMeshInfo, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        g_SceneSrvs = g_pRaytracingDescriptorHeap->GetGpuHandle(srvDescriptorIndex);

        UINT unused;
        g_pRaytracingDescriptorHeap->AllocateDescriptor(srvHandle, unused);
        Graphics::g_Device->CopyDescriptorsSimple(1, srvHandle, g_SceneIndices, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        g_pRaytracingDescriptorHeap->AllocateBufferSrv(*const_cast<ID3D12Resource*>(model.m_VertexBuffer.GetResource()));

        g_pRaytracingDescriptorHeap->AllocateDescriptor(srvHandle, unused);
        Graphics::g_Device->CopyDescriptorsSimple(1, srvHandle, g_ShadowBuffer.GetSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        g_pRaytracingDescriptorHeap->AllocateDescriptor(srvHandle, unused);
        Graphics::g_Device->CopyDescriptorsSimple(1, srvHandle, g_SSAOFullScreen.GetSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        for (UINT i = 0; i < model.m_Header.materialCount; i++)
        {
            UINT slot;
            g_pRaytracingDescriptorHeap->AllocateDescriptor(srvHandle, slot);
            Graphics::g_Device->CopyDescriptorsSimple(1, srvHandle, *model.GetSRVs(i), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            g_pRaytracingDescriptorHeap->AllocateDescriptor(srvHandle, unused);
            Graphics::g_Device->CopyDescriptorsSimple(1, srvHandle, model.GetSRVs(i)[3], D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            
            g_GpuSceneMaterialSrvs[i] = g_pRaytracingDescriptorHeap->GetGpuHandle(slot);
        }
    }
}

D3D12_STATE_SUBOBJECT CreateDxilLibrary(LPCWSTR entrypoint, const void *pShaderByteCode, SIZE_T bytecodeLength, D3D12_DXIL_LIBRARY_DESC &dxilLibDesc, D3D12_EXPORT_DESC &exportDesc)
{
    exportDesc = { entrypoint, nullptr, D3D12_EXPORT_FLAG_NONE };
    D3D12_STATE_SUBOBJECT dxilLibSubObject = {};
    dxilLibDesc.DXILLibrary.pShaderBytecode = pShaderByteCode;
    dxilLibDesc.DXILLibrary.BytecodeLength = bytecodeLength;
    dxilLibDesc.NumExports = 1;
    dxilLibDesc.pExports = &exportDesc;
    dxilLibSubObject.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
    dxilLibSubObject.pDesc = &dxilLibDesc;
    return dxilLibSubObject;
}

void SetPipelineStateStackSize(LPCWSTR raygen, LPCWSTR closestHit, LPCWSTR miss, UINT maxRecursion, ID3D12StateObject *pStateObject)
{
    ID3D12StateObjectProperties* stateObjectProperties = nullptr;
    ThrowIfFailed(pStateObject->QueryInterface(IID_PPV_ARGS(&stateObjectProperties)));
    UINT64 closestHitStackSize = stateObjectProperties->GetShaderStackSize(closestHit);
    UINT64 missStackSize = stateObjectProperties->GetShaderStackSize(miss);
    UINT64 raygenStackSize = stateObjectProperties->GetShaderStackSize(raygen);

    UINT64 totalStackSize = raygenStackSize + std::max(missStackSize, closestHitStackSize) * maxRecursion;
    stateObjectProperties->SetPipelineStackSize(totalStackSize);
}

void InitializeRaytracingStateObjects(const Model &model)
{
    ZeroMemory(&g_dynamicCb, sizeof(g_dynamicCb));

    D3D12_STATIC_SAMPLER_DESC staticSamplerDescs[1] = {};
    D3D12_STATIC_SAMPLER_DESC &defaultSampler = staticSamplerDescs[0];
    defaultSampler.Filter = D3D12_FILTER_ANISOTROPIC;
    defaultSampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    defaultSampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    defaultSampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    defaultSampler.MipLODBias = 0.0f;
    defaultSampler.MaxAnisotropy = 16;
    defaultSampler.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    defaultSampler.MinLOD = 0.0f;
    defaultSampler.MaxLOD = D3D12_FLOAT32_MAX;
    defaultSampler.MaxAnisotropy = 8;
    defaultSampler.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
    defaultSampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    defaultSampler.ShaderRegister = 0;

    D3D12_DESCRIPTOR_RANGE1 sceneBuffersDescriptorRange = {};
    sceneBuffersDescriptorRange.BaseShaderRegister = 1;
    sceneBuffersDescriptorRange.NumDescriptors = 5;
    sceneBuffersDescriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    sceneBuffersDescriptorRange.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE;

    D3D12_DESCRIPTOR_RANGE1 srvDescriptorRange = {};
    srvDescriptorRange.BaseShaderRegister = 12;
    srvDescriptorRange.NumDescriptors = 2;
    srvDescriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvDescriptorRange.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE;

    D3D12_DESCRIPTOR_RANGE1 uavDescriptorRange = {};
    uavDescriptorRange.BaseShaderRegister = 2;
    uavDescriptorRange.NumDescriptors = 10;
    uavDescriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uavDescriptorRange.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE;

    CD3DX12_ROOT_PARAMETER1 globalRootSignatureParameters[8];
    globalRootSignatureParameters[0].InitAsDescriptorTable(1, &sceneBuffersDescriptorRange);
    globalRootSignatureParameters[1].InitAsConstantBufferView(0);
    globalRootSignatureParameters[2].InitAsConstantBufferView(1);
    globalRootSignatureParameters[3].InitAsDescriptorTable(1, &srvDescriptorRange);
    globalRootSignatureParameters[4].InitAsDescriptorTable(1, &uavDescriptorRange);
    globalRootSignatureParameters[5].InitAsUnorderedAccessView(0);
    globalRootSignatureParameters[6].InitAsUnorderedAccessView(1);
    globalRootSignatureParameters[7].InitAsShaderResourceView(0);
    auto globalRootSignatureDesc = CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(_countof(globalRootSignatureParameters), globalRootSignatureParameters, _countof(staticSamplerDescs), staticSamplerDescs);

    CComPtr<ID3DBlob> pGlobalRootSignatureBlob;
    CComPtr<ID3DBlob> pErrorBlob;
    if (FAILED(D3D12SerializeVersionedRootSignature(&globalRootSignatureDesc, &pGlobalRootSignatureBlob, &pErrorBlob)))
    {
        OutputDebugStringA((LPCSTR)pErrorBlob->GetBufferPointer());
    }
    g_pRaytracingDevice->CreateRootSignature(0, pGlobalRootSignatureBlob->GetBufferPointer(), pGlobalRootSignatureBlob->GetBufferSize(), IID_PPV_ARGS(&g_GlobalRaytracingRootSignature));

    D3D12_DESCRIPTOR_RANGE1 localTextureDescriptorRange = {};
    localTextureDescriptorRange.BaseShaderRegister = 6;
    localTextureDescriptorRange.NumDescriptors = 2;
    localTextureDescriptorRange.RegisterSpace = 0;
    localTextureDescriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    localTextureDescriptorRange.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE;

    CD3DX12_ROOT_PARAMETER1 localRootSignatureParameters[2];
    UINT sizeOfRootConstantInDwords = (sizeof(RootConstants) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    localRootSignatureParameters[0].InitAsDescriptorTable(1, &localTextureDescriptorRange);
    localRootSignatureParameters[1].InitAsConstants(sizeOfRootConstantInDwords, 3);
    auto localRootSignatureDesc = CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(_countof(localRootSignatureParameters), localRootSignatureParameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE);

    CComPtr<ID3DBlob> pLocalRootSignatureBlob;
    D3D12SerializeVersionedRootSignature(&localRootSignatureDesc, &pLocalRootSignatureBlob, nullptr);
    g_pRaytracingDevice->CreateRootSignature(0, pLocalRootSignatureBlob->GetBufferPointer(), pLocalRootSignatureBlob->GetBufferSize(), IID_PPV_ARGS(&g_LocalRaytracingRootSignature));

    LPCWSTR exportName_RayGen = L"RayGen";
    LPCWSTR exportName_Hit = L"Hit";
    LPCWSTR exportName_Miss = L"Miss";
    LPCWSTR exportName_HitGroup = L"HitGroup";

    UINT nodeMask = 1;

    D3D12_RAYTRACING_PIPELINE_CONFIG pipelineConfig;
    pipelineConfig.MaxTraceRecursionDepth = MaxRayRecursion;

    D3D12_HIT_GROUP_DESC hitGroupDesc = {};
    hitGroupDesc.ClosestHitShaderImport = exportName_Hit;
    hitGroupDesc.HitGroupExport = exportName_HitGroup;

    const UINT shaderIdentifierSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    const UINT offsetToDescriptorHandle = ALIGN(sizeof(D3D12_GPU_DESCRIPTOR_HANDLE), shaderIdentifierSize);
    const UINT offsetToMaterialConstants = ALIGN(sizeof(UINT32), offsetToDescriptorHandle + sizeof(D3D12_GPU_DESCRIPTOR_HANDLE));
    const UINT shaderRecordSizeInBytes = ALIGN(D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT, offsetToMaterialConstants + sizeof(MaterialRootConstant));
    
    uint32_t meshCount = model.m_Header.meshCount;
    std::vector<byte> pHitShaderTable(shaderRecordSizeInBytes * meshCount);
    auto GetShaderTable = [=](const Model &model, ID3D12StateObject *pPSO, byte *pShaderTable)
    {
        ID3D12StateObjectProperties* stateObjectProperties = nullptr;
        ThrowIfFailed(pPSO->QueryInterface(IID_PPV_ARGS(&stateObjectProperties)));
        void *pHitGroupIdentifierData = stateObjectProperties->GetShaderIdentifier(exportName_HitGroup);
        for (UINT i = 0; i < meshCount; i++)
        {
            byte *pShaderRecord = i * shaderRecordSizeInBytes + pShaderTable;
            memcpy(pShaderRecord, pHitGroupIdentifierData, shaderIdentifierSize);

            UINT materialIndex = model.m_pMesh[i].materialIndex;
            memcpy(pShaderRecord + offsetToDescriptorHandle, &g_GpuSceneMaterialSrvs[materialIndex].ptr, sizeof(g_GpuSceneMaterialSrvs[materialIndex].ptr));

            MaterialRootConstant material;
            material.MaterialID = i;
            memcpy(pShaderRecord + offsetToMaterialConstants, &material, sizeof(material));
        }
    };

    // ray shaders
    {
        D3D12_RAYTRACING_SHADER_CONFIG shaderConfig;
        shaderConfig.MaxAttributeSizeInBytes = 8;
        shaderConfig.MaxPayloadSizeInBytes = sizeof(RayPayload);

        D3D12_EXPORT_DESC exportDesc[] =
        {
            { exportName_RayGen, nullptr, D3D12_EXPORT_FLAG_NONE },
            { exportName_Hit,    nullptr, D3D12_EXPORT_FLAG_NONE },
            { exportName_Miss,   nullptr, D3D12_EXPORT_FLAG_NONE },
        };
        D3D12_DXIL_LIBRARY_DESC dxilLibDesc =
        {
            { // DXILLibrary
                g_pRaysLib,
                sizeof(g_pRaysLib)
            },
            _countof(exportDesc), // NumExports
            exportDesc // pExports
        };

        D3D12_STATE_SUBOBJECT stateSubobjects[] =
        {
            { D3D12_STATE_SUBOBJECT_TYPE_NODE_MASK, &nodeMask },
            { D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &g_GlobalRaytracingRootSignature.p },
            { D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG, &pipelineConfig },
            { D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &dxilLibDesc },
            { D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shaderConfig },
            { D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, &hitGroupDesc },
            { D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE, &g_LocalRaytracingRootSignature.p },
        };
        D3D12_STATE_OBJECT_DESC stateObjectDesc =
        {
            D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE,
            _countof(stateSubobjects),
            stateSubobjects
        };

        CComPtr<ID3D12StateObject> pDiffusePSO;
        g_pRaytracingDevice->CreateStateObject(&stateObjectDesc, IID_PPV_ARGS(&pDiffusePSO));
        GetShaderTable(model, pDiffusePSO, pHitShaderTable.data());
        g_RaytracingInputs_Ray = RaytracingDispatchRayInputs(
            *g_pRaytracingDevice, pDiffusePSO, pHitShaderTable.data(), shaderRecordSizeInBytes,
            (UINT)pHitShaderTable.size(), exportName_RayGen, exportName_Miss);

        WCHAR hitGroupExportNameClosestHitType[64];
        swprintf_s(hitGroupExportNameClosestHitType, L"%s::closesthit", exportName_HitGroup);
        SetPipelineStateStackSize(
            exportName_RayGen, hitGroupExportNameClosestHitType, exportName_Miss, MaxRayRecursion, g_RaytracingInputs_Ray.m_pPSO);
    }

    // beam shaders
    {
        D3D12_RAYTRACING_SHADER_CONFIG shaderConfig;
        shaderConfig.MaxAttributeSizeInBytes = 8;
        shaderConfig.MaxPayloadSizeInBytes = sizeof(BeamPayload);

        D3D12_EXPORT_DESC exportDesc[] =
        {
            { exportName_RayGen, nullptr, D3D12_EXPORT_FLAG_NONE },
            { exportName_Hit,    nullptr, D3D12_EXPORT_FLAG_NONE },
            { exportName_Miss,   nullptr, D3D12_EXPORT_FLAG_NONE },
        };
        D3D12_DXIL_LIBRARY_DESC dxilLibDesc =
        {
            { // DXILLibrary
                g_pBeamsLib,
                sizeof(g_pBeamsLib)
            },
            _countof(exportDesc), // NumExports
            exportDesc // pExports
        };

        D3D12_STATE_SUBOBJECT stateSubobjects[] =
        {
            { D3D12_STATE_SUBOBJECT_TYPE_NODE_MASK, &nodeMask },
            { D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &g_GlobalRaytracingRootSignature.p },
            { D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG, &pipelineConfig },
            { D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &dxilLibDesc },
            { D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shaderConfig },
            { D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, &hitGroupDesc },
            { D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE, &g_LocalRaytracingRootSignature.p },
        };
        D3D12_STATE_OBJECT_DESC stateObjectDesc =
        {
            D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE,
            _countof(stateSubobjects),
            stateSubobjects
        };

        CComPtr<ID3D12StateObject> pBeamsPSO;
        g_pRaytracingDevice->CreateStateObject(&stateObjectDesc, IID_PPV_ARGS(&pBeamsPSO));
        GetShaderTable(model, pBeamsPSO, pHitShaderTable.data());
        g_RaytracingInputs_Beam = RaytracingDispatchRayInputs(
            *g_pRaytracingDevice, pBeamsPSO, pHitShaderTable.data(), shaderRecordSizeInBytes,
            (UINT)pHitShaderTable.size(), exportName_RayGen, exportName_Miss);

        WCHAR hitGroupExportNameClosestHitType[64];
        swprintf_s(hitGroupExportNameClosestHitType, L"%s::closesthit", exportName_HitGroup);
        SetPipelineStateStackSize(
            exportName_RayGen, hitGroupExportNameClosestHitType, exportName_Miss, MaxRayRecursion, g_RaytracingInputs_Beam.m_pPSO);
    }

    // beam post processing shaders
    {
        D3D12_EXPORT_DESC exportDesc[] =
        {
            { L"ShadeQuads", nullptr, D3D12_EXPORT_FLAG_NONE },
        };
        D3D12_DXIL_LIBRARY_DESC dxilLibDesc =
        {
            { // DXILLibrary
                g_pBeamsLib,
                sizeof(g_pBeamsLib)
            },
            _countof(exportDesc), // NumExports
            exportDesc // pExports
        };

        D3D12_STATE_SUBOBJECT stateSubobjects[] =
        {
            { D3D12_STATE_SUBOBJECT_TYPE_NODE_MASK, &nodeMask },
            { D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &g_GlobalRaytracingRootSignature.p },
            { D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &dxilLibDesc },
            { D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE, &g_LocalRaytracingRootSignature.p },
        };
        D3D12_STATE_OBJECT_DESC stateObjectDesc =
        {
            D3D12_STATE_OBJECT_TYPE_COLLECTION,
            _countof(stateSubobjects),
            stateSubobjects
        };

        g_BeamPostRootSig.Reset(1, 0);
        g_BeamPostRootSig[0].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 1, D3D12_SHADER_VISIBILITY_ALL);
        g_BeamPostRootSig.Finalize(L"g_BeamPostRootSig");

        g_BeamShadeQuadsPSO.SetRootSignature(g_BeamPostRootSig);
        // looks like I need to split my compute shaders out into separate files
        g_BeamShadeQuadsPSO.SetComputeShader(g_pBeamsShadeQuads, sizeof(g_pBeamsShadeQuads));
        g_BeamShadeQuadsPSO.Finalize();
    }
}

void DxrMsaaDemo::createBvh(const Model &model, BVH &bvh, bool triangles)
{
    uint32_t meshCount = m_Model.m_Header.meshCount;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topDesc = {};
    topDesc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    topDesc.Inputs.NumDescs = bvhBottomCount;
    topDesc.Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    topDesc.Inputs.pGeometryDescs = nullptr;
    topDesc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topPrebuildInfo;
    g_pRaytracingDevice->GetRaytracingAccelerationStructurePrebuildInfo(&topDesc.Inputs, &topPrebuildInfo);

    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geoDesc(meshCount);
    for (uint32_t m = 0; m < meshCount; m++)
    {
        const Model::Mesh &mesh = m_Model.m_pMesh[m];

        if (triangles)
        {
            geoDesc[m].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
            geoDesc[m].Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
            geoDesc[m].Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
            geoDesc[m].Triangles.VertexCount = mesh.vertexCount;
            geoDesc[m].Triangles.VertexBuffer.StartAddress = m_Model.m_VertexBuffer.GetGpuVirtualAddress()
                + mesh.vertexDataByteOffset + mesh.attrib[Model::attrib_position].offset;
            geoDesc[m].Triangles.IndexBuffer = m_Model.m_IndexBuffer.GetGpuVirtualAddress() + mesh.indexDataByteOffset;
            geoDesc[m].Triangles.VertexBuffer.StrideInBytes = mesh.vertexStride;
            geoDesc[m].Triangles.IndexCount = mesh.indexCount;
            geoDesc[m].Triangles.IndexFormat = DXGI_FORMAT_R16_UINT;
            geoDesc[m].Triangles.Transform3x4 = 0;
        }
        else
        {
            geoDesc[m].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
            geoDesc[m].Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
            geoDesc[m].AABBs.AABBCount = mesh.indexCount / 3;
// TODO: create and upload per-triangle AABB buffers...
// ... as a perf improvement, it might make sense to sense to do two tris per AABB and have the Intersection shader loop,
// but I can probably punt on that for this demo. Or until I benchmark and optimize.
            geoDesc[m].AABBs.AABBs.StartAddress =
            geoDesc[m].AABBs.AABBs.StrideInBytes = sizeof(D3D12_RAYTRACING_AABB);
        }
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomDesc = {};
    bottomDesc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    bottomDesc.Inputs.NumDescs = meshCount;
    bottomDesc.Inputs.pGeometryDescs = geoDesc.data();
    bottomDesc.Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    bottomDesc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bottomPrebuildInfo;
    g_pRaytracingDevice->GetRaytracingAccelerationStructurePrebuildInfo(&bottomDesc.Inputs, &bottomPrebuildInfo);

    uint64_t scratchBufferSizeNeeded = std::max(bottomPrebuildInfo.ScratchDataSizeInBytes, topPrebuildInfo.ScratchDataSizeInBytes);

    ByteAddressBuffer scratchBuffer;
    scratchBuffer.Create(L"Acceleration Structure Scratch Buffer", (uint32_t)scratchBufferSizeNeeded, 1);

    D3D12_HEAP_PROPERTIES defaultHeapDesc = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    auto topLevelDesc = CD3DX12_RESOURCE_DESC::Buffer(topPrebuildInfo.ResultDataMaxSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    g_Device->CreateCommittedResource(
        &defaultHeapDesc,
        D3D12_HEAP_FLAG_NONE,
        &topLevelDesc,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        nullptr,
        IID_PPV_ARGS(&bvh.top));
    topDesc.DestAccelerationStructureData = bvh.top->GetGPUVirtualAddress();
    topDesc.ScratchAccelerationStructureData = scratchBuffer.GetGpuVirtualAddress();

    D3D12_RAYTRACING_INSTANCE_DESC instanceDesc = {};
    auto bottomLevelDesc = CD3DX12_RESOURCE_DESC::Buffer(bottomPrebuildInfo.ResultDataMaxSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    g_Device->CreateCommittedResource(
        &defaultHeapDesc,
        D3D12_HEAP_FLAG_NONE, 
        &bottomLevelDesc, 
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        nullptr, 
        IID_PPV_ARGS(&bvh.bottom));
    bottomDesc.DestAccelerationStructureData = bvh.bottom->GetGPUVirtualAddress();
    bottomDesc.ScratchAccelerationStructureData = scratchBuffer.GetGpuVirtualAddress();

    // Identity matrix
    ZeroMemory(instanceDesc.Transform, sizeof(instanceDesc.Transform));
    instanceDesc.Transform[0][0] = 1.0f;
    instanceDesc.Transform[1][1] = 1.0f;
    instanceDesc.Transform[2][2] = 1.0f;
    instanceDesc.AccelerationStructure = bvh.bottom->GetGPUVirtualAddress();
    instanceDesc.Flags = 0;
    instanceDesc.InstanceID = 0;
    instanceDesc.InstanceMask = 1;
    instanceDesc.InstanceContributionToHitGroupIndex = 0;

    ByteAddressBuffer instanceDataBuffer;
    instanceDataBuffer.Create(L"Instance Data Buffer", bvhBottomCount, sizeof(D3D12_RAYTRACING_INSTANCE_DESC), &instanceDesc);
    topDesc.Inputs.InstanceDescs = instanceDataBuffer.GetGpuVirtualAddress();
    topDesc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;

    // build the acceleration structures
    {
        GraphicsContext &gfxContext = GraphicsContext::Begin(L"Create Acceleration Structure");
        ID3D12GraphicsCommandList *pCommandList = gfxContext.GetCommandList();

        CComPtr<ID3D12GraphicsCommandList4> pRaytracingCommandList;
        pCommandList->QueryInterface(IID_PPV_ARGS(&pRaytracingCommandList));

        ID3D12DescriptorHeap *descriptorHeaps[] = { &g_pRaytracingDescriptorHeap->GetDescriptorHeap() };
        pRaytracingCommandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

        auto uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
        pRaytracingCommandList->BuildRaytracingAccelerationStructure(&bottomDesc, 0, nullptr);
        pCommandList->ResourceBarrier(1, &uavBarrier);

        pRaytracingCommandList->BuildRaytracingAccelerationStructure(&topDesc, 0, nullptr);

        gfxContext.Finish(true);
    }
}

void DxrMsaaDemo::Startup()
{
    ThrowIfFailed(g_Device->QueryInterface(IID_PPV_ARGS(&g_pRaytracingDevice)), L"Couldn't get DirectX Raytracing interface for the device.\n");

    g_pRaytracingDescriptorHeap = std::unique_ptr<DescriptorHeapStack>(
        new DescriptorHeapStack(*g_Device, 200, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 0));

    D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1;
    HRESULT hr = g_Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1));

    SamplerDesc DefaultSamplerDesc;
    DefaultSamplerDesc.MaxAnisotropy = 8;

    m_RootSig.Reset(4, 1);
    m_RootSig.InitStaticSampler(0, DefaultSamplerDesc, D3D12_SHADER_VISIBILITY_PIXEL);
    m_RootSig[0].InitAsConstantBuffer(0, D3D12_SHADER_VISIBILITY_VERTEX);
    m_RootSig[1].InitAsConstantBuffer(0, D3D12_SHADER_VISIBILITY_PIXEL);
    m_RootSig[2].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 6, D3D12_SHADER_VISIBILITY_PIXEL);
    m_RootSig[3].InitAsConstants(1, 2, D3D12_SHADER_VISIBILITY_VERTEX);
    m_RootSig.Finalize(L"DxrMsaaDemo", D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    D3D12_INPUT_ELEMENT_DESC vertElem[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "BITANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
    };

    // Full color pass
    DXGI_FORMAT formats[] { g_SceneColorBuffer.GetFormat() };
    m_ModelPSO.SetRootSignature(m_RootSig);
    m_ModelPSO.SetRasterizerState(RasterizerDefault);
    m_ModelPSO.SetInputLayout(_countof(vertElem), vertElem);
    m_ModelPSO.SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    m_ModelPSO.SetBlendState(BlendDisable);
    m_ModelPSO.SetDepthStencilState(DepthStateReadWrite);
    m_ModelPSO.SetRenderTargetFormats(_countof(formats), formats, g_SceneDepthBuffer.GetFormat());
    m_ModelPSO.SetVertexShader( g_pModelViewerVS, sizeof(g_pModelViewerVS) );
    m_ModelPSO.SetPixelShader( g_pModelViewerPS, sizeof(g_pModelViewerPS) );
    m_ModelPSO.Finalize();

#define ASSET_DIRECTORY "../../../../../MiniEngine/ModelViewer/"
    TextureManager::Initialize(ASSET_DIRECTORY L"Textures/");
    bool bModelLoadSuccess = m_Model.Load(ASSET_DIRECTORY "Models/sponza.h3d");
    ASSERT(bModelLoadSuccess, "Failed to load model");
    ASSERT(m_Model.m_Header.meshCount > 0, "Model contains no meshes");

    g_shadeConstantBuffer.Create(L"Hit Constant Buffer", 1, sizeof(ShadeConstants));
    g_dynamicConstantBuffer.Create(L"Dynamic Constant Buffer", 1, sizeof(DynamicCB));

    InitializeSceneInfo(m_Model);
    InitializeViews(m_Model);

    createBvh(m_Model, g_bvhTriangles, true);
    createBvh(m_Model, g_bvhAABBs, false);

    InitializeRaytracingStateObjects(m_Model);

    float modelRadius = Length(m_Model.m_Header.boundingBox.max - m_Model.m_Header.boundingBox.min) * .5f;
    const Vector3 eye = (m_Model.m_Header.boundingBox.min + m_Model.m_Header.boundingBox.max) * .5f + Vector3(modelRadius * .5f, 0.0f, 0.0f);
    m_Camera.SetEyeAtUp( eye, Vector3(kZero), Vector3(kYUnitVector) );
    
    m_CameraPosArrayCurrentPosition = 0;
    
    // Lion's head
    m_CameraPosArray[0].position = Vector3(-1100.0f, 170.0f, -30.0f);
    m_CameraPosArray[0].heading = 1.5707f;
    m_CameraPosArray[0].pitch = 0.0f;

    // View of columns
    m_CameraPosArray[1].position = Vector3(299.0f, 208.0f, -202.0f);
    m_CameraPosArray[1].heading = -3.1111f;
    m_CameraPosArray[1].pitch = 0.5953f;

    // Bottom-up view from the floor
    m_CameraPosArray[2].position = Vector3(-1237.61f, 80.60f, -26.02f);
    m_CameraPosArray[2].heading = -1.5707f;
    m_CameraPosArray[2].pitch = 0.268f;

    // Top-down view from the second floor
    m_CameraPosArray[3].position = Vector3(-977.90f, 595.05f, -194.97f);
    m_CameraPosArray[3].heading = -2.077f;
    m_CameraPosArray[3].pitch =  - 0.450f;

    // View of corridors on the second floor
    m_CameraPosArray[4].position = Vector3(-1463.0f, 600.0f, 394.52f);
    m_CameraPosArray[4].heading = -1.236f;
    m_CameraPosArray[4].pitch = 0.0f;

    m_Camera.SetZRange( 1.0f, 10000.0f );

    m_CameraController.reset(new CameraController(m_Camera, Vector3(kYUnitVector)));
    
    MotionBlur::Enable = false;
    TemporalEffects::EnableTAA = false;
    FXAA::Enable = false;
    PostEffects::EnableHDR = false;
    PostEffects::EnableAdaptation = true;
    SSAO::Enable = false;
}

void DxrMsaaDemo::Cleanup()
{
    m_Model.Clear();
}

namespace Graphics
{
    extern EnumVar DebugZoom;
}

void DxrMsaaDemo::Update( float deltaT )
{
    ScopedTimer _prof(L"Update State");

    if (GameInput::IsFirstPressed(GameInput::kLShoulder))
        DebugZoom.Decrement();
    else if (GameInput::IsFirstPressed(GameInput::kRShoulder))
        DebugZoom.Increment();

    if (GameInput::IsFirstPressed(GameInput::kKey_r))
        renderMode = (renderMode + 1) % int(RenderMode::count);
    if (GameInput::IsFirstPressed(GameInput::kKey_m))
        enableMsaa = !enableMsaa;
    
    static bool freezeCamera = false;
    
    if (GameInput::IsFirstPressed(GameInput::kKey_f))
    {
        freezeCamera = !freezeCamera;
    }

    if (GameInput::IsFirstPressed(GameInput::kKey_left))
    {
        m_CameraPosArrayCurrentPosition = (m_CameraPosArrayCurrentPosition + c_NumCameraPositions - 1) % c_NumCameraPositions;
        SetCameraToPredefinedPosition(m_CameraPosArrayCurrentPosition);
    }
    else if (GameInput::IsFirstPressed(GameInput::kKey_right))
    {
        m_CameraPosArrayCurrentPosition = (m_CameraPosArrayCurrentPosition + 1) % c_NumCameraPositions;
        SetCameraToPredefinedPosition(m_CameraPosArrayCurrentPosition);
    }

    if (!freezeCamera) 
    {
        m_CameraController->Update(deltaT);
    }

    m_ViewProjMatrix = m_Camera.GetViewProjMatrix();

    float costheta = cosf(m_SunOrientation);
    float sintheta = sinf(m_SunOrientation);
    float cosphi = cosf(m_SunInclination * 3.14159f * 0.5f);
    float sinphi = sinf(m_SunInclination * 3.14159f * 0.5f);
    m_SunDirection = Normalize(Vector3( costheta * cosphi, sinphi, sintheta * cosphi ));

    // We use viewport offsets to jitter sample positions from frame to frame (for TAA.)
    // D3D has a design quirk with fractional offsets such that the implicit scissor
    // region of a viewport is floor(TopLeftXY) and floor(TopLeftXY + WidthHeight), so
    // having a negative fractional top left, e.g. (-0.25, -0.25) would also shift the
    // BottomRight corner up by a whole integer.  One solution is to pad your viewport
    // dimensions with an extra pixel.  My solution is to only use positive fractional offsets,
    // but that means that the average sample position is +0.5, which I use when I disable
    // temporal AA.
    TemporalEffects::GetJitterOffset(m_MainViewport.TopLeftX, m_MainViewport.TopLeftY);

    m_MainViewport.Width = (float)g_SceneColorBuffer.GetWidth();
    m_MainViewport.Height = (float)g_SceneColorBuffer.GetHeight();
    m_MainViewport.MinDepth = 0.0f;
    m_MainViewport.MaxDepth = 1.0f;

    m_MainScissor.left = 0;
    m_MainScissor.top = 0;
    m_MainScissor.right = (LONG)g_SceneColorBuffer.GetWidth();
    m_MainScissor.bottom = (LONG)g_SceneColorBuffer.GetHeight();
}

void DxrMsaaDemo::RenderObjects(GraphicsContext& gfxContext, const Matrix4& ViewProjMat)
{
    struct VSConstants
    {
        Matrix4 modelToProjection;
        XMFLOAT3 viewerPos;
    } vsConstants;
    vsConstants.modelToProjection = ViewProjMat;
    XMStoreFloat3(&vsConstants.viewerPos, m_Camera.GetPosition());

    gfxContext.SetDynamicConstantBufferView(0, sizeof(vsConstants), &vsConstants);

    uint32_t VertexStride = m_Model.m_VertexStride;

    for (uint32_t meshIndex = 0; meshIndex < m_Model.m_Header.meshCount; meshIndex++)
    {
        const Model::Mesh& mesh = m_Model.m_pMesh[meshIndex];

        uint32_t indexCount = mesh.indexCount;
        uint32_t startIndex = mesh.indexDataByteOffset / sizeof(uint16_t);
        uint32_t baseVertex = mesh.vertexDataByteOffset / VertexStride;

        uint32_t materialIndex = mesh.materialIndex;
        gfxContext.SetDynamicDescriptors(2, 0, 6, m_Model.GetSRVs(materialIndex));
        gfxContext.SetConstants(3, baseVertex, materialIndex);

        gfxContext.DrawIndexed(indexCount, startIndex, baseVertex);
    }
}

void DxrMsaaDemo::SetCameraToPredefinedPosition(int cameraPosition) 
{
    if (cameraPosition < 0 || cameraPosition >= c_NumCameraPositions)
        return;
    
    m_CameraController->SetCurrentHeading(m_CameraPosArray[m_CameraPosArrayCurrentPosition].heading);
    m_CameraController->SetCurrentPitch(m_CameraPosArray[m_CameraPosArrayCurrentPosition].pitch);

    Matrix3 neworientation = Matrix3(m_CameraController->GetWorldEast(), m_CameraController->GetWorldUp(), -m_CameraController->GetWorldNorth()) 
                           * Matrix3::MakeYRotation(m_CameraController->GetCurrentHeading())
                           * Matrix3::MakeXRotation(m_CameraController->GetCurrentPitch());
    m_Camera.SetTransform(AffineTransform(neworientation, m_CameraPosArray[m_CameraPosArrayCurrentPosition].position));
    m_Camera.Update();
}

void DxrMsaaDemo::RenderScene()
{
    const bool skipDiffusePass = (renderMode != int(RenderMode::raster));

    GraphicsContext& gfxContext = GraphicsContext::Begin(L"Scene Render");

    ParticleEffects::Update(gfxContext.GetComputeContext(), Graphics::GetFrameTime());

    uint32_t FrameIndex = TemporalEffects::GetFrameIndexMod2();

    ShadeConstants shadeConstants = {};
    shadeConstants.sunDirection = m_SunDirection;
    shadeConstants.sunColor = Vector3(1.0f, 1.0f, 1.0f) * m_SunLightIntensity;
    shadeConstants.ambientColor = Vector3(1.0f, 1.0f, 1.0f) * m_AmbientIntensity;

    // Set the default state for command lists
    auto& pfnSetupGraphicsState = [&](void)
    {
        gfxContext.SetRootSignature(m_RootSig);
        gfxContext.SetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        gfxContext.SetIndexBuffer(m_Model.m_IndexBuffer.IndexBufferView());
        gfxContext.SetVertexBuffer(0, m_Model.m_VertexBuffer.VertexBufferView());
    };

    pfnSetupGraphicsState();

    if (!skipDiffusePass)
    {
        {
            ScopedTimer _prof(L"Render Color - Clear", gfxContext);

            gfxContext.TransitionResource(g_SceneDepthBuffer, D3D12_RESOURCE_STATE_DEPTH_WRITE, true);
            gfxContext.ClearDepth(g_SceneDepthBuffer);

            gfxContext.TransitionResource(g_SceneColorBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, true);
            gfxContext.ClearColor(g_SceneColorBuffer);
        }

        {
            ScopedTimer _prof(L"Render Color - Geo", gfxContext);

            gfxContext.SetDynamicConstantBufferView(1, sizeof(shadeConstants), &shadeConstants);

            D3D12_CPU_DESCRIPTOR_HANDLE rtvs[]{ g_SceneColorBuffer.GetRTV() };
            gfxContext.SetRenderTargets(_countof(rtvs), rtvs, g_SceneDepthBuffer.GetDSV());
            gfxContext.SetViewportAndScissor(m_MainViewport, m_MainScissor);

            gfxContext.SetPipelineState(m_ModelPSO);
            RenderObjects(gfxContext, m_ViewProjMatrix);
        }
    }

    Raytrace(gfxContext);

    gfxContext.Finish();
}

void DxrMsaaDemo::RaytraceDiffuse(
    GraphicsContext& context,
    const Math::Camera& camera,
    ColorBuffer& colorTarget)
{
    ScopedTimer _p0(L"RaytracingWithHitShader", context);

    // Prepare constants
    DynamicCB inputs = g_dynamicCb;
    auto m0 = camera.GetViewProjMatrix();
    auto m1 = Transpose(Invert(m0));
    memcpy(&inputs.cameraToWorld, &m1, sizeof(inputs.cameraToWorld));
    memcpy(&inputs.worldCameraPosition, &camera.GetPosition(), sizeof(inputs.worldCameraPosition));
    inputs.resolution.x = (float)colorTarget.GetWidth();
    inputs.resolution.y = (float)colorTarget.GetHeight();

    ShadeConstants shadeConstants = {};
    shadeConstants.sunDirection = m_SunDirection;
    shadeConstants.sunColor = Vector3(1.0f, 1.0f, 1.0f) * m_SunLightIntensity;
    shadeConstants.ambientColor = Vector3(1.0f, 1.0f, 1.0f) * m_AmbientIntensity;
    context.WriteBuffer(g_shadeConstantBuffer, 0, &shadeConstants, sizeof(shadeConstants));
    context.WriteBuffer(g_dynamicConstantBuffer, 0, &inputs, sizeof(inputs));

    context.TransitionResource(g_dynamicConstantBuffer, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
    context.TransitionResource(g_SSAOFullScreen, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    context.TransitionResource(g_shadeConstantBuffer, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
    context.TransitionResource(g_ShadowBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    context.TransitionResource(colorTarget, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    context.FlushResourceBarriers();

    ID3D12GraphicsCommandList * pCommandList = context.GetCommandList();

    CComPtr<ID3D12GraphicsCommandList4> pRaytracingCommandList;
    pCommandList->QueryInterface(IID_PPV_ARGS(&pRaytracingCommandList));

    ID3D12DescriptorHeap *pDescriptorHeaps[] = { &g_pRaytracingDescriptorHeap->GetDescriptorHeap() };
    pRaytracingCommandList->SetDescriptorHeaps(_countof(pDescriptorHeaps), pDescriptorHeaps);

    pCommandList->SetComputeRootSignature(g_GlobalRaytracingRootSignature);
    pCommandList->SetComputeRootDescriptorTable(0, g_SceneSrvs);
    pCommandList->SetComputeRootConstantBufferView(1, g_shadeConstantBuffer.GetGpuVirtualAddress());
    pCommandList->SetComputeRootConstantBufferView(2, g_dynamicConstantBuffer.GetGpuVirtualAddress());
    pCommandList->SetComputeRootDescriptorTable(4, g_OutputUAV);
    pRaytracingCommandList->SetComputeRootShaderResourceView(7, g_bvhTriangles.top->GetGPUVirtualAddress());

    D3D12_DISPATCH_RAYS_DESC dispatchRaysDesc = g_RaytracingInputs_Ray.GetDispatchRayDesc(colorTarget.GetWidth(), colorTarget.GetHeight());
    pRaytracingCommandList->SetPipelineState1(g_RaytracingInputs_Ray.m_pPSO);
    pRaytracingCommandList->DispatchRays(&dispatchRaysDesc);
}

void DxrMsaaDemo::RaytraceDiffuseBeams(
    GraphicsContext& context,
    const Math::Camera& camera,
    ColorBuffer& colorTarget)
{
    ScopedTimer _p0(L"RaytraceDiffuseBeams", context);

    // Prepare constants
    DynamicCB inputs = g_dynamicCb;
    auto m0 = camera.GetViewProjMatrix();
    auto m1 = Transpose(Invert(m0));
    memcpy(&inputs.cameraToWorld, &m1, sizeof(inputs.cameraToWorld));
    memcpy(&inputs.worldCameraPosition, &camera.GetPosition(), sizeof(inputs.worldCameraPosition));
    inputs.resolution.x = (float)colorTarget.GetWidth();
    inputs.resolution.y = (float)colorTarget.GetHeight();

    ShadeConstants shadeConstants = {};
    shadeConstants.sunDirection = m_SunDirection;
    shadeConstants.sunColor = Vector3(1.0f, 1.0f, 1.0f) * m_SunLightIntensity;
    shadeConstants.ambientColor = Vector3(1.0f, 1.0f, 1.0f) * m_AmbientIntensity;
    context.WriteBuffer(g_shadeConstantBuffer, 0, &shadeConstants, sizeof(shadeConstants));
    context.WriteBuffer(g_dynamicConstantBuffer, 0, &inputs, sizeof(inputs));

    context.TransitionResource(g_dynamicConstantBuffer, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
    context.TransitionResource(g_SSAOFullScreen, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    context.TransitionResource(g_shadeConstantBuffer, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
    context.TransitionResource(g_ShadowBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    context.TransitionResource(colorTarget, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    context.FlushResourceBarriers();

    ID3D12GraphicsCommandList* pCommandList = context.GetCommandList();

    CComPtr<ID3D12GraphicsCommandList4> pRaytracingCommandList;
    pCommandList->QueryInterface(IID_PPV_ARGS(&pRaytracingCommandList));

    ID3D12DescriptorHeap* pDescriptorHeaps[] = { &g_pRaytracingDescriptorHeap->GetDescriptorHeap() };
    pRaytracingCommandList->SetDescriptorHeaps(_countof(pDescriptorHeaps), pDescriptorHeaps);

    pCommandList->SetComputeRootSignature(g_GlobalRaytracingRootSignature);
    pCommandList->SetComputeRootDescriptorTable(0, g_SceneSrvs);
    pCommandList->SetComputeRootConstantBufferView(1, g_shadeConstantBuffer.GetGpuVirtualAddress());
    pCommandList->SetComputeRootConstantBufferView(2, g_dynamicConstantBuffer.GetGpuVirtualAddress());
    pCommandList->SetComputeRootDescriptorTable(4, g_OutputUAV);
    pRaytracingCommandList->SetComputeRootShaderResourceView(7, g_bvhTriangles.top->GetGPUVirtualAddress());

    // we'll just keep it simple for the demo and round down
    uint32_t beamsX = colorTarget.GetWidth() / BEAM_SIZE;
    uint32_t beamsY = colorTarget.GetHeight() / BEAM_SIZE;

    D3D12_DISPATCH_RAYS_DESC dispatchRaysDesc = g_RaytracingInputs_Beam.GetDispatchRayDesc(
        beamsX, beamsY);
    pRaytracingCommandList->SetPipelineState1(g_RaytracingInputs_Beam.m_pPSO);
    pRaytracingCommandList->DispatchRays(&dispatchRaysDesc);

    // shade quads
    pRaytracingCommandList->SetComputeRootSignature(g_BeamPostRootSig.GetSignature());
    pRaytracingCommandList->SetComputeRootDescriptorTable(0, g_OutputUAV);
    pRaytracingCommandList->SetPipelineState(g_BeamShadeQuadsPSO.GetPipelineStateObject());
    pRaytracingCommandList->Dispatch(beamsX, beamsY, 1);
}

void DxrMsaaDemo::RenderUI(class GraphicsContext& gfxContext)
{
    const UINT framesToAverage = 20;
    static float frameRates[framesToAverage] = {};
    frameRates[Graphics::GetFrameCount() % framesToAverage] = Graphics::GetFrameRate();
    float rollingAverageFrameRate = 0.0;
    for (auto frameRate : frameRates)
    {
        rollingAverageFrameRate += frameRate / framesToAverage;
    }

    float primaryRaysPerSec = g_SceneColorBuffer.GetWidth() * g_SceneColorBuffer.GetHeight() * rollingAverageFrameRate / (1000000.0f);
    TextContext text(gfxContext);
    text.Begin();
    text.DrawFormattedString("\nMillion Primary Rays/s: %7.3f", primaryRaysPerSec);
    text.DrawFormattedString("\nRenderMode: %s MSAA: %s", renderModeStr[renderMode], enableMsaa ? "Y" : "N");
    text.End();
}

void DxrMsaaDemo::Raytrace(class GraphicsContext& gfxContext)
{
    ScopedTimer _prof(L"Raytrace", gfxContext);

    gfxContext.TransitionResource(g_SSAOFullScreen, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    switch (RenderMode(int(renderMode)))
    {
    case RenderMode::raster:
        break;

    case RenderMode::rays:
        RaytraceDiffuse(gfxContext, m_Camera, g_SceneColorBuffer);
        break;

    case RenderMode::beams:
        RaytraceDiffuseBeams(gfxContext, m_Camera, g_SceneColorBuffer);
        break;
    }

    // Clear the gfxContext's descriptor heap since ray tracing changes this underneath the sheets
    gfxContext.SetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, nullptr);
}

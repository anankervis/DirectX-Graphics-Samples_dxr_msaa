//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
// Developed by Minigraph
//
// Author(s):    James Stanard
//               Alex Nankervis
//
// Thanks to Michal Drobot for his feedback.

#define HLSL

#include "ModelViewerRS.h"
#include "Shading.h"

// outdated warning about for-loop variable scope
#pragma warning (disable: 3078)
// single-iteration loop
#pragma warning (disable: 3557)

struct VSOutput
{
    sample float4 position : SV_Position;
    sample float3 worldPos : WorldPos;
    sample float2 uv : TexCoord0;
    sample float3 viewDir : TexCoord1;
    sample float3 normal : Normal;
    sample float3 tangent : Tangent;
    sample float3 bitangent : Bitangent;
};

Texture2D<float3> texDiffuse        : register(t0);
Texture2D<float3> texSpecular        : register(t1);
//Texture2D<float4> texEmissive        : register(t2);
Texture2D<float3> texNormal            : register(t3);
//Texture2D<float4> texLightmap        : register(t4);
//Texture2D<float4> texReflection    : register(t5);

cbuffer b0 : register(b0)
{
    ShadeConstants shadeConstants;
}

SamplerState sampler0 : register(s0);

struct MRT
{
    float3 Color : SV_Target0;
};

[RootSignature(ModelViewer_RootSig)]
MRT main(VSOutput vsOutput)
{
    MRT mrt;

    float gloss = 128.0;
    float3 normal = texNormal.Sample(sampler0, vsOutput.uv) * 2.0 - 1.0;
    AntiAliasSpecular(normal, gloss);
    float3x3 tbn = float3x3(normalize(vsOutput.tangent), normalize(vsOutput.bitangent), normalize(vsOutput.normal));
    normal = normalize(mul(normal, tbn));

    mrt.Color = Shade(
        texDiffuse.Sample(sampler0, vsOutput.uv),
        shadeConstants.ambientColor,
        float3(0.56, 0.56, 0.56),
        texSpecular.Sample(sampler0, vsOutput.uv).g,
        gloss,
        normal,
        normalize(vsOutput.viewDir),
        shadeConstants.sunDirection,
        shadeConstants.sunColor);

    return mrt;
}

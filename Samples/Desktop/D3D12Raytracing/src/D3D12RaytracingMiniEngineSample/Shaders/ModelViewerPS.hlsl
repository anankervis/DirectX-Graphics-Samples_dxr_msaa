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

cbuffer PSConstants : register(b0)
{
    float3 SunDirection;
    float3 SunColor;
    float3 AmbientColor;
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
    mrt.Color = 0.0;

    uint2 pixelPos = uint2(vsOutput.position.xy);
# define SAMPLE_TEX(texName) texName.Sample(sampler0, vsOutput.uv)

    float3 diffuseAlbedo = SAMPLE_TEX(texDiffuse);
    float3 colorSum = 0;
    {
        float ssao = 1.0f;
        colorSum += AmbientColor * diffuseAlbedo * ssao;
    }

    float gloss = 128.0;
    float3 normal;
    {
        normal = SAMPLE_TEX(texNormal) * 2.0 - 1.0;
        AntiAliasSpecular(normal, gloss);
        float3x3 tbn = float3x3(normalize(vsOutput.tangent), normalize(vsOutput.bitangent), normalize(vsOutput.normal));
        normal = mul(normal, tbn);

        // Normalize result...
        float lenSq = dot(normal, normal);

        // Some Sponza content appears to have no tangent space provided, resulting in degenerate normal vectors.
        if (!isfinite(lenSq) || lenSq < 1e-6)
            return mrt;

        normal *= rsqrt(lenSq);
    }

    float3 specularAlbedo = float3(0.56, 0.56, 0.56);
    float specularMask = SAMPLE_TEX(texSpecular).g;
    float3 viewDir = normalize(vsOutput.viewDir);

    float shadow = 1.0f;
    colorSum += shadow * ApplyLightCommon(
        diffuseAlbedo,
        specularAlbedo,
        specularMask,
        gloss,
        normal,
        viewDir,
        SunDirection,
        SunColor);

    mrt.Color = colorSum;

    return mrt;
}


#undef POINT_LIGHT_PARAMS
#undef SPOT_LIGHT_PARAMS
#undef SHADOWED_LIGHT_PARAMS

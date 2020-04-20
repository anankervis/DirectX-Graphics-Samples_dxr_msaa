#pragma once

#ifndef HLSL
# include "HlslCompat.h"
#endif

#ifdef HLSL
# define STRUCT_ALIGN(x)
#else
# define STRUCT_ALIGN(x) __declspec(align(x))
#endif

#define SHADOW_MODE_NONE 0
#define SHADOW_MODE_HARD 1
#define SHADOW_MODE_SOFT 2
#define SHADOW_MODE_BEAM 3

#define SHADOW_MODE SHADOW_MODE_NONE

// For soft shadow mode, how many extra samples do we take?
// This is a multiplier on top of the number of AA samples we're already taking.
// 0 = 1x, 1 = 2x, 2 = 4x, 3 = 8x, 4 = 16x
#if SHADOW_MODE == SHADOW_MODE_SOFT
# define SHADOW_SAMPLES_LOG2 2
#else
# define SHADOW_SAMPLES_LOG2 0
#endif
#define SHADOW_SAMPLES (1 << SHADOW_SAMPLES_LOG2)

// smaller numbers to scale the area light extents down, to create harder shadows
#define SHADOW_AREA_LIGHT_SCALE .01f

#define AREA_LIGHT_CENTER float3(-61, 1296, -38)
#define AREA_LIGHT_EXTENT (float3(907 * SHADOW_AREA_LIGHT_SCALE, 0 * SHADOW_AREA_LIGHT_SCALE, 189 * SHADOW_AREA_LIGHT_SCALE))

STRUCT_ALIGN(16) struct ShadeConstants
{
    float3 sunDirection; uint pad0;
    float3 sunColor; uint pad1;
    float3 ambientColor; uint pad2;
};

#ifdef HLSL

# if SHADOW_MODE == SHADOW_MODE_SOFT
float2 shadowRandom(uint2 p, uint sampleIndex)
{
    uint x = sampleIndex * 1704635963 + p.x * 2704612033 + p.y * 3704636251;
    x = x * (x >> 17) + 0x3fe6835b;
    uint y = sampleIndex * 2104636799 + p.x * 3004635791 + p.y * 4104635947;
    y = y * (y >> 15) + 0xd3ab4b52;
    return float2(
        (x % 2048) / 2048.0f,
        (y % 2048) / 2048.0f) * 2.0f - 1.0f;
}
# endif

void AntiAliasSpecular(inout float3 texNormal, inout float gloss)
{
    float normalLenSq = dot(texNormal, texNormal);
    float invNormalLen = rsqrt(normalLenSq);
    texNormal *= invNormalLen;
    gloss = lerp(1, gloss, rcp(invNormalLen));
}

// Apply fresnel to modulate the specular albedo
void FSchlick(inout float3 specular, inout float3 diffuse, float3 lightDir, float3 halfVec)
{
    float fresnel = pow(1.0 - saturate(dot(lightDir, halfVec)), 5.0);
    specular = lerp(specular, 1, fresnel);
    diffuse = lerp(diffuse, 0, fresnel);
}

float3 ApplyLightCommon(
    float3    diffuseColor,  // Diffuse albedo
    float3    specularColor, // Specular albedo
    float     specularMask,  // Where is it shiny or dingy?
    float     gloss,         // Specular power
    float3    normal,        // World-space normal
    float3    viewDir,       // World-space vector from eye to point
    float3    lightDir,      // World-space vector from point to light
    float3    lightColor     // Radiance of directional light
)
{
    float3 halfVec = normalize(lightDir - viewDir);
    float nDotH = saturate(dot(halfVec, normal));

    FSchlick(specularColor, diffuseColor, lightDir, halfVec);

    float specularFactor = specularMask * pow(nDotH, gloss) * (gloss + 2) / 8;

    float nDotL = saturate(dot(normal, lightDir));

    return nDotL * lightColor * (diffuseColor + specularFactor * specularColor);
}

float3 Shade(
    float3 diffuseColor,
    float3 ambientColor,
    float3 specularColor,
    float  specularMask,
    float  gloss,
    float3 normal,
    float3 viewDir,
    float3 lightDir,
    float3 lightColor,
    float  shadow
)
{
    float3 colorSum = 0;

    float ssao = 1.0f;
    colorSum += ambientColor * diffuseColor * ssao;

    colorSum += shadow * ApplyLightCommon(
        diffuseColor,
        specularColor,
        specularMask,
        gloss,
        normal,
        viewDir,
        lightDir,
        lightColor);

    return colorSum;
}

#endif

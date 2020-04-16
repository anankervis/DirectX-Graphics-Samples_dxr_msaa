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

#define SHADOW_MODE SHADOW_MODE_SOFT

STRUCT_ALIGN(16) struct ShadeConstants
{
    float3 sunDirection; uint pad0;
    float3 sunColor; uint pad1;
    float3 ambientColor; uint pad2;
};

#ifdef HLSL

# if SHADOW_MODE == SHADOW_MODE_SOFT
# define AREA_LIGHT_CENTER float3(-61, 1296, -38)
# define AREA_LIGHT_EXTENT float3(907, 0, 189)

// https://stackoverflow.com/questions/5149544/can-i-generate-a-random-number-inside-a-pixel-shader
float shadowRandom(float2 p)
{
    float2 K1 = float2(
        23.14069263277926f, // e^pi (Gelfond's constant)
         2.665144142690225f // 2^sqrt(2) (Gelfond-Schneider constant)
    );
    return frac(cos(dot(p, K1)) * 12345.6789f);
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

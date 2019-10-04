#pragma once

#ifndef HLSL
# include "HlslCompat.h"
#endif

#ifdef HLSL
# define STRUCT_ALIGN(x)
#else
# define STRUCT_ALIGN(x) __declspec(align(x))
#endif

STRUCT_ALIGN(16) struct ShadeConstants
{
    float3 sunDirection;
    float3 sunColor;
    float3 ambientColor;
};

#ifdef HLSL

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
    float3    diffuseColor,  
    float3    ambientColor,
    float3    specularColor, 
    float     specularMask,  
    float     gloss,         
    float3    normal,        
    float3    viewDir,       
    float3    lightDir,      
    float3    lightColor     
)
{
    float3 colorSum = 0;

    float ssao = 1.0f;
    colorSum += ambientColor * diffuseColor * ssao;

    float shadow = 1.0f;
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

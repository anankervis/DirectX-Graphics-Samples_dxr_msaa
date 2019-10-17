#pragma once

#include "Math/Vector.h"

// Keep these rules in mind when laying out your constant buffers:
// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-packing-rules

#define OUTPARAM(type, name)    type& name
#define INOUTPARAM(type, name)    type& name

struct float2
{
    float x, y;
};

struct float3
{
    float x, y, z;

    float3() {}
    float3(float x, float y, float z) : x(x), y(y), z(z) {}
    float3(const Vector3 &v) : x(v.GetX()), y(v.GetY()), z(v.GetZ()) {}
};

struct float4
{
    float x, y, z, w;
};

typedef uint32_t uint;

struct uint4
{
    uint32_t x, y, z, w;
};

struct float4x4
{
    float mat[16];
};

inline float3 operator+(const float3 &a, const float3 &b)
{
    return float3(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    );
}

inline float3 operator-(const float3 &a, const float3 &b)
{
    return float3(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    );
}

inline float3 operator*(const float3 &a, const float3 &b)
{
    return float3(
        a.x * b.x,
        a.y * b.y,
        a.z * b.z
    );
}

inline float3 abs(const float3 &a)
{
    return float3(
        std::abs(a.x),
        std::abs(a.y),
        std::abs(a.z)
    );
}

inline float min(float a, float b)
{
    return std::min(a, b);
}

inline float3 min(const float3 &a, const float3 &b)
{
    return float3(
        std::min(a.x, b.x),
        std::min(a.y, b.y),
        std::min(a.z, b.z)
    );
}

inline float max(float a, float b)
{
    return std::max(a, b);
}

inline float3 max(const float3 &a, const float3 &b)
{
    return float3(
        std::max(a.x, b.x),
        std::max(a.y, b.y),
        std::max(a.z, b.z)
    );
}

inline float sign(float v)
{
    if (v < 0)
        return -1;
    return 1;
}

inline float dot(const float3 &a, const float3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float3 cross(const float3 &a, const float3 &b)
{
    return float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

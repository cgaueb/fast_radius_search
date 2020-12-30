// Fast Radius Search Exploiting Ray Tracing Frameworks
// Authors: I. Evangelou, G. Papaioannou, K. Vardis, A. A. Vasilakis

#pragma once

#include <optix.h>
#include <stdint.h>
#include <cstdio>

#include "../../source/sutil/vec_math.hpp"
#include "../../source/cuda_types.hpp"

namespace bvh_radSearch
{
    struct payload_t
    {
        query_t query;
        uint32_t count;

        uint32_t offset;
        int32_t maxDistElemi;
        int32_t foundNeighbors;
        float_t maxDistElemf;
    };

    __device__ __forceinline__ void* unpackPointer(
        uint32_t i0,
        uint32_t i1) noexcept
    {
        const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
        void* ptr = reinterpret_cast<void*>(uptr);
        return ptr;
    }

    __device__ __forceinline__ void packPointer(
        const void* ptr,
        uint32_t& i0,
        uint32_t& i1) noexcept
    {
        const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }

    template<typename T>
    __device__ __forceinline__ T* getPayload(void) noexcept
    {
        const uint32_t u0 = optixGetPayload_0();
        const uint32_t u1 = optixGetPayload_1();
        return reinterpret_cast<T*>(unpackPointer(u0, u1));
    }

    extern "C" { __constant__ Params params; }

    __device__ void findLargestDist(payload_t& payload) noexcept
    {
        payload.maxDistElemi = payload.offset;
        payload.maxDistElemf = params.optixDists[payload.maxDistElemi];

        for (int32_t k = 1; k < params.knn; ++k)
        {
            float_t tmpDist = params.optixDists[payload.offset + k];
            if (tmpDist > payload.maxDistElemf)
            {
                payload.maxDistElemi = payload.offset + k;
                payload.maxDistElemf = tmpDist;
            }
        }
    }

    extern "C" __global__ void __raygen__radSearch_count_bruteforce(void)
    {
        const uint3& idx = optixGetLaunchIndex();
        query_t& query = params.queries[idx.x];
        query.count = 0;

        for (size_t s = 0; s < params.numSamples; ++s)
        {
            const float3 diff = params.samplePos[s] - query.position;
            const float_t t = dot(diff, diff);

            if (t < query.radius * query.radius)
            {
                ++query.count;
            }
        }

        atomicAdd(&params.totalCount[0], query.count);
        atomicMax(&params.maxCount[0], query.count);
        atomicMin(&params.minCount[0], query.count);
    }

    extern "C" __global__ void __intersection__radSearch_count_bruteforce(void) { /* Empty */ }

    extern "C" __global__ void __raygen__radSearch_count(void)
    {
        const uint3 & idx = optixGetLaunchIndex();
        query_t & query = params.queries[idx.x];
        payload_t payload;
        payload.query = query;
        payload.count = 0;

        uint32_t u0, u1;
        packPointer(&payload, u0, u1);

        optixTrace(params.gasHandle,
            query.position, make_float3(1.e-16f),
            0.f, 1.e-16f, 0.f,
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT |
            OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
            0, 4, 0,
            u0, u1);

        query.count = payload.count;
        atomicAdd(&params.totalCount[0], payload.count);
        atomicMax(&params.maxCount[0], payload.count);
        atomicMin(&params.minCount[0], payload.count);
    }

    extern "C" __global__ void __intersection__radSearch_count(void)
    {
        payload_t& payload = *getPayload<payload_t>();

        float3& sample = params.samplePos[optixGetPrimitiveIndex()];

        const float3 diff = sample - optixGetWorldRayOrigin();
        const float_t t = dot(diff, diff);

        if (t < payload.query.radius * payload.query.radius)
        {
            ++payload.count;
        }
    }

    extern "C" __global__ void __raygen__radSearch(void)
    {
        const uint3& idx = optixGetLaunchIndex();
        query_t& query = params.queries[idx.x];
        payload_t payload;
        payload.query = query;
        payload.count = 0;
        payload.offset = idx.x * params.knn;
        payload.maxDistElemi = idx.x * params.knn;
        payload.maxDistElemf = query.radius + 1.f;
        payload.foundNeighbors = 0;
        query.count = params.knn;

        uint32_t u0, u1;
        packPointer(&payload, u0, u1);

        optixTrace(params.gasHandle,
            query.position, make_float3(1.e-16f),
            0.f, 1.e-16f, 0.f,
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT |
            OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
            0, 4, 0,
            u0, u1);
    }

    extern "C" __global__ void __intersection__radSearch(void)
    {
        payload_t& payload = *getPayload<payload_t>();

        float3& sample = params.samplePos[optixGetPrimitiveIndex()];

        const float3 diff = sample - optixGetWorldRayOrigin();
        const float_t t = dot(diff, diff);

        if (t < payload.query.radius * payload.query.radius)
        {
            if (t < payload.maxDistElemf)
            {
                if (payload.foundNeighbors < params.knn)
                {
                    const uint32_t idxToSave = payload.offset + payload.foundNeighbors;
                    params.optixIndices[idxToSave] = optixGetPrimitiveIndex();
                    params.optixDists[idxToSave] = t;

                    if (payload.foundNeighbors == params.knn - 1)
                    {
                        findLargestDist(payload);
                    }

                    ++payload.foundNeighbors;
                }
                else
                {
                    params.optixIndices[payload.maxDistElemi] = optixGetPrimitiveIndex();
                    params.optixDists[payload.maxDistElemi] = t;
                    findLargestDist(payload);
                }
            }
        }
    }

    extern "C" __global__ void __raygen__radSearch_bruteforce(void)
    {
        const uint3& idx = optixGetLaunchIndex();
        query_t& query = params.queries[idx.x];
        payload_t payload;
        payload.query = query;
        payload.count = 0;
        payload.offset = idx.x * params.knn;
        payload.maxDistElemi = idx.x * params.knn;
        payload.maxDistElemf = query.radius + 1.f;
        payload.foundNeighbors = 0;
        query.count = params.knn;

        uint32_t u0, u1;
        packPointer(&payload, u0, u1);

        for (size_t s = 0; s < params.numSamples; ++s)
        {
            const float3 diff = params.samplePos[s] - query.position;
            const float_t t = dot(diff, diff);

            if (t < query.radius * query.radius)
            {
                if (t < payload.maxDistElemf)
                {
                    if (payload.foundNeighbors < params.knn)
                    {
                        const uint32_t idxToSave = payload.offset + payload.foundNeighbors;
                        params.optixIndices[idxToSave] = s;
                        params.optixDists[idxToSave] = t;

                        if (payload.foundNeighbors == params.knn - 1)
                        {
                            findLargestDist(payload);
                        }

                        ++payload.foundNeighbors;
                    }
                    else
                    {
                        params.optixIndices[payload.maxDistElemi] = s;
                        params.optixDists[payload.maxDistElemi] = t;
                        findLargestDist(payload);
                    }
                }
            }
        }
    }

    extern "C" __global__ void __intersection__radSearch_bruteforce(void) { /* Empty */ }
}
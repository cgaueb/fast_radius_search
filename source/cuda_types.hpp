// Fast Radius Search Exploiting Ray Tracing Frameworks
// Authors: I. Evangelou, G. Papaioannou, K. Vardis, A. A. Vasilakis
#pragma once

struct HitGroupData { int32_t dummy; };
struct RayGenData { int32_t dummy; };
struct MissData { int32_t dummy; };

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

struct statistics_t
{
    uint32_t totalGather;
    uint32_t maxGather;
    uint32_t minGather;

    float_t avgGather;
};

struct query_t
{
    float3 position;
    float_t radius;
    uint32_t count;
};

struct Params
{
    OptixTraversableHandle gasHandle;

    query_t* queries;
    float3* samplePos;

    uint32_t numSamples;
    uint32_t knn;

    uint32_t* totalCount;
    uint32_t* minCount;
    uint32_t* maxCount;

    int32_t* optixIndices;
    float_t* optixDists;
};

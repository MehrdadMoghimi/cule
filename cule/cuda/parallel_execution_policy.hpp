#pragma once

#include <cule/config.hpp>

#include <agency/cuda.hpp>

#include <cuda.h>

namespace cule
{
namespace cuda
{

class parallel_execution_policy : public agency::cuda::parallel_execution_policy
{
private:
    using super_t = agency::cuda::parallel_execution_policy;

public:
    parallel_execution_policy()
    {
        CULE_ERRCHK(cudaStreamCreate(&stream));
    }

    ~parallel_execution_policy()
    {
        CULE_ERRCHK(cudaStreamDestroy(stream));
    }

    void sync() const
    {
        CULE_ERRCHK(cudaStreamSynchronize(getStream()));
    }

    // Launch all subsequent kernels on a caller-owned stream (e.g. torch's
    // current stream) instead of the internal one. Handle 0 selects the
    // legacy default stream. Passing the caller's stream removes the implicit
    // legacy-stream ordering dependency and makes the dispatch sequence
    // capturable by CUDA graphs.
    void set_external_stream(const cudaStream_t& externalStream)
    {
        external_stream = externalStream;
        use_external_stream = true;
    }

    void insert_other_stream(const cudaStream_t& otherStream) const
    {
        cudaEvent_t event;

        CULE_ERRCHK(cudaEventCreate(&event));
        CULE_ERRCHK(cudaEventRecord(event, otherStream));
        CULE_ERRCHK(cudaStreamWaitEvent(stream, event, 0));
        CULE_ERRCHK(cudaEventDestroy(event));
    }

    void insert_this_stream(const cudaStream_t& otherStream) const
    {
        cudaEvent_t event;

        CULE_ERRCHK(cudaEventCreate(&event));
        CULE_ERRCHK(cudaEventRecord(event, stream));
        CULE_ERRCHK(cudaStreamWaitEvent(otherStream, event, 0));
        CULE_ERRCHK(cudaEventDestroy(event));
    }

    cudaStream_t getStream() const
    {
        return use_external_stream ? external_stream : stream;
    }

private:
    cudaStream_t stream;
    cudaStream_t external_stream = nullptr;
    bool use_external_stream = false;
};

const parallel_execution_policy par{};

} // end namespace cuda
} // end namespace cule


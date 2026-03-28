#ifndef LLM_CPP_TIMER_HPP
#define LLM_CPP_TIMER_HPP

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#include "errorUtils.hpp"

// abstract Timer class
class Timer
{
public:
    virtual void Start() = 0;
    virtual void Stop() = 0;

    virtual float ElapsedMillis() const = 0;
    virtual float ElapsedSeconds() const {
        return ElapsedMillis() / 1000.0f;
    }

    virtual float TotalElapsedMillis() const = 0;
    virtual float TotalElapsedSeconds() const {
        return TotalElapsedMillis() / 1000.0f;
    }

    virtual void Reset() = 0;
    virtual ~Timer() = default;
};

// Timer implementation for GPU using CUDA events
class TimerGPU : public Timer
{
private:
    float totalElapsedMillis = 0.0f;
    float elapsedMillis = 0.0f;
    bool running = false;
    cudaEvent_t startEvent, stopEvent;

public:
    TimerGPU() {
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&stopEvent));
    }

    ~TimerGPU() override {
        CUDA_CHECK(cudaEventDestroy(startEvent));
        CUDA_CHECK(cudaEventDestroy(stopEvent));
    }

    void Start() override {
        if (running) return;
        CUDA_CHECK(cudaEventRecord(startEvent, 0));
        running = true;
    }

    void Stop() override {
        if (!running) return;
        CUDA_CHECK(cudaEventRecord(stopEvent, 0));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));

        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

        elapsedMillis = milliseconds;
        totalElapsedMillis += milliseconds;
        running = false;
    }

    float ElapsedMillis() const override { return elapsedMillis; }
    float TotalElapsedMillis() const override { return totalElapsedMillis; }

    void Reset() override {
        totalElapsedMillis = 0.0f;
        elapsedMillis = 0.0f;
        running = false;
    }
};

// Timer implementation for CPU using std::chrono
class TimerCPU : public Timer
{
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    float totalElapsedMillis = 0.0f;
    float elapsedMillis = 0.0f;
    bool running = false;

public:
    TimerCPU() = default;

    void Start() override {
        if (running) return;
        startTime = std::chrono::high_resolution_clock::now();
        running = true;
    }

    void Stop() override {
        if (!running) return;
        auto stopTime = std::chrono::high_resolution_clock::now();
        elapsedMillis = std::chrono::duration<float, std::milli>(stopTime - startTime).count();
        totalElapsedMillis += elapsedMillis;
        running = false;
    }

    float ElapsedMillis() const override { return elapsedMillis; }
    float TotalElapsedMillis() const override { return totalElapsedMillis; }

    void Reset() override {
        totalElapsedMillis = 0.0f;
        elapsedMillis = 0.0f;
        running = false;
    }
};

// TimerManager to safely manage a timer instance
class TimerManager
{
private:
    Timer* timer = nullptr;

public:
    TimerManager() = default;

    void SetTimer(Timer* t) {
        timer = t;
    }

    void Start() {
        if (timer) timer->Start();
    }

    void Stop() {
        if (timer) timer->Stop();
    }

    float ElapsedMillis() const {
        return timer ? timer->ElapsedMillis() : 0.0f;
    }

    float ElapsedSeconds() const {
        return timer ? timer->ElapsedSeconds() : 0.0f;
    }

    float TotalElapsedMillis() const {
        return timer ? timer->TotalElapsedMillis() : 0.0f;
    }

    float TotalElapsedSeconds() const {
        return timer ? timer->TotalElapsedSeconds() : 0.0f;
    }

    void Reset() {
        if (timer) timer->Reset();
    }
};

#endif //LLM_CPP_TIMER_HPP

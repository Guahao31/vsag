#pragma once

#include <time.h>
#include <sys/time.h>
#include <inttypes.h>

static inline uint64_t NowNanos()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

static inline uint64_t NowMicros()
{
	struct timeval tv;
	gettimeofday(&tv, nullptr);
	return (uint64_t)(tv.tv_sec) * 1000000 + tv.tv_usec;
}

static inline uint64_t ElapsedNanos(uint64_t start_time)
{
	uint64_t now = NowNanos();
	return now - start_time;
}

static inline uint64_t ElapsedMicros(uint64_t start_time)
{
	uint64_t now = NowMicros();
	return now - start_time;
}
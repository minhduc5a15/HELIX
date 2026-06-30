#pragma once

#include <chrono>

namespace helix {
    namespace benchmark {

        class Timer {
        public:
            void start() { start_time_ = std::chrono::steady_clock::now(); }

            void stop() { end_time_ = std::chrono::steady_clock::now(); }

            double elapsed_ns() const {
                return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_ - start_time_).count();
            }

            double elapsed_us() const { return elapsed_ns() / 1e3; }

            double elapsed_ms() const { return elapsed_ns() / 1e6; }

        private:
            std::chrono::steady_clock::time_point start_time_;
            std::chrono::steady_clock::time_point end_time_;
        };

    }  // namespace benchmark
}  // namespace helix

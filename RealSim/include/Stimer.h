#pragma once
#include <chrono>
#include "printconfig.h"
#include "types.h"

namespace RealSim {

class c_timer {

public:
    void clear() {
        begin.clear();
        end.clear();
    }

    void start() {
        begin.emplace_back(std::chrono::steady_clock::now());
    }

    void stop() {
        end.emplace_back(std::chrono::steady_clock::now());
    }

    real elapsed_ms() const {
        if(begin.empty() || end.empty()) return 0.0;
        else return std::chrono::duration_cast<std::chrono::microseconds>(end.back() - begin.back()).count() / 1e3;
    }

    unsigned times() const{
        if(begin.empty() || end.empty()) return 0;

        if(begin.size() != end.size()) {
            std::cout << RED << "WARNING in c_timer: inconsistent timer count" << RESET << std::endl;
            return std::min(begin.size(), end.size());
        }

        return begin.size();
    }

    real total_elapsed_ms() const{
        real res = 0.0;
        unsigned size = times();
        for(unsigned i=0; i<size; i++)
        {
            res += std::chrono::duration_cast<std::chrono::microseconds>(end[i] - begin[i]).count() / 1e3;
        }

        return res;
    }

    real mean_elapsed_ms() const{
        if(begin.empty() || end.empty()) return 0.0;

        return total_elapsed_ms()/times();
    }

protected:
    std::vector<std::chrono::steady_clock::time_point> begin;
    std::vector<std::chrono::steady_clock::time_point> end;
};

}
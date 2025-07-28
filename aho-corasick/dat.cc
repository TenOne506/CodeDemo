#include "dat.h"
#include <sys/types.h>
#include <cstddef>
#include <cstdint>
#include <vector>

DoubleArrayTrie::DoubleArrayTrie() = default;
DoubleArrayTrie::~DoubleArrayTrie() = default;

bool DoubleArrayTrie::IsUsed(int cur_pos) {
    return free_list_.find(cur_pos) != free_list_.end();
}

int DoubleArrayTrie::CheckAndExpand(int next_pos) {
    size_t old_size = base_.size();
    if (next_pos > old_size) {
        base_.resize(next_pos + 1);
        check_.resize(next_pos + 1);
        for (size_t j = old_size; j < next_pos + 1; ++j) {
            free_list_.insert(j);
        }
        return next_pos;
    }
    if (!IsUsed(next_pos)) { return next_pos; }
    return -1;
}

void DoubleArrayTrie::GenTrie(const std::vector<Sentence>& words) {
    check_.resize(2, 0);
    base_.resize(2, 0);
    for (auto word : words) {
        int start_pos = 1;
        for (auto c : word) {
            auto unicode_c = static_cast<uint32_t>(c);
            uint32_t next_pos = base_[start_pos] + c;
        }
    }
}
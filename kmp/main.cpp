#include <string.h>
#include <cstddef>
#include <vector>

std::vector<size_t> calculate_prefix(const std::string& s) {
    size_t n = s.size();
    std::vector<size_t> prefix(n, 0);
    for (size_t i = 1; i < n; ++i) {
        size_t j = prefix[i - 1];
        while (j > 0 && s[i] != s[j]) {
            j = prefix[j - 1];
        }
        if (s[i] == s[j]) { ++j; }
        prefix[i] = j;
    }
    return prefix;
}
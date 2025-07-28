#include <iostream>
#include <unordered_map>

int main() {
    std::unordered_map<int, std::string> um;

    // 设置最大负载因子为 0.5
    um.max_load_factor(0.5);

    // 插入元素
    for (int i = 0; i < 10; ++i) {
        um[i] = "value" + std::to_string(i);
        std::cout << "Size: " << um.size() << ", Bucket count: " << um.bucket_count() << std::endl;
    }

    // 手动触发扩容
    um.rehash(20);
    std::cout << "After rehash, Bucket count: " << um.bucket_count() << std::endl;

    return 0;
}
#include <iostream>
#include <memory>
#include <vector>

// 将所有测试封装在一个函数里
void runSizeofTests() {
    // 1. 测试 int a = 10;
    {
        int a = 10;
        std::cout << "当 a 是 int 类型 (a = 10):" << std::endl;
        std::cout << "  sizeof(a) = " << sizeof(a) << " 字节" << std::endl;
        std::cout << "  (通常 sizeof(int) = " << sizeof(int) << " 字节)" << std::endl;
        std::cout << std::endl;
    }

    // 2. 测试 int a[10];
    {
        int a[10];
        std::cout << "当 a 是 int 数组 (int a[10]):" << std::endl;
        std::cout << "  sizeof(a) = " << sizeof(a) << " 字节" << std::endl;
        std::cout << "  (计算: 10 个元素 * 每个 int " << sizeof(int) << " 字节 = " << (10 * sizeof(int)) << " 字节)"
                  << std::endl;
        std::cout << "  sizeof(a[0]) = " << sizeof(a[0]) << " 字节 (单个元素大小)" << std::endl;
        std::cout << std::endl;
    }

    // 3. 测试 int *a[10]; (指针数组)
    {
        int* a[10];

        std::cout << "当 a 是 int* 数组 (int *a[10]):" << std::endl;
        std::cout << "  sizeof(a) = " << sizeof(a) << " 字节" << std::endl;
        std::cout << "  (计算: 10 个元素 * 每个指针 " << sizeof(int*) << " 字节 = " << (10 * sizeof(int*)) << " 字节)"
                  << std::endl;
        std::cout << "  sizeof(a[0]) = " << sizeof(a[0]) << " 字节 (单个指针的大小)" << std::endl;
        std::cout << "  sizeof(int*) = " << sizeof(int*) << " 字节 (指针类型大小)" << std::endl;
        std::cout << std::endl;

        int some_value = 42;
        int* ptr = &some_value;
        std::cout << "额外演示 - 单个指针变量:" << std::endl;
        std::cout << "  sizeof(ptr) = " << sizeof(ptr) << " 字节 (指针变量本身的大小)" << std::endl;
        std::cout << "  sizeof(*ptr) = " << sizeof(*ptr) << " 字节 (指针所指向对象的大小，即 int)" << std::endl;
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "sizeof(unique_ptr<int>): " << sizeof(std::unique_ptr<int>) << " bytes\n";
    std::cout << "sizeof(shared_ptr<int>): " << sizeof(std::shared_ptr<int>) << " bytes\n";
    std::cout << "sizeof(weak_ptr<int>):   " << sizeof(std::weak_ptr<int>) << " bytes\n";
    std::cout << "sizeof(vector<int>):     " << sizeof(std::vector<int>) << " bytes\n";
    runSizeofTests();
    return 0;
}
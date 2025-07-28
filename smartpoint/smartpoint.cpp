#include <iostream>

template <typename T>
class SharedPtr {
 private:
    T* ptr;         // 指向动态分配的内存
    int* refCount;  // 引用计数

    // 释放资源
    void release() {
        if (refCount) {
            (*refCount)--;
            if (*refCount == 0) {
                delete ptr;
                delete refCount;
                std::cout << "Resource released!" << std::endl;
            }
        }
    }

 public:
    // 构造函数
    SharedPtr(T* p = nullptr) : ptr(p), refCount(new int(1)) {
        std::cout << "SharedPtr created! Ref count: " << *refCount << std::endl;
    }

    // 拷贝构造函数
    SharedPtr(const SharedPtr& other) : ptr(other.ptr), refCount(other.refCount) {
        (*refCount)++;
        std::cout << "SharedPtr copied! Ref count: " << *refCount << std::endl;
    }

    // 拷贝赋值运算符
    SharedPtr& operator=(const SharedPtr& other) {
        if (this != &other) {  // 防止自赋值
            release();         // 释放当前资源
            ptr = other.ptr;
            refCount = other.refCount;
            (*refCount)++;
            std::cout << "SharedPtr assigned! Ref count: " << *refCount << std::endl;
        }
        return *this;
    }

    // 移动构造函数 (C++11)
    SharedPtr(SharedPtr&& other) noexcept : ptr(other.ptr), refCount(other.refCount) {
        other.ptr = nullptr;
        other.refCount = nullptr;
        std::cout << "SharedPtr moved! Ref count: " << *refCount << std::endl;
    }

    // 移动赋值运算符 (C++11)
    SharedPtr& operator=(SharedPtr&& other) noexcept {
        if (this != &other) {  // 防止自赋值
            release();         // 释放当前资源
            ptr = other.ptr;
            refCount = other.refCount;
            other.ptr = nullptr;
            other.refCount = nullptr;
            std::cout << "SharedPtr move-assigned! Ref count: " << *refCount << std::endl;
        }
        return *this;
    }

    // 析构函数
    ~SharedPtr() { release(); }

    // 解引用运算符
    T& operator*() const { return *ptr; }

    // 箭头运算符
    T* operator->() const { return ptr; }

    // 获取引用计数
    int use_count() const { return refCount ? *refCount : 0; }

    // 检查是否为空
    bool is_null() const { return ptr == nullptr; }
};

// 测试代码
int main() {
    // 创建一个 SharedPtr
    SharedPtr<int> p1(new int(10));
    std::cout << "p1 value: " << *p1 << ", ref count: " << p1.use_count() << std::endl;

    // 拷贝构造
    SharedPtr<int> p2 = p1;
    std::cout << "p2 value: " << *p2 << ", ref count: " << p2.use_count() << std::endl;

    // 拷贝赋值
    SharedPtr<int> p3;
    p3 = p2;
    std::cout << "p3 value: " << *p3 << ", ref count: " << p3.use_count() << std::endl;

    // 移动构造
    SharedPtr<int> p4 = std::move(p3);
    std::cout << "p4 value: " << *p4 << ", ref count: " << p4.use_count() << std::endl;
    std::cout << "p3 is null: " << p3.is_null() << std::endl;

    // 移动赋值
    SharedPtr<int> p5;
    p5 = std::move(p4);
    std::cout << "p5 value: " << *p5 << ", ref count: " << p5.use_count() << std::endl;
    std::cout << "p4 is null: " << p4.is_null() << std::endl;

    return 0;
}
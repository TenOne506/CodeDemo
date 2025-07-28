#include <iostream>
#include <memory>

int main() {
    // 使用 std::make_shared 创建 shared_ptr
    std::shared_ptr<int> ptr1 = std::make_shared<int>(42);

    // 复制 shared_ptr，引用计数增加
    std::shared_ptr<int> ptr2 = ptr1;

    std::cout << "ptr1 use count: " << ptr1.use_count() << std::endl;  // 输出 2

    // 重置 ptr2，引用计数减少
    ptr2.reset();
    std::cout << "ptr1 use count: " << ptr1.use_count() << std::endl;  // 输出 1

    // ptr1 离开作用域，引用计数归零，内存自动释放
    return 0;
}
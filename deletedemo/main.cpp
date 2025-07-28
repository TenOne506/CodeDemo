#include <iostream>

class MyClass {
 public:
    MyClass() { std::cout << "Constructor\n"; }

    ~MyClass() { std::cout << "Destructor\n"; }
};

int main() {
    // 单个对象
    MyClass* p = new MyClass;
    delete p;  // 正确

    // 数组对象
    MyClass* arr = new MyClass[3];
    delete[] arr;  // 正确

    // 错误示例
    // delete[] p;   // 未定义行为
    // delete arr;   // 未定义行为

    return 0;
}
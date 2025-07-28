#include <iostream>

class Test {
    constexpr static int a = 100;

    int b = 200;

 public:
    void PrintA() { std::cout << "a=" << a << std::endl; }

    void PrintB() { std::cout << "b=" << b << std::endl; }

    void FuncA() { std::cout << "hello world..." << std::endl; }

    static void FuncB() { std::cout << "a=" << a << std::endl; }

    virtual void FuncC() { std::cout << "hello world 3!" << std::endl; }
};

int main() {
    Test* ptr = nullptr;

    // 下面几个函数的调用， 分别会发生什么， 给出输出预期以及具体原因.

    // ptr-> PrintA(); //输出a=100
    // ptr-> PrintB(); //报错
    // ptr-> FuncA(); //输出hello world...
    // ptr-> FuncB(); //输出a=100

    // ptr-> FuncC();//报错 昨天这里说错了在他提醒下说对了
    return 0;
}
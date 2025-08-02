#include <iostream>

void myswap(int& a, int& b) {
    a = a ^ b;
    b = a ^ b;
    a = a ^ b;
}

int main() {
    int a = 1024;
    int b = 4096;
    std::cout << "before swap\n";
    std::cout << "a= " << a << " b=" << b << std::endl;
    myswap(a, b);
    std::cout << "after swap\n";
    std::cout << "a= " << a << " b=" << b << std::endl;
}
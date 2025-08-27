#include <iostream>
#include <memory>
#include <vector>

int main() {
    std::cout << "sizeof(unique_ptr<int>): " << sizeof(std::unique_ptr<int>) << " bytes\n";
    std::cout << "sizeof(shared_ptr<int>): " << sizeof(std::shared_ptr<int>) << " bytes\n";
    std::cout << "sizeof(weak_ptr<int>):   " << sizeof(std::weak_ptr<int>) << " bytes\n";
    std::cout << "sizeof(vector<int>):     " << sizeof(std::vector<int>) << " bytes\n";
    return 0;
}
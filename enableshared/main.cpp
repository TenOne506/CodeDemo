#include <iostream>
#include <memory>

class A : public std::enable_shared_from_this<A> {
  public:
  std::shared_ptr<A> get_shared() { return shared_from_this(); }

  void print() { std::cout << "A::print()" << std::endl; }
};

int main() {
  std::shared_ptr<A> ptr1 = std::make_shared<A>();
  std::shared_ptr<A> ptr2 = ptr1->get_shared();

  std::cout << "ptr1 use count: " << ptr1.use_count() << std::endl;// 输出 2
  std::cout << "ptr2 use count: " << ptr2.use_count() << std::endl;// 输出 2

  ptr2->print();// 正常调用
  return 0;
}

#include <iostream>
#include <string>

class MyClass {
  private:
  std::string name;
  int *data;// 动态分配的资源

  public:
  // 1. 构造函数
  MyClass(const std::string &name, int value) : name(name) {
    data = new int(value);// 动态分配内存
    std::cout << "Constructor called for " << name << std::endl;
  }

  // 2. 析构函数
  ~MyClass() {
    delete data;// 释放动态分配的内存
    std::cout << "Destructor called for " << name << std::endl;
  }

  // 3. 拷贝构造函数
  MyClass(const MyClass &other) : name(other.name + " (copy)") {
    data = new int(*other.data);// 深拷贝
    std::cout << "Copy constructor called for " << name << std::endl;
  }

  // 4. 拷贝赋值运算符
  MyClass &operator=(const MyClass &other) {
    if (this != &other) {// 防止自赋值
      name = other.name + " (copy assigned)";
      delete data;                // 释放原有资源
      data = new int(*other.data);// 深拷贝
      std::cout << "Copy assignment operator called for " << name << std::endl;
    }
    return *this;
  }

  // 5. 移动构造函数（C++11 及以上）
  MyClass(MyClass &&other) noexcept : name(std::move(other.name)), data(other.data) {
    other.data = nullptr;// 将原对象置为空
    std::cout << "Move constructor called for " << name << std::endl;
  }

  // 6. 移动赋值运算符（C++11 及以上）
  MyClass &operator=(MyClass &&other) noexcept {
    if (this != &other) {// 防止自赋值
      name = std::move(other.name);
      delete data;         // 释放原有资源
      data = other.data;   // 转移资源
      other.data = nullptr;// 将原对象置为空
      std::cout << "Move assignment operator called for " << name << std::endl;
    }
    return *this;
  }

  // 7. 其他成员函数
  void print() const { std::cout << "Name: " << name << ", Data: " << (data ? *data : 0) << std::endl; }

  void setData(int value) {
    if (data) { *data = value; }
  }
};

int main() {
  // 测试构造函数
  MyClass obj1("Object1", 10);
  obj1.print();

  // 测试拷贝构造函数
  MyClass obj2 = obj1;
  obj2.print();

  // 测试拷贝赋值运算符
  MyClass obj3("Object3", 20);
  obj3 = obj1;
  obj3.print();

  // 测试移动构造函数
  MyClass obj4 = std::move(obj1);
  obj4.print();
  obj1.print();// obj1 的资源已被转移

  // 测试移动赋值运算符
  MyClass obj5("Object5", 30);
  obj5 = std::move(obj2);
  obj5.print();
  obj2.print();// obj2 的资源已被转移

  return 0;
}
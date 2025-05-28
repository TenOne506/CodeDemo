#include <iostream>
#include <mutex>

class Singleton {
  private:
  // 私有构造函数，防止外部创建实例
  Singleton() { std::cout << "Singleton instance created!" << std::endl; }

  // 禁用拷贝构造函数和赋值运算符
  Singleton(const Singleton &) = delete;
  Singleton &operator=(const Singleton &) = delete;

  // 静态成员变量，保存唯一实例
  static Singleton *instance;

  // 互斥锁，用于线程安全
  static std::mutex mtx;

  public:
  // 静态成员函数，提供全局访问点
  static Singleton *getInstance() {
    // 双重检查锁定，确保线程安全
    if (instance == nullptr) {
      std::lock_guard<std::mutex> lock(mtx);// 加锁
      if (instance == nullptr) { instance = new Singleton(); }
    }
    return instance;
  }

  // 示例成员函数
  void doSomething() { std::cout << "Doing something!" << std::endl; }

  // 析构函数
  ~Singleton() { std::cout << "Singleton instance destroyed!" << std::endl; }
};

// 初始化静态成员变量
Singleton *Singleton::instance = nullptr;
std::mutex Singleton::mtx;

int main() {
  // 获取单例实例
  Singleton *singleton = Singleton::getInstance();
  singleton->doSomething();

  // 再次获取单例实例
  Singleton *anotherSingleton = Singleton::getInstance();
  anotherSingleton->doSomething();

  // 检查是否是同一个实例
  if (singleton == anotherSingleton) { std::cout << "Both pointers point to the same instance!" << std::endl; }

  return 0;
}
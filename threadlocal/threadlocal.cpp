#include <iostream>
#include <thread>

thread_local int counter = 0;// 每个线程都有自己的 counter

void increment() {
  counter++;
  std::cout << "Thread ID: " << std::this_thread::get_id() << ", Counter: " << counter << std::endl;
}

int main() {
  std::thread t1(increment);
  std::thread t2(increment);
  t1.join();
  t2.join();
  return 0;
}
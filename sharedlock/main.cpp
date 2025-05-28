#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <thread>

std::shared_mutex rw_mutex;

void readData(int id) {
  std::shared_lock<std::shared_mutex> lock(rw_mutex);// 共享锁
  std::cout << "Reader " << id << " is reading data." << std::endl;
}

void writeData(int id) {
  std::unique_lock<std::shared_mutex> lock(rw_mutex);// 独占锁
  std::cout << "Writer " << id << " is writing data." << std::endl;
}

int main() {
  std::thread t1(readData, 1);
  std::thread t2(readData, 2);
  std::thread t3(writeData, 3);

  t1.join();
  t2.join();
  t3.join();

  return 0;
}

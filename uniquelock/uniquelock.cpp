#include <iostream>
#include <mutex>
#include <thread>

std::mutex mtx;

void printNumbers(int id) {
    std::unique_lock<std::mutex> lock(mtx);  // 构造时锁定互斥锁
    std::cout << "Thread " << id << " is running." << std::endl;
    lock.unlock();  // 手动解锁

    // 做一些不需要锁的工作
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    lock.lock();  // 再次手动锁定
    std::cout << "Thread " << id << " is finished." << std::endl;
    // 析构时自动解锁
}

int main() {
    std::thread t1(printNumbers, 1);
    std::thread t2(printNumbers, 2);

    t1.join();
    t2.join();

    return 0;
}
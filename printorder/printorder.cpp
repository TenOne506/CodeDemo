#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

std::mutex mtx;
std::condition_variable cv;
int current = 1;
const int loopCount = 5;  // 设置循环次数

void printNumber(int num) {
    for (int i = 0; i < loopCount; ++i) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [num] { return current == num; });
        std::cout << num << std::endl;
        current = (current % 3) + 1;  // 循环更新 current 的值（1 -> 2 -> 3 -> 1）
        cv.notify_all();              // 通知其他线程
    }
}

int main() {
    std::thread t1(printNumber, 1);
    std::thread t2(printNumber, 2);
    std::thread t3(printNumber, 3);

    t1.join();
    t2.join();
    t3.join();
    return 0;
}
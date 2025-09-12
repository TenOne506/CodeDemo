#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

const int BUFFER_SIZE = 5;   // 缓冲区大小
std::queue<int> buffer;      // 缓冲区
std::mutex mtx;              // 互斥锁
std::condition_variable cv;  // 单个条件变量 (更常见)

void producer(int id) {
    for (int i = 0; i < 5; ++i) {
        std::unique_lock<std::mutex> lock(mtx);
        // 等待缓冲区未满
        cv.wait(lock, []() { return buffer.size() < BUFFER_SIZE; });

        auto data = id * 100 + i;
        buffer.emplace(data);

        // ✅ 在锁的保护下进行输出，避免交错
        std::cout << "Producer: " << id << " produces data: " << data << std::endl;

        // 解锁前通知（或解锁后通知都可以）
        lock.unlock();
        cv.notify_one();  // 通知一个等待的消费者
    }
}

void consumer(int id) {
    for (int i = 0; i < 5; ++i) {
        std::unique_lock<std::mutex> lock(mtx);
        // 等待缓冲区非空
        cv.wait(lock, []() { return !buffer.empty(); });

        auto data = buffer.front();
        buffer.pop();

        // ✅ 在锁的保护下进行输出
        std::cout << "Consumer: " << id << " consumes data: " << data << std::endl;

        lock.unlock();
        cv.notify_one();  // 通知一个等待的生产者
    }
}

// ------------------------ 使用示例 ------------------------
int main() {
    std::thread p1(producer, 1);
    std::thread p2(producer, 2);
    std::thread c1(consumer, 1);
    std::thread c2(consumer, 2);

    p1.join();
    p2.join();
    c1.join();
    c2.join();

    return 0;
}
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

const int BUFFER_SIZE = 5;                         // 缓冲区大小
std::queue<int> buffer;                            // 缓冲区
std::mutex mtx;                                    // 互斥锁
std::condition_variable cv_producer, cv_consumer;  // 条件变量

void producer(int id) {
    for (int i = 0; i < 5; ++i) {
        std::unique_lock<std::mutex> lock(mtx);

        cv_producer.wait(lock, []() { return buffer.size() < BUFFER_SIZE; });
        int data = id * 100 + i;
        buffer.push(data);
        std::cout << "Producer " << id << " produced: " << data << std::endl;
        lock.unlock();
        cv_consumer.notify_one();
    }
}

void consumer(int id) {
    for (int i = 0; i < 5; ++i) {
        std::unique_lock<std::mutex> lock(mtx);

        cv_consumer.wait(lock, []() { return !buffer.empty(); });
        int data = buffer.front();
        buffer.pop();
        std::cout << "consumer" << id << " consumer " << data << std::endl;
        lock.unlock();
        cv_producer.notify_one();
    }
}

int main() {
    // 创建生产者和消费者线程
    std::thread producers[2];
    std::thread consumers[2];

    for (int i = 0; i < 2; ++i) {
        producers[i] = std::thread(producer, i + 1);
        consumers[i] = std::thread(consumer, i + 1);
    }

    // 等待所有线程完成
    for (int i = 0; i < 2; ++i) {
        producers[i].join();
        consumers[i].join();
    }

    return 0;
}
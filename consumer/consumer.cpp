// #include <iostream>
// #include <thread>
// #include <queue>
// #include <mutex>
// #include <condition_variable>
// #include <chrono>

// const int BUFFER_SIZE = 5; // 缓冲区大小
// std::queue<int> buffer;    // 缓冲区
// std::mutex mtx;            // 互斥锁
// std::condition_variable cv_producer, cv_consumer; // 条件变量

// // 生产者函数
// void producer(int id) {
//     for (int i = 0; i < 10; ++i) {
//         std::this_thread::sleep_for(std::chrono::milliseconds(100)); //
//         模拟生产耗时 std::unique_lock<std::mutex> lock(mtx);

//         // 如果缓冲区已满，等待消费者消费
//         cv_producer.wait(lock, [] { return buffer.size() < BUFFER_SIZE; });

//         // 生产数据并放入缓冲区
//         int data = id * 100 + i;
//         buffer.push(data);
//         std::cout << "Producer " << id << " produced: " << data << std::endl;

//         lock.unlock();
//         cv_consumer.notify_all(); // 通知消费者
//     }
// }

// // 消费者函数
// void consumer(int id) {
//     for (int i = 0; i < 10; ++i) {
//         std::this_thread::sleep_for(std::chrono::milliseconds(200)); //
//         模拟消费耗时 std::unique_lock<std::mutex> lock(mtx);

//         // 如果缓冲区为空，等待生产者生产
//         cv_consumer.wait(lock, [] { return !buffer.empty(); });

//         // 从缓冲区取出数据并消费
//         int data = buffer.front();
//         buffer.pop();
//         std::cout << "Consumer " << id << " consumed: " << data << std::endl;

//         lock.unlock();
//         cv_producer.notify_all(); // 通知生产者
//     }
// }

// int main() {
//     // 创建生产者和消费者线程
//     std::thread producers[2];
//     std::thread consumers[2];

//     for (int i = 0; i < 2; ++i) {
//         producers[i] = std::thread(producer, i + 1);
//         consumers[i] = std::thread(consumer, i + 1);
//     }

//     // 等待所有线程完成
//     for (int i = 0; i < 2; ++i) {
//         producers[i].join();
//         consumers[i].join();
//     }

//     return 0;
// }
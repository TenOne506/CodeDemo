#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

class ReadWriteLock {
 public:
    ReadWriteLock() = default;

    void read_lock() {
        std::unique_lock<std::mutex> lock(mtx_);
        read_cond_.wait(lock, [&] { return num_writer == 0 && pending_writers == 0; });
        num_reader++;
    }

    void read_unlock() {
        std::unique_lock<std::mutex> lock(mtx_);
        num_reader--;
        if (num_reader == 0) {
            write_cond_.notify_one();  // 唤醒一个写者
        }
    }

    void write_lock() {
        std::unique_lock<std::mutex> lock(mtx_);
        pending_writers++;
        write_cond_.wait(lock, [&] { return num_reader == 0 && num_writer == 0; });
        num_writer++;
        pending_writers--;
    }

    void write_unlock() {
        std::unique_lock<std::mutex> lock(mtx_);
        num_writer--;
        if (pending_writers > 0) {
            write_cond_.notify_one();  // 优先唤醒写者
        } else {
            read_cond_.notify_all();  // 无写者等待时唤醒读者
        }
    }

 private:
    std::mutex mtx_;
    std::condition_variable read_cond_;
    std::condition_variable write_cond_;
    int num_reader{};
    int num_writer{};
    int pending_writers{};  // 等待中的写者数量
};

void TestReadWriteLock() {
    ReadWriteLock rw_lock;
    int shared_data = 0;  // 共享数据
    const int kReaders = 5;
    const int kWriters = 3;
    const int kIterations = 15;

    auto reader = [&](int id) {
        for (int i = 0; i < kIterations; ++i) {
            rw_lock.read_lock();
            // 模拟读取操作
            std::cout << "Reader " << id << " reads: " << shared_data << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            rw_lock.read_unlock();
        }
    };

    auto writer = [&](int id) {
        for (int i = 0; i < kIterations; ++i) {
            rw_lock.write_lock();
            // 模拟写入操作
            shared_data++;
            std::cout << "Writer " << id << " writes: " << shared_data << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            rw_lock.write_unlock();
        }
    };

    std::vector<std::thread> threads;

    // 创建读者线程
    threads.reserve(kReaders);
    for (int i = 0; i < kReaders; ++i) {
        threads.emplace_back(reader, i);
    }

    // 创建写者线程
    for (int i = 0; i < kWriters; ++i) {
        threads.emplace_back(writer, i);
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Final shared_data value: " << shared_data << std::endl;
    std::cout << "Expected value: " << kWriters * kIterations << std::endl;
}

int main() {
    TestReadWriteLock();
    return 0;
}
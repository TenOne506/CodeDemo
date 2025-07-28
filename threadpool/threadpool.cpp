#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <list>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>

class NoneCopy {
 public:
    ~NoneCopy() {}

 protected:
    NoneCopy() {}

 private:
    NoneCopy(const NoneCopy&) = delete;
    NoneCopy& operator=(const NoneCopy&) = delete;
};

using Task = std::packaged_task<void()>;

class ThreadPool : public NoneCopy {
 public:
    ~ThreadPool() { stop(); }

    static ThreadPool& instance() {
        static ThreadPool ins;
        return ins;
    }

    int idleThreadCount() { return thread_num_; }

    template <class F, class... Args>
    auto commit(F&& f, Args&&... args) -> std::future<decltype(std::forward<F>(f)(std::forward<Args>(args)...))> {
        using RetType = decltype(std::forward<F>(f)(std::forward<Args>(args)...));
        if (stop_.load()) { return std::future<RetType>{}; }

        auto task = std::make_shared<std::packaged_task<RetType()>>(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<RetType> ret = task->get_future();
        {
            std::lock_guard<std::mutex> cv_mt(cv_mt_);
            tasks_.emplace([task] { (*task)(); });
        }
        cv_lock_.notify_one();
        return ret;
    }

 private:
    std::atomic_int thread_num_;
    std::queue<Task> tasks_;
    std::vector<std::thread> pool_;
    std::atomic_bool stop_;
    std::mutex cv_mt_;
    std::condition_variable cv_lock_;

    ThreadPool(unsigned int num = std::thread::hardware_concurrency()) {
        if (num <= 1) {
            thread_num_ = 2;
        } else {
            thread_num_ = num;
        }

        start();
    }

    void start() {
        for (int i = 0; i < thread_num_; i++) {
            pool_.emplace_back([this]() {
                Task task;
                {
                    std::unique_lock<std::mutex> cv_mt(cv_mt_);
                    this->cv_lock_.wait(cv_mt, [this] { return this->stop_.load() || !this->tasks_.empty(); });
                    if (this->tasks_.empty()) { return; }
                    task = std::move(this->tasks_.front());
                    this->tasks_.pop();
                }
                this->thread_num_--;
                task();
                this->thread_num_++;
            });
        }
    }

    void stop() {
        stop_.store(true);
        cv_lock_.notify_all();
        for (auto& td : pool_) {
            if (td.joinable()) {
                std::cout << "join thread " << td.get_id() << std::endl;
                td.join();
            }
        }
    }
};

template <typename T>
std::list<T> pool_thread_quick_sort(std::list<T> input) {
    if (input.empty()) { return input; }
    std::list<T> result;
    result.splice(result.begin(), input, input.begin());
    T const& partition_val = *result.begin();
    typename std::list<T>::iterator divide_point =
            std::partition(input.begin(), input.end(), [&](T const& val) { return val < partition_val; });
    std::list<T> new_lower_chunk;
    new_lower_chunk.splice(new_lower_chunk.end(), input, input.begin(), divide_point);
    std::future<std::list<T>> new_lower = ThreadPool::instance().commit(pool_thread_quick_sort<T>, new_lower_chunk);
    std::list<T> new_higher(pool_thread_quick_sort(input));
    result.splice(result.end(), new_higher);
    result.splice(result.begin(), new_lower.get());
    return result;
}

void TestThreadPoolSort() {
    std::list<int> nlist = {6, 1, 0, 5, 2, 9, 11};
    auto sortlist = pool_thread_quick_sort<int>(nlist);
    for (auto& value : sortlist) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}

int main() {
    TestThreadPoolSort();
}
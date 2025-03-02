


#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <ostream>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

using Task = std::packaged_task<void()>;

class MyThreadPool {


private:
    MyThreadPool(const MyThreadPool &other) = delete;
    MyThreadPool &operator=(const MyThreadPool &other) = delete;

    std::atomic<bool> stop_;
    std::queue<Task> task_;
    std::vector<std::thread> pool_;
    std::atomic_int thread_num_;
    std::mutex mtx_;
    std::condition_variable cv_;

public:
    MyThreadPool(int num = std::thread::hardware_concurrency()) {
        if (num == 1) {
            thread_num_ = 2;
        } else {
            thread_num_ = num;
        }

        start();
    }

    void start() {
        for(int i=0;i<thread_num_;++i){
            pool_.emplace_back(
                [this](){
                    Task task;
                    {
                        std::unique_lock<std::mutex> lock(mtx_);
                        cv_.wait(lock,[this]{return stop_.load()||!task_.empty();});
                        if(task_.empty()){return ;}
                        task=std::move(task_.front());
                        task_.pop();
                    }
                    thread_num_--;
                    task();
                    thread_num_++;
                }
            );
        }
    }

    void stop() {
        stop_.store(true);
        cv_.notify_all();

        for (auto &iter: pool_) {
            if (iter.joinable()) {
                std::cout << "desturction" << iter.get_id() << std::endl;
                iter.join();
            }
        }
    }

    template<class F,class ...Args>
    auto commit(F&&  f,Args&& ...args)->std::future<decltype(std::forward<F>(f)(std::forward<Args>(args)...))>{
        using RetType = decltype(std::forward<F>(f)(std::forward<Args>(args)...));

        if(stop_.load()){ return std::future<RetType>(RetType{});}

        auto task=std::make_shared<std::packaged_task<RetType>>(std::bind<RetType>(std::forward<F>(f), std::forward<Args>(args)...));
        auto ret = task->get_future();
        {
            std::unique_lock<std::mutex> mtx_;
            task_.emplace([task]{(*task)();});
        }
        cv_.notify_one();
        return ret;
    }
};

int main(){
    return 0;
}
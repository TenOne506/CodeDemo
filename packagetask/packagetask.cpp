#include <iostream>
#include <future>
#include <thread>

int task_function(int x) {
    return x * 2;
}

int main() {
    // 创建一个 packaged_task，包装一个函数
    std::packaged_task<int(int)> task(task_function);

    // 获取与任务关联的 future
    std::future<int> result = task.get_future();

    // 在另一个线程中执行任务
    std::thread t(std::move(task), 10); // 传递参数 10
    t.join();

    // 获取任务的结果
    std::cout << "Result: " << result.get() << std::endl;

    return 0;
}
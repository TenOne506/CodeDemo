#include <iostream>
#include <thread>
#include <chrono>

void background_task() {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Background task completed!" << std::endl;
}

int main() {
    std::thread t(background_task);
    t.detach(); // 分离线程，使其在后台运行

    std::cout << "Main thread continues to run..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3)); // 等待足够的时间以确保后台任务完成

    return 0;
}
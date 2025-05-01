#include "MyClass.h"
#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
template<typename T, size_t Cap>
class CircularQueLk : private std::allocator<T> {
public:
    CircularQueLk() : _max_size(Cap + 1), _data(std::allocator<T>::allocate(_max_size)), _head(0), _tail(0) {}
    CircularQueLk(const CircularQueLk &) = delete;
    CircularQueLk &operator=(const CircularQueLk &) volatile = delete;
    CircularQueLk &operator=(const CircularQueLk &) = delete;
    ~CircularQueLk() {
        //循环销毁
        std::lock_guard<std::mutex> lock(_mtx);
        //调用内部元素的析构函数
        while (_head != _tail) {
            //std::allocator<T>::destroy(_data + _head);
            std::__destroy_at(_data + _head);// 使用 std::destroy_at
            _head = (_head + 1) % _max_size;
        }
        //调用回收操作
        std::allocator<T>::deallocate(_data, _max_size);
    }
    //先实现一个可变参数列表版本的插入函数最为基准函数
    template<typename... Args>
    bool emplace(Args &&...args) {
        std::lock_guard<std::mutex> lock(_mtx);
        //判断队列是否满了
        if ((_tail + 1) % _max_size == _head) {
            std::cout << "circular que full ! " << std::endl;
            return false;
        }
        //在尾部位置构造一个T类型的对象，构造参数为args...
        std::__construct_at(_data + _tail, std::forward<Args>(args)...);
        //更新尾部元素位置
        _tail = (_tail + 1) % _max_size;
        return true;
    }
    //push 实现两个版本，一个接受左值引用，一个接受右值引用
    //接受左值引用版本
    bool push(const T &val) {
        std::cout << "called push const T& version" << std::endl;
        return emplace(val);
    }
    //接受右值引用版本，当然也可以接受左值引用，T&&为万能引用
    // 但是因为我们实现了const T&
    bool push(T &&val) {
        std::cout << "called push T&& version" << std::endl;
        return emplace(std::move(val));
    }
    //出队函数
    bool pop(T &val) {
        std::lock_guard<std::mutex> lock(_mtx);
        //判断头部和尾部指针是否重合，如果重合则队列为空
        if (_head == _tail) {
            std::cout << "circular que empty ! " << std::endl;
            return false;
        }
        //取出头部指针指向的数据
        val = std::move(_data[_head]);
        //更新头部指针
        _head = (_head + 1) % _max_size;
        return true;
    }

private:
    size_t _max_size;
    T *_data;
    std::mutex _mtx;
    size_t _head = 0;
    size_t _tail = 0;
};

void TestCircularQue() {
    //最大容量为10
    CircularQueLk<MyClass, 5> cq_lk;
    MyClass mc1(1);
    MyClass mc2(2);
    cq_lk.push(mc1);
    cq_lk.push(std::move(mc2));
    for (int i = 3; i <= 5; i++) {
        MyClass mc(i);
        auto res = cq_lk.push(mc);
        if (res == false) { break; }
    }
    cq_lk.push(mc2);
    for (int i = 0; i < 5; i++) {
        MyClass mc1;
        auto res = cq_lk.pop(mc1);
        if (!res) { break; }
        std::cout << "pop success, " << mc1 << std::endl;
    }
    auto res = cq_lk.pop(mc1);
}

template<typename T, size_t Cap>
class CircularQueSeq : private std::allocator<T> {
public:
    CircularQueSeq()
        : _max_size(Cap + 1), _data(std::allocator<T>::allocate(_max_size)), _atomic_using(false), _head(0), _tail(0) {}
    CircularQueSeq(const CircularQueSeq &) = delete;
    CircularQueSeq &operator=(const CircularQueSeq &) volatile = delete;
    CircularQueSeq &operator=(const CircularQueSeq &) = delete;
    ~CircularQueSeq() {
        //循环销毁
        bool use_expected = false;
        bool use_desired = true;
        do {
            use_expected = false;
            use_desired = true;
        } while (!_atomic_using.compare_exchange_strong(use_expected, use_desired));
        //调用内部元素的析构函数
        while (_head != _tail) {
            std::__destroy_at(_data + _head);
            _head = (_head + 1) % _max_size;
        }
        //调用回收操作
        std::allocator<T>::deallocate(_data, _max_size);
        do {
            use_expected = true;
            use_desired = false;
        } while (!_atomic_using.compare_exchange_strong(use_expected, use_desired));
    }
    //先实现一个可变参数列表版本的插入函数最为基准函数
    template<typename... Args>
    bool emplace(Args &&...args) {
        bool use_expected = false;
        bool use_desired = true;
        do {
            use_expected = false;
            use_desired = true;
        } while (!_atomic_using.compare_exchange_strong(use_expected, use_desired));
        //判断队列是否满了
        if ((_tail + 1) % _max_size == _head) {
            std::cout << "circular que full ! " << std::endl;
            do {
                use_expected = true;
                use_desired = false;
            } while (!_atomic_using.compare_exchange_strong(use_expected, use_desired));
            return false;
        }
        //在尾部位置构造一个T类型的对象，构造参数为args...
        std::__construct_at(_data + _tail, std::forward<Args>(args)...);
        //更新尾部元素位置
        _tail = (_tail + 1) % _max_size;
        do {
            use_expected = true;
            use_desired = false;
        } while (!_atomic_using.compare_exchange_strong(use_expected, use_desired));
        return true;
    }
    //push 实现两个版本，一个接受左值引用，一个接受右值引用
    //接受左值引用版本
    bool push(const T &val) {
        std::cout << "called push const T& version" << std::endl;
        return emplace(val);
    }
    //接受右值引用版本，当然也可以接受左值引用，T&&为万能引用
    // 但是因为我们实现了const T&
    bool push(T &&val) {
        std::cout << "called push T&& version" << std::endl;
        return emplace(std::move(val));
    }
    //出队函数
    bool pop(T &val) {
        bool use_expected = false;
        bool use_desired = true;
        do {
            use_desired = true;
            use_expected = false;
        } while (!_atomic_using.compare_exchange_strong(use_expected, use_desired));
        //判断头部和尾部指针是否重合，如果重合则队列为空
        if (_head == _tail) {
            std::cout << "circular que empty ! " << std::endl;
            do {
                use_expected = true;
                use_desired = false;
            } while (!_atomic_using.compare_exchange_strong(use_expected, use_desired));
            return false;
        }
        //取出头部指针指向的数据
        val = std::move(_data[_head]);
        //更新头部指针
        _head = (_head + 1) % _max_size;
        do {
            use_expected = true;
            use_desired = false;
        } while (!_atomic_using.compare_exchange_strong(use_expected, use_desired));
        return true;
    }

private:
    size_t _max_size;
    T *_data;
    std::atomic<bool> _atomic_using;
    size_t _head = 0;
    size_t _tail = 0;
};

void TestCircularQueSeq() {
    CircularQueSeq<MyClass, 3> cq_seq;
    for (int i = 0; i < 4; i++) {
        MyClass mc1(i);
        auto res = cq_seq.push(mc1);
        if (!res) { break; }
    }
    for (int i = 0; i < 4; i++) {
        MyClass mc1;
        auto res = cq_seq.pop(mc1);
        if (!res) { break; }
        std::cout << "pop success, " << mc1 << std::endl;
    }
    for (int i = 0; i < 4; i++) {
        MyClass mc1(i);
        auto res = cq_seq.push(mc1);
        if (!res) { break; }
    }
    for (int i = 0; i < 4; i++) {
        MyClass mc1;
        auto res = cq_seq.pop(mc1);
        if (!res) { break; }
        std::cout << "pop success, " << mc1 << std::endl;
    }
}

template<typename T, size_t Cap>
class CircularQueLight : private std::allocator<T> {
public:
    CircularQueLight() : _max_size(Cap + 1), _data(std::allocator<T>::allocate(_max_size)), _head(0), _tail(0) {}
    CircularQueLight(const CircularQueLight &) = delete;
    CircularQueLight &operator=(const CircularQueLight &) volatile = delete;
    CircularQueLight &operator=(const CircularQueLight &) = delete;
    bool pop(T &val);
    bool push(T &val);

private:
    size_t _max_size;
    T *_data;
    std::atomic<size_t> _head;
    std::atomic<size_t> _tail;
    std::atomic<size_t> _tail_update;
};

template<typename T, size_t Cap>
bool CircularQueLight<T, Cap>::pop(T &val) {
    // size_t h;
    // do {
    //     h = _head.load();//1 处
    //     //判断头部和尾部指针是否重合，如果重合则队列为空
    //     if (h == _tail.load()) { return false; }
    //     val = _data[h];// 2处
    // } while (!_head.compare_exchange_strong(h,
    //                                         (h + 1) % _max_size));//3 处
    size_t h;
    do {
        h = _head.load();//1 处
        //判断头部和尾部指针是否重合，如果重合则队列为空
        if (h == _tail.load()) { return false; }
        //判断如果此时要读取的数据和tail_update是否一致，如果一致说明尾部数据未更新完
        if (h == _tail_update.load()) { return false; }
        val = _data[h];// 2处
    } while (!_head.compare_exchange_strong(h,
                                            (h + 1) % _max_size));//3 处
    return true;
}
template<typename T, size_t Cap>
bool CircularQueLight<T, Cap>::push(T &val) {
    // size_t t;
    // do {
    //     t = _tail.load();//1
    //     //判断队列是否满
    //     if ((t + 1) % _max_size == _head.load()) { return false; }
    //     _data[t] = val;//2
    // } while (!_tail.compare_exchange_strong(t,
    //                                         (t + 1) % _max_size));//3
    /////
    // size_t t;
    // do {
    //     t = _tail.load();//1
    //     //判断队列是否满
    //     if ((t + 1) % _max_size == _head.load()) { return false; }
    // } while (!_tail.compare_exchange_strong(t,
    //                                         (t + 1) % _max_size));//3
    // _data[t] = val;//2
    size_t t;
    do {
        t = _tail.load();//1
        //判断队列是否满
        if ((t + 1) % _max_size == _head.load()) { return false; }
    } while (!_tail.compare_exchange_strong(t,
                                            (t + 1) % _max_size));//3
    _data[t] = val;//2
    size_t tailup;
    do { tailup = t; } while (_tail_update.compare_exchange_strong(tailup, (tailup + 1) % _max_size));

    return true;
}
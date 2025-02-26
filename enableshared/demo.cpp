#include <memory>
template<typename T>
class enable_shared_from_this {
private:
    mutable std::weak_ptr<T> weak_this; // 弱引用

protected:
    // 默认构造函数
    enable_shared_from_this() noexcept {}

    // 拷贝构造函数
    enable_shared_from_this(const enable_shared_from_this&) noexcept {}

    // 赋值运算符
    enable_shared_from_this& operator=(const enable_shared_from_this&) noexcept {
        return *this;
    }

public:
    // 获取当前对象的 shared_ptr
    std::shared_ptr<T> shared_from_this() {
        return std::shared_ptr<T>(weak_this); // 通过 weak_ptr 构造 shared_ptr
    }

    std::shared_ptr<const T> shared_from_this() const {
        return std::shared_ptr<const T>(weak_this);
    }

    // 供 std::shared_ptr 调用的内部接口
    void _internal_accept_owner(std::shared_ptr<T>* ptr) const {
        if (weak_this.expired()) {
            weak_this = std::shared_ptr<T>(*ptr);
        }
    }
};
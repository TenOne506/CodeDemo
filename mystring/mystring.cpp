#include <iostream>
#include <cstring>

class MyString {
private:
    char *str{};
    int len;

public:
    // 构造函数
    MyString(const char *s = nullptr, int l = 0) : str(nullptr), len(0) {
        if (s && l > 0) {
            len = l;
            str = new char[len + 1]; // 分配内存，+1 用于存储 '\0'
            std::memcpy(str, s, len);
            str[len] = '\0'; // 确保字符串以 '\0' 结尾
        }
    }

    // 析构函数
    ~MyString() {
        delete[] str; // 释放动态分配的内存
    }

    // 拷贝构造函数
    MyString(const MyString &other) : str(nullptr), len(0) {
        if (other.str) {
            len = other.len;
            str = new char[len + 1];
            std::memcpy(str, other.str, len + 1);
        }
    }

    // 移动构造函数
    MyString(MyString &&other) noexcept : str(other.str), len(other.len) {
        other.str = nullptr; // 将原对象的指针置空，避免双重释放
        other.len = 0;
    }

    // 拷贝赋值运算符
    MyString &operator=(const MyString &other) {
        if (this != &other) { // 避免自赋值
            delete[] str; // 释放当前对象的资源
            str = nullptr;
            len = 0;

            if (other.str) {
                len = other.len;
                str = new char[len + 1];
                std::memcpy(str, other.str, len + 1);
            }
        }
        return *this;
    }

    // 移动赋值运算符
    MyString &operator=(MyString &&other) noexcept {
        if (this != &other) { // 避免自赋值
            delete[] str; // 释放当前对象的资源
            str = other.str; // 接管资源
            len = other.len;

            other.str = nullptr; // 将原对象的指针置空
            other.len = 0;
        }
        return *this;
    }

    // 获取字符串长度
    int length() const {
        return len;
    }

    // 打印字符串
    void print() const {
        if (str) {
            std::cout << str;
        }
    }
};

int main() {
    // 测试构造函数
    MyString s1("Hello", 5);
    s1.print(); // 输出: Hello
    std::cout << ", Length: " << s1.length() << std::endl; // 输出: 5

    // 测试拷贝构造函数
    MyString s2 = s1;
    s2.print(); // 输出: Hello
    std::cout << ", Length: " << s2.length() << std::endl; // 输出: 5

    // 测试移动构造函数
    MyString s3 = std::move(s1);
    s3.print(); // 输出: Hello
    std::cout << ", Length: " << s3.length() << std::endl; // 输出: 5
    s1.print(); // 输出: (空)
    std::cout << ", Length: " << s1.length() << std::endl; // 输出: 0

    // 测试拷贝赋值运算符
    MyString s4;
    s4 = s2;
    s4.print(); // 输出: Hello
    std::cout << ", Length: " << s4.length() << std::endl; // 输出: 5

    // 测试移动赋值运算符
    MyString s5;
    s5 = std::move(s2);
    s5.print(); // 输出: Hello
    std::cout << ", Length: " << s5.length() << std::endl; // 输出: 5
    s2.print(); // 输出: (空)
    std::cout << ", Length: " << s2.length() << std::endl; // 输出: 0

    return 0;
}
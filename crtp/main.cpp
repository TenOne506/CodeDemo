#include <iostream>

// 基类使用 CRTP
template <typename Derived>
class Shape {
public:
    // 通过静态多态调用派生类的方法
    void draw() {
        static_cast<Derived*>(this)->drawImpl();
    }
};

// 派生类1
class Circle : public Shape<Circle> {
public:
    void drawImpl() {
        std::cout << "Drawing Circle" << std::endl;
    }
};

// 派生类2
class Rectangle : public Shape<Rectangle> {
public:
    void drawImpl() {
        std::cout << "Drawing Rectangle" << std::endl;
    }
};

int main() {
    Circle circle;
    Rectangle rectangle;
    
    circle.draw();      // Drawing Circle
    rectangle.draw();   // Drawing Rectangle
    
    return 0;
}

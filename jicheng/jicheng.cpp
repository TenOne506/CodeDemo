#include <iostream>
#include <ostream>
class A {
public:
    int x;
};

class B : public A {};

class C : public A {};

class D : public B, public C {};

struct  E{
 char A;
 char B;
 int C;
};
struct F{
 char A;
 int C;
 char B;
};
int main(){
    D d;
    E e;
    F f;
    std::cout<<sizeof(d)<<std::endl;
    std::cout<<sizeof(e)<<std::endl;
    std::cout<<sizeof(f)<<std::endl;
}
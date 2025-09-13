#include <stdio.h>

// 模拟 f(x)，打印调用信息并返回 x*2
int f(int x) {
    printf("f(%d) called\n", x);
    return x * 2;
}

// 简单的 fun，打印参数和返回和
int fun(int a, int b, int c) {
    printf("fun(%d, %d, %d) called\n", a, b, c);
    return a + b + c;
}

int main() {
    printf("Calling fun(f(1), f(2), f(3))...\n");
    int result = fun(f(1), f(2), f(3));
    printf("Result: %d\n", result);
    return 0;
}
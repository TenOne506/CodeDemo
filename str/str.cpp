#include <iostream>
#include <string>
using namespace std;

long long calculateCombinations(int n, int k) {
    if (k > n - k) k = n - k;  // 优化，取较小的k减少计算次数
    long long result = 1;
    for (int i = 1; i <= k; ++i) {
        result = result * (n - k + i) / i;
    }
    return result;
}

int main() {
    string s;
    cin >> s;
    int count0 = 0;
    for (char c : s) {
        if (c == '0') ++count0;
    }
    int count1 = s.size() - count0;

    if (count0 == 0 || count1 == 0) {
        cout << 1 << endl;
        return 0;
    }

    int n = count0 * count1;
    if (count0 != 1 || count1 != 1) { n++; }
    cout << n << endl;

    return 0;
}
#include <ios>
#include <iostream>
#include <vector>
using namespace std;

void printzigarray(int n) {
    vector<vector<int>> res(n, vector<int>(n, 0));

    int val = 1;
    int x = 0;
    int y = 0;
    int s = 0;     // 对角线和
    int flag = 1;  // 控制方向

    res[0][0] = val;
    val++;

    while (val <= n * n) {
        s++;
        if (s > (n - 1)) {  // 下半部分
            // int temp = s % (n - 1);
            int temp = s - (n - 1);
            if (flag % 2 == 1) {
                x = temp;
                y = s - temp;
                res[x][y] = val;
                val++;
                while (y > temp) {
                    x++;
                    y--;
                    res[x][y] = val;
                    val++;
                }
            } else {
                y = temp;
                x = s - y;
                res[x][y] = val;
                val++;
                while (x > temp) {
                    y++;
                    x--;
                    res[x][y] = val;
                    val++;
                }
            }
        } else {  // 上半部分
            if (flag % 2 == 1) {
                x = 0;
                y = s;
                res[x][y] = val;
                val++;
                while (y > 0) {
                    x++;
                    y--;
                    res[x][y] = val;
                    val++;
                }
            } else {
                y = 0;
                x = s;
                res[x][y] = val;
                val++;
                while (x > 0) {
                    y++;
                    x--;
                    res[x][y] = val;
                    val++;
                }
            }
        }
        flag++;
    }

    // 输出矩阵
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << res[i][j] << " ";
        }
        cout << endl;
    }
}

// 计算 (i, j) 位置的值
int getSpiralValue(int i, int j, int n) {
    // 计算当前层的最小值
    int layer = min(min(i, n - 1 - i), min(j, n - 1 - j));
    // 计算当前层的起始值
    int startValue = 4 * layer * (n - layer);
    // 根据位置确定具体值
    if (i == layer) {  // 顶部行
        return startValue + (j - layer) + 1;
    } else if (j == n - 1 - layer) {  // 右侧列
        return startValue + (n - 2 * layer - 1) + (i - layer) + 1;
    } else if (i == n - 1 - layer) {  // 底部行
        return startValue + 2 * (n - 2 * layer - 1) + (n - 1 - layer - j) + 1;
    } else {  // 左侧列
        return startValue + 3 * (n - 2 * layer - 1) + (n - 1 - layer - i) + 1;
    }
}

void printSpiralMatrix(int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << getSpiralValue(i, j, n) << "\t";
        }
        cout << endl;
    }
}

int main() {
    int n;
    cout << "Enter the size of the matrix: ";
    cin >> n;
    printzigarray(n);
    return 0;
}

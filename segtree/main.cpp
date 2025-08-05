#include <iostream>
#include "segtree.h"
using namespace std;

int main() {
    SegmentTree t(8, 0LL);  // 如果这里写 0LL，那么 SegmentTree 存储的就是 long long 数据
    t.update(0, 1LL << 60);
    cout << t.query(0, 7) << endl;

    vector<int> nums = {3, 1, 4, 1, 5, 9, 2, 6};
    // 注意：如果要让 SegmentTree 存储 long long 数据，需要传入 vector<long long>
    SegmentTree t2(nums);  // 这里 SegmentTree 存储的是 int 数据
    cout << t2.query(0, 7) << endl;
    return 0;
}
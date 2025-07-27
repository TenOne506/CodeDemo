#include <iostream>
#include <vector>

using namespace std;

int findMin(vector<int> &nums) {
  int left = 0;
  int right = nums.size();
  if (nums[right - 1] > nums[0]) {
    // 如果没有旋转，就直接返回最小的值
    return nums[0];
  }
  while (left < right) {
    int mid = left + (right - left) / 2;

    if (nums[mid] > nums[0]) {
      // 最小值在右半部分
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  return nums[left];
}

int main() {
  vector<int> nums1 = {3, 4, 5, 1, 2};
  cout << "Min in [3,4,5,1,2]: " << findMin(nums1) << endl;

  vector<int> nums2 = {4, 5, 6, 7, 0, 1, 2};
  cout << "Min in [4,5,6,7,0,1,2]: " << findMin(nums2) << endl;

  vector<int> nums3 = {11, 13, 15, 17};
  cout << "Min in [11,13,15,17]: " << findMin(nums3) << endl;

  return 0;
}
#include <iostream>
#include <utility>
#include <vector>

int partrtion(std::vector<int> &nums, int left, int right) {
  int pivot = nums[right];
  int i = left - 1;
  for (int j = left; j < right; ++j) {
    if (nums[j] < pivot) {
      i++;
      std::swap(nums[i], nums[j]);
    }
  }
  std::swap(nums[right], nums[i + 1]);
  return i + 1;
}

void quicksort(std::vector<int> &nums, int left, int right) {
  if (left < right) {
    int pivot = partrtion(nums, 0, right);
    quicksort(nums, left, pivot - 1);
    quicksort(nums, pivot + 1, right);
  }
}
int quickfind(std::vector<int> &nums, int left, int right, int k) {
  if (left == right) { return nums[left]; }
  if (left < right) {
    int pivotindex = partrtion(nums, left, right);
    if (pivotindex == k) {
      return nums[k];
    } else if (pivotindex > k) {
      return quickfind(nums, left, pivotindex, k);
    } else {
      return quickfind(nums, pivotindex, right, k);
    }
  }
  return -1;
}
// 辅助函数，用于打印数组
void printVector(const std::vector<int> &nums) {
  for (int num: nums) { std::cout << num << " "; }
  std::cout << std::endl;
}

// 辅助函数，用于检查数组是否有序
bool isSorted(const std::vector<int> &nums) {
  for (size_t i = 1; i < nums.size(); ++i) {
    if (nums[i] < nums[i - 1]) { return false; }
  }
  return true;
}

int main() {
  // 测试用例 1: 基本排序
  std::vector<int> nums1 = {9, 5, 7, 2, 4};
  std::cout << "Before sorting: ";
  printVector(nums1);
  quicksort(nums1, 0, nums1.size() - 1);
  std::cout << "After sorting: ";
  printVector(nums1);
  if (isSorted(nums1)) {
    std::cout << "Test case 1 passed." << std::endl;
  } else {
    std::cout << "Test case 1 failed." << std::endl;
  }

  // 测试用例 2: 空数组
  std::vector<int> nums2;
  std::cout << "Before sorting: ";
  printVector(nums2);
  quicksort(nums2, 0, nums2.size() - 1);
  std::cout << "After sorting: ";
  printVector(nums2);
  if (isSorted(nums2)) {
    std::cout << "Test case 2 passed." << std::endl;
  } else {
    std::cout << "Test case 2 failed." << std::endl;
  }

  // 测试用例 3: 单元素数组
  std::vector<int> nums3 = {1};
  std::cout << "Before sorting: ";
  printVector(nums3);
  quicksort(nums3, 0, nums3.size() - 1);
  std::cout << "After sorting: ";
  printVector(nums3);
  if (isSorted(nums3)) {
    std::cout << "Test case 3 passed." << std::endl;
  } else {
    std::cout << "Test case 3 failed." << std::endl;
  }

  // 测试用例 4: 已排序数组
  std::vector<int> nums4 = {1, 2, 3, 4, 5};
  std::cout << "Before sorting: ";
  printVector(nums4);
  quicksort(nums4, 0, nums4.size() - 1);
  std::cout << "After sorting: ";
  printVector(nums4);
  if (isSorted(nums4)) {
    std::cout << "Test case 4 passed." << std::endl;
  } else {
    std::cout << "Test case 4 failed." << std::endl;
  }

  // 测试用例 5: 逆序数组
  std::vector<int> nums5 = {5, 4, 3, 2, 1};
  std::cout << "Before sorting: ";
  printVector(nums5);
  quicksort(nums5, 0, nums5.size() - 1);
  std::cout << "After sorting: ";
  printVector(nums5);
  if (isSorted(nums5)) {
    std::cout << "Test case 5 passed." << std::endl;
  } else {
    std::cout << "Test case 5 failed." << std::endl;
  }

  return 0;
}
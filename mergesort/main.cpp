
#include <iostream>
#include <vector>
// 合并两个有序子数组
void merge(std::vector<int> &arr, int left, int mid, int right) {
  int n1 = mid - left + 1;// 左子数组的大小
  int n2 = right - mid;   // 右子数组的大小

  // 创建临时数组
  std::vector<int> L(n1), R(n2);

  // 拷贝数据到临时数组
  for (int i = 0; i < n1; i++) L[i] = arr[left + i];
  for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];

  // 合并两个有序数组
  int i = 0, j = 0, k = left;
  while (i < n1 && j < n2) {
    if (L[i] <= R[j]) {
      arr[k] = L[i];
      i++;
    } else {
      arr[k] = R[j];
      j++;
    }
    k++;
  }

  // 拷贝剩余元素（如果有）
  while (i < n1) {
    arr[k] = L[i];
    i++;
    k++;
  }
  while (j < n2) {
    arr[k] = R[j];
    j++;
    k++;
  }
}

// 归并排序递归函数
void mergeSort(std::vector<int> &arr, int left, int right) {
  if (left < right) {
    int mid = left + (right - left) / 2;// 计算中间位置

    // 递归排序左子数组和右子数组
    mergeSort(arr, left, mid);
    mergeSort(arr, mid + 1, right);

    // 合并两个有序子数组
    merge(arr, left, mid, right);
  }
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
  mergeSort(nums1, 0, nums1.size() - 1);
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
  mergeSort(nums2, 0, nums2.size() - 1);
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
  mergeSort(nums3, 0, nums3.size() - 1);
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
  mergeSort(nums4, 0, nums4.size() - 1);
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
  mergeSort(nums5, 0, nums5.size() - 1);
  std::cout << "After sorting: ";
  printVector(nums5);
  if (isSorted(nums5)) {
    std::cout << "Test case 5 passed." << std::endl;
  } else {
    std::cout << "Test case 5 failed." << std::endl;
  }

  return 0;
}

#include <iostream>
#include <stack>
#include <vector>
#include "tree.h"

std::vector<int> preorder(TreeNode* root) {
    std::stack<TreeNode*> st;
    std::vector<int> ans;

    TreeNode* cur = root;
    st.push(cur);
    while (!st.empty()) {
        cur = st.top();
        ans.push_back(cur->val);
        st.pop();
        if (cur->right != nullptr) { st.push(cur->right); }
        if (cur->left != nullptr) { st.push(cur->left); }
    }
    return ans;
}

std::vector<int> midorder(TreeNode* root) {
    std::stack<TreeNode*> st;
    std::vector<int> ans;

    TreeNode* cur = root;
    while ((cur != nullptr) || !st.empty()) {
        while (cur != nullptr) {
            st.push(cur);
            cur = cur->left;
        }

        cur = st.top();
        st.pop();
        ans.push_back(cur->val);
        cur = cur->right;
    }
    return ans;
}

std::vector<int> suforder(TreeNode* root) {
    std::stack<TreeNode*> st;
    std::vector<int> ans;

    TreeNode* cur = root;
    TreeNode* pre = nullptr;

    while ((cur != nullptr) || !st.empty()) {
        if (cur != nullptr) {
            st.push(cur);
            cur = cur->left;
        } else {
            auto* temp = st.top();
            if ((temp->right != nullptr) && temp->right != pre) {
                cur = temp->right;
            } else {
                ans.push_back(temp->val);
                pre = temp;
                st.pop();
            }
        }
    }
    return ans;
}

// 新增主函数
int main() {
    // 构建示例树：
    //     1
    //    / \
    //   2   3
    //  / \
    // 4   5
    auto* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);
    // 执行前序遍历
    std::vector<int> result = preorder(root);

    // 输出结果
    std::cout << "Preorder traversal: ";
    for (int val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    result = midorder(root);
    std::cout << "Midorder traversal: ";
    for (int val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    result = suforder(root);
    std::cout << "Midorder traversal: ";
    for (int val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    // 清理内存
    delete root->left->left;
    delete root->left->right;
    delete root->left;
    delete root->right;
    delete root;

    return 0;
}
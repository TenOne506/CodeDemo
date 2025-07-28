class TreeNode {
 public:
    int val;
    TreeNode* left;
    TreeNode* right;

    TreeNode(int v) : val(v) {}

    TreeNode(int v, TreeNode* l, TreeNode* r) : val(v), left(l), right(r) {}
};
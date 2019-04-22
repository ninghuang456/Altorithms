---
description: >-
  树，主要是二叉树，实战题大概分三类：Preorder/Inorder/Postorder Traverse相关的题，递归和迭代，略占20%；
  Levelorder Traverse相关的题（BFS），递归和迭代，略占10%； 剩下70%的题，都用递归或者分治法来做，Iterative的方法不强求。
---

# Tree

```text
二叉树的数据结构
class TreeNode {
int val;
// Left Child
TreeNode left;
// Right Child
TreeNode right;
}
```

```text
1. 求二叉树的最大深度
也就是根节点到最远叶子节点的距离
int maxDepth(TreeNode root) {
if(root == null) {
return 0;
}
int left = maxDepth(root.left);
int right = maxDepth(root.right);
return Math.max(left, right) + 1; //递归的层数+1
}
```

```text
2. 求二叉树的最小深度
也就是根节点到最近叶子节点的距离
int getMinDepth(TreeNode node) {
if (root == null) {
return 0;
}
return getMin(root);
}
int getMin(TreeNode root) {
if (root == null) {
return Integer.MAX_VALUE;
}
if (root.left == null && root.right == null) {//没有左右儿子了就在该结点处返回1
return 1;
}
return Math.min(getMin(root.left), getMin(root.right)) + 1;
}
```

```text
3. 求二叉树中结点的个数
int numOfTreeNodes(TreeNode root) {
if (root == null) {
return 0;
}
int left = numOfTreeNodes(root.left);
int right = numOfTreeNodes(root.right);
return left + right + 1;//算上root自己，+1
}
```

```text
4. 求二叉树中叶子结点的个数
int numOfChildNodes(Tree root) {
if (root == null) {
return 0;
}
if (root.left == null && root.right == null) {
return 1;
}
return numOfChildNodes(root.left) + numOfChildNodes(root.right);//这里就不用+1了
}
```


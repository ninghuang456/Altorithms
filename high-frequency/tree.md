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

```text
5. 求二叉树中第k层的结点的个数
k从1开始计数
int numOfLevelKNodes(TreeNode root, int k) {
if (root == null || k < 1) {
return 0;
}
if (k == 1) {
return 1;
}
//k-1，每次递归往k层递进一步
int left = numOfLevelKNodes(root.left, k - 1);
int right = numOfLevelKNodes(root.right, k - 1);
return left + right;
}
```

```text
6. 判断是否为平衡二叉树
递归检查左右子树的最大深度的差是否超过1
boolean isBalancedBinaryTree(TreeNode root) {
if (root == null) {
return true;
}
return maxDepth(root) != -1;
}
//返回最大深度
int maxDepth(TreeNode root) {
if (root == null) {
return 0;
}
int left = maxDepth(root.left);
int right = maxDepth(root.right);
//分别递归检查左右子树之间是否平衡和各自是否平衡
if (left == -1 || right == -1 || Math.abs(left - right) > 1) {
return -1;
}
//optional: 是平衡树的情况下返回最大深度
return Math.max(left, right) + 1;
}
```

```text
7. 判断是否是完全二叉树
boolean isCompleteBinaryTree(TreeNode root) {
if (root == null) {
return false; //注意null结点是返回false
}
boolean result = true;
Queue<TreeNode> queue = new LinkedList<TreeNode>();
queue.add(root);
boolean hasNoChild = false;// 判断左子树或右子树是否还有孩子
while (!queue.isEmpty()) {
TreeNode current = queue.remove();
if (hasNoChild) {//其中左子树和右子树的其中之一没有孩子了
if (current.left != null || current.right != null) {// 而左子树和右子树的其中之一则还有孩子
result = false;//肯定不平衡
break;
}
} else {
if (current.left != null && current.right != null) {//左右孩子都还有，入队列继续检查
queue.add(current.left);
queue.add(current.right);
} else if (current.left != null && current.right != null) {//存在左孩子，没有右孩子，可能是可能不是
queue.add(current.left);
hasNoChild = true;//只有左孩子是非满结点
} else if (current.left == null && current.right != null) {//没有左孩子，存在右孩子，根据完全二叉树的
定义，肯定不是
result = false;
break;
} else { // 左右孩子都存在，并为非满的状态，需要看后续
hasNoChild = true;
}
}
}
return result;
}
```

```text
8. 判断两个二叉树是否相同
boolean isSameTree(TreeNode node1, TreeNode node2) {
//递归的终止条件1
if (node1 == null && node2 == null) {
return true;
} else if (node1 == null || node2 == null) {
return false;
}
//递归的终止条件2
if (node1.val != node2.val) {
return false;
}
boolean left = isSameTree(node1.left, node2.left);
boolean right = isSameTree(node2.right, node2.right);
return left && right;
}
```

```text
9. 判读两个二叉树是否互为镜像
boolean isSymmetryTree(TreeNode node1, TreeNode node2) {
if (node1 == null && node2 == null) {
return true;
} else if (node1 == null || node2 == null) {
return false;
}
if (node1.val != node2.val) {
return false;
}
boolean left = isSymmertyTree(node1.left, node2.right);
boolean right = isSymmetryTree(node1.right, node2.left);
return left && right;
}
```

```text
10. 翻转二叉树（镜像二叉树）
TreeNode reverseTree(TreeNode root) {
if (root == null) {
return root;
}
//新建两个变量来保存左右子树
TreeNode left = reverseTree(root.left);
TreeNode right = reverseTree(root.right);
//赋值
root.left = right;
root.right = left;
return root;
}
```

```text
11. 求两个二叉树的最低公共祖先
二叉树两个节点的LCA的查找可以使用自顶向下，自底向上和寻找公共路径的方法
如果是BST，那么可以直接和root的val比较，方便很多
自顶向下，这个办法会重复遍历结点（查看结点在哪里）
TreeNode lowestCommonAncestor(TreeNode root, TreeNode node1, TreeNode node2) {
if (root.left, node1) {//node1在左子树的情况下
if (root.right, node2) {//node1在左子树的情况下，node2在右子树
return root;
} else {//node1在左子树的情况下，node2也在左子树，继续递归
lowestCommonAncestor(root.left, node1,node2);
}
} else {//node1在右子树的情况下
if (root.left, node2) {//node1在右子树的情况下，node2在左子树
return root;
} else {//node1在右子树的情况下，node2也在右子树，继续递归
lowestCommonAncestor(root.right, node1, node2);
}
}
}
//查找结点是否在当前的二叉树中
boolean hasNode(TreeNode root, TreeNode node) {
if (root == null || node == null) {
return false;
}
if (root == node) {
return true;
}
return hasNode(root.left, node) || hasNode(root.right, node);
}
自底向上，一旦遇到结点等于p或者q，则将其向上传递给它的父结点。父结点会判断它的左右子树是否都包含其中一个结点，
如果是，则父结点一定是这两个节点p和q的LCA，传递父结点到root。如果不是，继续向上传递其中的包含结点p或者q的子结点，
或者NULL(如果子结点不包含任何一个)。该方法时间复杂度为O(N)。
TreeNode lowestCommonAncestor(TreeNode root, TreeNode node1, TreeNode node2) {
if (root == null) {
return null;
}
//递归时遇到两个nodes之一的停止条件
if (root == node1 || root == node2) {
return root;
}
TreeNode left = lowestCommonAncestor(root.left, node1, node2);
TreeNode right = lowestCommonAncestor(root.right, node1, node2);
if (left != null && right != null) {//node1和node2分别在左右子树
return root;
}
return left != null ? left : right;//node1和node2在都在左子树中，或者二者都不在左子树（在右子树）
}
公共路径法，依次得到从根结点到结点p和q的路径，找出它们路径中的最后一个公共结点即是它们的LCA。该方法时间复杂度为O(N)。剑指offer第
50题。
```

```text
12. 二叉树的前序遍历
迭代解法
List<TreeNode> preorderTraversal(TreeNode root) {
List<Integer> result = new ArrayList<>();
Stack<TreeNode> stack = new Stack<>();
TreeNode current = root;
while (current != null) {
result.add(current.val);
if (current.right != null) {//从上到下，依次放入右儿子
stack.push(current.right);
}
current = current.left;
if (current == null && !stack.isEmpty()) {//这时候左儿子已经放完了
current = stack.pop();
}
}
return result;
}
递归解法
List<TreeNode> preorderTraversal(TreeNode root) {
List<TreeNode> result = new ArrayList<>();
preorderTraversalHelper(root, result);
return result;
}
private void preorderTraversalHelper(TreeNode root, List<TreeNode> result) {
if (root == null) {
return;
}
result.add(root.val);
preorderTraversalHelper(root.left, result);
preorderTraversalHelper(root.right, result);
}
```

```text
13. 二叉树的中序遍历
迭代解法
List<Integer> inorderTraversal(TreeNode root) {
List<Integer> result = new ArrayList<>();
Stack<TreeNode> stack = new Stack<>();
TreeNode current = root;
while (current != null || !stack.isEmpty()) {
while (current != null) {
stack.add(current);
current = current.left;//先移动指针去左儿子
}
//左儿子没了开始从二叉树的最底层弹，同时考虑每个结点的右儿子
current = stack.pop();
result.add(current.val);//这时候可以add了
current = current.right;
}
return result;
}
递归解法
public List<Integer> inorderTraversal(TreeNode root) {
List<Integer> result = new ArrayList<>();
inorderTraversalHelper(root, result);
return result;
}
private void inorderTraversalHelper(TreeNode root, List<Integer> result) {
if (root == null) {
return;
}
inorderTraversalHelper(root.left, result);
result.add(root.val);
inorderTraversalHelper(root.right, result);
}
```

```text
14. 二叉树的后序遍历
迭代解法
List<Integer> postorderTraversal(TreeNode root) {
List<Integer> result = new ArrayList<>();
if (root == null) {
return result;
}
Stack<TreeNode> stack = new Stack<>();
stack.push(root);
while (!stack.isEmpty()) {
TreeNode current = stack.pop();
result.add(0, current.val);
if (current.left != null) {
stack.push(current.left);
}
if (current.right != null) {
stack.push(current.right);
}
}
return result;
}
递归解法
List<Integer> postorderTraversal(TreeNode root) {
List<Integer> result = new ArrayList<Integer>();
postorderTraversal (root, result);
return result;
}
private void postorderTraversal(TreeNode root, List<Integer> result) {
if (root == null) {
return;
}
postorderTraversal (root.left, result);
postorderTraversal (root.right, result);
result.add(root.val);
}
```

```text
18. 二叉树的层序遍历
迭代
List<List<Integer>> levelOrder(TreeNode root) {
List<List<Integer>> result = new ArrayList<List<Integer>>();
if (root == null) {
return result;
}
Queue<TreeNode> queue = new LinkedList<>();
queue.add(root);
while (!queue.isEmpty()) {
int size = queue.size();// 每一层的元素个数
List<Integer> level = new ArrayList();
while (size > 0) {//BFS
TreeNode node = queue.poll();
level.add(node.val);
if (node.left != null) {
queue.add(node.left);
}
if (node.right != null) {
queue.add(node.right);
}
size--;
}
result.add(level);
}
return result;
}
递归
List<List<Integer>> levelOrder(TreeNode root) {
List<List<Integer>> result = new ArrayList<List<Integer>>();
if (root == null) {
return result;
}
levelOrderHelper(result, root, 0);
return result;
}
private void levelOrderHelper(List<List<Integer>> result, TreeNode current, int level) {
if (current == null) {
return;
}
if (result.size() == level) {
result.add(new ArrayList<Integer>());
}
result.get(level).add(current.val);
levelOrderHelper(result, current.left, level + 1);
levelOrderHelper(result, current.right, level + 1);
}
```

```text
19. 在二叉树中插入结点
TreeNode insertNode(TreeNode root,TreeNode node){
if(root == node){
return node;
}
TreeNode tmp = new TreeNode();
temp = root;
TreeNode last = null;
while(temp!=null){
last = temp;
if(temp.val>node.val){
temp = temp.left;
}else{
temp = temp.right;
}
}
if(last!=null){
if(last.val>node.val){
last.left = node;
}else{
last.right = node;
}
}
return root;
}
```

```text
20. 输入一个二叉树和一个整数，打印出二叉树中节点值的和等于输入整数所有的路径
void findPath(TreeNode r,int i){
if(root == null){
return;
}
Stack<Integer> stack = new Stack<Integer>();
int currentSum = 0;
findPath(r, i, stack, currentSum);
}
void findPath(TreeNode r,int i,Stack<Integer> stack,int currentSum){
currentSum+=r.val;
stack.push(r.val);
if(r.left==null&&r.right==null){
if(currentSum==i){
for(int path:stack){
System.out.println(path);
}
}
}
if(r.left!=null){
findPath(r.left, i, stack, currentSum);
}
if(r.right!=null){
findPath(r.right, i, stack, currentSum);
}
stack.pop();
}
```

```text

```


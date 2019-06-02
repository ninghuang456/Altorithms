# 21~30 classic

```java
21. 翻转二叉树(镜像二叉树)
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

22. Binary Tree Level Order Traversal
public class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        List<List<Integer>> wrapList = new LinkedList<List<Integer>>();
        
        if(root == null) return wrapList;
        
        queue.offer(root);
        while(!queue.isEmpty()){
            int levelNum = queue.size();
            List<Integer> subList = new LinkedList<Integer>();
            for(int i=0; i<levelNum; i++) {
                if(queue.peek().left != null) queue.offer(queue.peek().left);
                if(queue.peek().right != null) queue.offer(queue.peek().right);
                subList.add(queue.poll().val);
            }
            wrapList.add(subList);
        }
        return wrapList;
    }
}

class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        levelOrderHelper(root,result,0);
        return result;
        
    }
    public void levelOrderHelper(TreeNode root,List<List<Integer>> result, int level){
        if (root == null) return ;
        if (level == result.size()){
            result.add(new LinkedList<Integer>());
        }
       result.get(level).add(root.val);
        levelOrderHelper(root.left, result, level+1);
        levelOrderHelper(root.right, result, level+1); 
    }
}

23. Symmetric Tree
boolean isSymmetryTree(TreeNode node1, TreeNode node2) {
   if (node1 == null && node2 == null) {
    return true;
}  else if (node1 == null || node2 == null) {
   return false;
}
   if (node1.val != node2.val) {
    return false;
}
    boolean left = isSymmertyTree(node1.left, node2.right);
    boolean right = isSymmetryTree(node1.right, node2.left);
    return left && right;
}

24. 求二叉树中第k层的结点的个数
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

25. Maximum Depth of Binary Tree
int maxDepth(TreeNode root) {
    if(root == null) {
     return 0;
}
     int left = maxDepth(root.left);
     int right = maxDepth(root.right);
     return Math.max(left, right) + 1; //递归的层数+1
}

26.Binary Tree Preorder Traversal
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> output = new ArrayList<Integer>();
        preorderrecur(root,output);
        return output;
    }
    private void preorderrecur(TreeNode root, List<Integer> output){
        if (root == null) return;
        output.add(root.val);
        preorderrecur(root.left,output);
        preorderrecur(root.right,output);
    }
}
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> output = new ArrayList<Integer>();
        preorderrecur(root,output);
        return output;
    }
    private void preorderrecur(TreeNode root, List<Integer> output){
        if (root == null) return;
        Stack<TreeNode> sk = new Stack<TreeNode>();
        TreeNode cur = root;
        while (cur != null) {
            output.add(cur.val);
            if (cur.right != null) {
                sk.push(cur.right);
            }
            cur = cur.left;
            if (cur == null && !sk.isEmpty()){
                cur = sk.pop();
            }
        } 
    }
}

27.Word Search
Given a 2D board and a word, find if the word exists in the grid.
The word can be constructed from letters of sequentially adjacent cell, 
where "adjacent" cells are those horizontally or vertically neighboring. 
The same letter cell may not be used more than once.
Example:
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false.
class Solution {
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0 || board[0].length == 0) {
            return false;
        }
        int m = board.length, n = board[0].length;
        boolean[][] visited = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                visited[i][j] = false;
            }
        }
        int index = 0;//字符串的索引
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (dfs(board, word, index, i, j, visited)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean dfs(char[][] board, String word, int index, int i, int j, boolean[][] visited) {
        if (index == word.length()) {//找完了
            return true;
        }
        int m = board.length, n = board[0].length;
        if (i < 0 || j < 0 || i >= m || j >= n
                || visited[i][j] //已被访问过
                || board[i][j] != word.charAt(index)) {//两个字符不相等
            return false;
        }
        visited[i][j] = true;//设定当前字符已被访问过
        boolean result = (dfs(board, word, index + 1, i - 1, j, visited)//左
                || dfs(board, word, index + 1, i + 1, j, visited)//右
                || dfs(board, word, index + 1, i, j - 1, visited)//上
                || dfs(board, word, index + 1, i, j + 1, visited));//下
        visited[i][j] = false;//让“当前的”位置归为初始值，为别的路径的查找准备
        return result;
    }
}
28.Permutations
Given a collection of distinct integers, return all possible permutations.
Example:
Input: [1,2,3]
Output:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
class Solution {
    public List<List<Integer>> permute(int[] nums) {      
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (nums == null || nums.length == 0) return result;       
        backtracking(result,nums, new ArrayList<Integer>());
        return result;
    }   
    private void backtracking(List<List<Integer>> result, int[] nums,List<Integer> oneline) {
        if (nums.length == oneline.size()){
            result.add(new ArrayList<>(oneline));
        } else {
            for (int i = 0; i< nums.length; i++) {
               if(!oneline.contains(nums[i])) {
                   oneline.add(nums[i]);
                   backtracking(result, nums, oneline);
                   oneline.remove(oneline.size() - 1);
               }
            }
        }  
    }
}

29.Coin Change
You are given coins of different denominations and a total amount of money amount. 
Write a function to compute the fewest number of coins that you need to make up that amount. 
If that amount of money cannot be made up by any combination of the coins, return -1.
Example 1:
Input: coins = [1, 2, 5], amount = 11
Output: 3 
Explanation: 11 = 5 + 5 + 1

class Solution {
    public int coinChange(int[] coins, int amount) {
         if(amount<1) return 0;
    return helper(coins, amount, new int[amount]);
}

private int helper(int[] coins, int rem, int[] count) { 
// rem: remaining coins after the last step; count[rem]: 
//minimum number of coins to sum up to rem
    if(rem<0) return -1; // not valid
    if(rem==0) return 0; // completed
    if(count[rem-1] != 0) return count[rem-1]; // already computed, so reuse
    int min = Integer.MAX_VALUE;
    for(int coin : coins) {
        int res = helper(coins, rem-coin, count);
        if(res>=0 && res < min)
            min = 1+res;
    }
    count[rem-1] = (min==Integer.MAX_VALUE) ? -1 : min;
    return count[rem-1];        
    }
}

30. Unique Paths
The robot can only move either down or right at any point in time. 
The robot is trying to reach the bottom-right corner of the grid 
How many possible unique paths are there?

class Solution {
    public int uniquePaths(int m, int n) {
        int[][] sum = new int[m][n];
        for (int i = 0; i < m; i++) {
            sum[i][0] = 1;
        }
        for (int i = 0; i < n; i++) {
            sum[0][i] = 1;
        }
        for (int i = 1; i < m; i ++) {
            for (int j = 1; j < n; j++) {
                sum [i][j] = sum[i][j-1] + sum [i - 1][j];
            }
        }
        return sum[m-1][n-1];
    }
}

```


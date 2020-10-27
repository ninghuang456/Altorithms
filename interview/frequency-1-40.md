# Frequency 1-40

## 766-Toeplitz Matrix

```java
A matrix is Toeplitz if every diagonal from top-left to bottom-right has the same element.

Now given an M x N matrix, return True if and only if the matrix is Toeplitz.
 

Example 1:

Input:
matrix = [
  [1,2,3,4],
  [5,1,2,3],
  [9,5,1,2]
]
Output: True
Explanation:
In the above grid, the diagonals are:
"[9]", "[5, 5]", "[1, 1, 1]", "[2, 2, 2]", "[3, 3]", "[4]".
In each diagonal all elements are the same, so the answer is True.

class Solution {
    public boolean isToeplitzMatrix(int[][] matrix) {
        int r = matrix.length;
        int c = matrix[0].length;
        for (int i = 1; i < r; i ++) {
            for (int j = 1; j < c; j ++) {
                if (matrix[i][j] != matrix[i - 1][j - 1]){
                    return false;
                }
            }
        }
        
        return true;
    }
}


```

## 317-Shortest Distance from All Buildings

```java
Input: [[1,0,2,0,1],[0,0,0,0,0],[0,0,1,0,0]]

1 - 0 - 2 - 0 - 1
|   |   |   |   |
0 - 0 - 0 - 0 - 0
|   |   |   |   |
0 - 0 - 1 - 0 - 0

Output: 7 

Explanation: Given three buildings at (0,0), (0,4), (2,2), and an obstacle at (0,2),
             the point (1,2) is an ideal empty land to build a house, as the total 
             travel distance of 3+3+1=7 is minimal. So return 7.
             
public class Solution {
public int shortestDistance(int[][] grid) {
    int row = grid.length;
    if (row == 0) {
        return -1;
    }
    int col = grid[0].length;
    int[][] record1 = new int[row][col]; // visited num
    int[][] record2 = new int[row][col]; // distance
    int num1 = 0;
    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            if (grid[r][c] == 1) {
                num1 ++;
                boolean[][] visited = new boolean[row][col];
                Queue<int[]> queue = new LinkedList<int[]>();
                queue.offer(new int[]{r, c});
                int dist = 0;
                while (!queue.isEmpty()) {
                    int size = queue.size();
                    for (int i = 0; i < size; i++) {
                        int[] node = queue.poll();
                        int x = node[0];
                        int y = node[1];
                        record2[x][y] += dist;
                        record1[x][y] ++;
                        if (x > 0 && grid[x - 1][y] == 0 && !visited[x - 1][y]) {
                            queue.offer(new int[]{x - 1, y});
                            visited[x - 1][y] = true;
                        }
                        if (x + 1 < row && grid[x + 1][y] == 0 && !visited[x + 1][y]) {
                            queue.offer(new int[]{x + 1, y});
                            visited[x + 1][y] = true;
                        }
                        if (y > 0 && grid[x][y - 1] == 0 && !visited[x][y - 1]) {
                            queue.offer(new int[]{x, y - 1});
                            visited[x][y - 1] = true;
                        }
                        if (y + 1 < col && grid[x][y + 1] == 0 && !visited[x][y + 1]) {
                            queue.offer(new int[]{x, y + 1});
                            visited[x][y + 1] = true;
                        }
                    }
                    dist ++;
                }
            }
        }
    }
    int result = Integer.MAX_VALUE;
    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            if (grid[r][c] == 0 && record1[r][c] == num1 && record2[r][c] < result) {
                result = record2[r][c];
            }
        }
    }
    return result == Integer.MAX_VALUE ? -1 : result;
}
}
```

## 76-Minimum Window Substring

```java
Given a string S and a string T, find the
 minimum window in S which will contain all the characters in T 
 in complexity O(n).
Input: S = "ADOBECODEBANC", T = "ABC"
Output: "BANC"

class Solution {
    public String minWindow(String s, String t) {
        if (s == null || t == null || s.length() < t.length()) {
            return "";
        }
        HashMap<Character, Integer> window = new HashMap<>(); 
        HashMap<Character, Integer> need = new HashMap<>();
        
        for (char t1 : t.toCharArray()) {
            need.put(t1, need.getOrDefault(t1,0) + 1);
        }
        int left = 0; int right = 0; int valid = 0;
        int start = 0; int len = Integer.MAX_VALUE; // use one minRight to record
        while (right < s.length()) { // 区间[left, right)是左闭右开的，所以初始情况下窗口没有包含任何元素：
            char s1 = s.charAt(right);
            right ++;
            if (need.containsKey(s1)) {
                window.put(s1,window.getOrDefault(s1,0) + 1);
                if (window.get(s1).equals(need.get(s1))) valid ++;
            }
       // 右指针移动当于在寻找一个「可行解」，然后移动左指针在优化这个「可行解」，最终找到最优解
            while (valid == need.size()) {
              if (right - left < len) {
                    start = left;
                    len = right - left;
                }
                char s2 = s.charAt(left);
                left ++;
                if (need.containsKey(s2)) { // every time need containsKey not contains
                   if (window.get(s2).equals(need.get(s2))) valid --;
                   window.put(s2,window.get(s2) - 1);
                }
              
            }
        }
        if (len == Integer.MAX_VALUE) return ""; // don't miss
        return s.substring(start, start + len);  
    }
}
```

## 896- Monotonic Array

```java
An array is monotonic if it is either monotone increasing or monotone decreasing.

An array A is monotone increasing if for all i <= j, A[i] <= A[j].  An array A is monotone decreasing if for all i <= j, A[i] >= A[j].

Return true if and only if the given array A is monotonic.
Input: [1,2,2,3]
Output: true

class Solution {
      public boolean isMonotonic(int[] A) {
        boolean inc = true, dec = true;
        for (int i = 1; i < A.length; ++i) {
            inc &= A[i - 1] <= A[i];
            dec &= A[i - 1] >= A[i];
        }
        return inc || dec;
    }
}
```

## 311-Sparse Matrix Multiplication

```java
Input:

A = [
  [ 1, 0, 0],
  [-1, 0, 3]
]

B = [
  [ 7, 0, 0 ],
  [ 0, 0, 0 ],
  [ 0, 0, 1 ]
]

Output:

     |  1 0 0 |   | 7 0 0 |   |  7 0 0 |
AB = | -1 0 3 | x | 0 0 0 | = | -7 0 3 |
                  | 0 0 1 |
                  
public class Solution {
    public int[][] multiply(int[][] A, int[][] B) {
        int m = A.length, n = A[0].length, nB = B[0].length;
        int[][] C = new int[m][nB];

        for(int i = 0; i < m; i++) {
            for(int k = 0; k < n; k++) {
                if (A[i][k] != 0){
                    for (int j = 0; j < nB; j++) {
                        if (B[k][j] != 0) C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }
        return C;   
    }
}
```

## 42-Trapping Rain Water

```java
class Solution {
    public int trap(int[] height) {
        if (height == null || height.length == 0) return 0;
        int left = 0; int right = height.length - 1;
        int leftMax = height[left];
        int rightMax = height[right];
        int sum = 0;
        while (left <= right){
           leftMax = Math.max(leftMax, height[left]);
           rightMax = Math.max(rightMax, height[right]);
           if (leftMax < rightMax){
               sum += leftMax - height[left];
               left ++;
           } else {
               sum += rightMax - height[right];
               right --;
           }
            
        }
        return sum;
    }
}
```

## 56-Merge Intervals

```java
class Solution {
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, (a,b) -> a[0] - b[0]); // -> not =>
        int k = 0;
        int i = 0;
        while (i < intervals.length) {
            int start = intervals[i][0];
            int end = intervals[i][1];
            while (i < intervals.length - 1 && end >= intervals[i + 1][0]){ // test overlap need to first check end with next start
                i ++;
                end = Math.max(intervals[i][1], end);
            }
            intervals[k][0] = start;
            intervals[k][1] = end;
            k ++;
            i ++; 
        }
        return Arrays.copyOf(intervals, k);
    }
}
```

## 1060-Missing Element in Sorted Array

```java
Input: A = [4,7,9,10], K = 1
Output: 5
Explanation: 
The first missing number is 5.

class Solution {
   public int missingElement(int[] nums, int k) {
        if(nums == null || nums.length == 0 ) return 0;
        int l = nums.length;
        int left = 0, right = nums.length - 1;
        int missing = nums[right] - nums[left] - (right - left);
        if(missing < k) return nums[right] + k - missing;
        while(left + 1 < right){
            int mid = left + (right - left) / 2;
            int missingLeft = nums[mid] - nums[left] - (mid - left);
            if(missingLeft >= k) right = mid;
            else{
                k -= missingLeft;
                left = mid;
            } 
        }
        return nums[left] + k;
    }
}
```

## 140-Word Break II

```java
Input:
s = "catsanddog"
wordDict = ["cat", "cats", "and", "sand", "dog"]
Output:
[
  "cats and dog",
  "cat sand dog"
]

class Solution {
   public List<String> wordBreak(String s, List<String> wordDict) {
    HashSet<String>  wordset = new HashSet<>(wordDict);
    return DFS(s, wordset, new HashMap<String, LinkedList<String>>());
}       

// DFS function returns an array including all substrings derived from s.
List<String> DFS(String s, Set<String> wordDict, HashMap<String, LinkedList<String>>map) {
    if (map.containsKey(s)) 
        return map.get(s);
        
    LinkedList<String>res = new LinkedList<String>();     
    if (s.length() == 0) {
        res.add("");
        return res;
    }               
    for (String word : wordDict) {
        if (s.startsWith(word)) {
            List<String>sublist = DFS(s.substring(word.length()), wordDict, map);
            for (String sub : sublist) 
                res.add(word + (sub.isEmpty() ? "" : " ") + sub);               
        }
    }       
    map.put(s, res);
    return res;
}
}
```

## 23-Merge k Sorted Lists

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        
       // if(lists == null || lists.length == 0) return null;
        PriorityQueue<ListNode> pq = new PriorityQueue<>((l1,l2)-> l1.val - l2.val);
        ListNode head = new ListNode(-1);
        ListNode cur = head;
        for(int i = 0; i < lists.length; i ++){
            if(lists[i] != null){
             pq.offer(lists[i]);   
            }
        }
        while(!pq.isEmpty()){
            ListNode temp = pq.poll();
            cur.next = temp;
            cur = cur.next;
            if(temp.next != null){
                pq.offer(temp.next);
            }
        }
        return head.next;
    }
}
```

## 670-Maximum Swap

```java
Input: 2736
Output: 7236
Explanation: Swap the number 2 and the number 7.

class Solution {
    public int maximumSwap(int num) {
        char[] digits = Integer.toString(num).toCharArray();
        
        int[] buckets = new int[10];
        for (int i = 0; i < digits.length; i++) {
            buckets[digits[i] - '0'] = i;
        }
        
        for (int i = 0; i < digits.length; i++) {
            for (int k = 9; k > digits[i] - '0'; k--) {
                if (buckets[k] > i) {
                    char tmp = digits[i];
                    digits[i] = digits[buckets[k]];
                    digits[buckets[k]] = tmp;
                    return Integer.valueOf(new String(digits));
                }
            }
        }
        
        return num;
    }
}
```

## 314-Binary Tree Vertical Order Traversal

```java
Examples 1:

Input: [3,9,20,null,null,15,7]

   3
  /\
 /  \
 9  20
    /\
   /  \
  15   7 

Output:

[
  [9],
  [3,15],
  [20],
  [7]
]

class Solution {
  public List<List<Integer>> verticalOrder(TreeNode root) {
    List<List<Integer>> result = new ArrayList<>();
    if (root == null) return result;
    Map<Integer, List<Integer>> map = new HashMap<>();
    int min = 0, max = 0;
    Queue<TreeNode> queue = new LinkedList<>();
    Queue<Integer> helper = new LinkedList<>();
    queue.offer(root);
    helper.offer(0);

    while (!queue.isEmpty()) {
        int size = queue.size();
        for (int i = 0; i < size; i++) {
            TreeNode cur = queue.poll();
            int pos = helper.poll();
            min = Math.min(min, pos);
            max = Math.max(max, pos);
            if (!map.containsKey(pos)) map.put(pos, new ArrayList<>());
            map.get(pos).add(cur.val);
            if (cur.left != null) {
                queue.offer(cur.left);
                helper.offer(pos - 1);
            }
            if (cur.right != null) {
                queue.offer(cur.right);
                helper.offer(pos + 1);   
            }
        }
    }

    for (int i = min; i <= max; i++) result.add(- min + i, map.get(i));
    
    return result;
}

}
```

## 1026-Maximum Difference Between Node and Ancestor

```java
class Solution {
    int res = 0;
    public int maxAncestorDiff(TreeNode root) {
        if (root == null) return 0;
        dfs(root, root.val, root.val);
        return res;
    }
    
    
   // 最大差值一定是ancestors里面的最大值或最小值跟当前值的差值的绝对值。
    //因此只保存最大和最新的ancestor值即可 
    private void dfs(TreeNode node, int min, int max) {
        if (node == null) return;
        min = Math.min(node.val, min);
        max = Math.max(node.val, max);
        res = Math.max(res, Math.max(Math.abs(max - node.val),Math.abs(min - node.val)));
        dfs(node.left, min, max);
        dfs(node.right, min, max);
    }
}
```

## 31-Next Permutation

```java
class Solution {
    public void nextPermutation(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int len = nums.length, p = len - 2, q = len - 1;
        
        // 1. 从后向前找到一个非增长的元素，等于的话也继续向前
        while (p >= 0 && nums[p] >= nums[p + 1]) {
            p--;
        }
        //全逆序的数组不会进入这个判断，全逆序p的位置为-1
        // 2. 从后向前找到第一个比p位置元素大的元素，注意这个数字肯定有，等于的话继续向前
        if (p >= 0) {
            while (nums[q] <= nums[p]) {
                q--;
            }
            swap(nums, p, q);
        }
        // 3. p位置后面的数组元素进行翻转
        reverse(nums, p + 1, len - 1);
    }
    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    private void reverse(int[] nums, int left, int right) {
        while (left < right) {
            swap(nums, left, right);
            left++;
            right--;
        }
    }
}
```

## 173- Binary Search Tree Iterator

```java

class BSTIterator {
    Stack<TreeNode> stack;
    TreeNode cur;
    
    public BSTIterator(TreeNode root) {
        stack = new Stack<>();
        cur = root;
    }
    
    public int next() {
        int val = 0;
        while (!stack.isEmpty() || cur != null) {
            while (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            TreeNode node = stack.pop();
            val = node.val;
            cur = node.right;
            break;
        }
        return val;
    }
    
    public boolean hasNext() {
        return !stack.isEmpty() || cur != null;
    }
}
```

## 124-Binary Tree Maximum Path Sum

```java
 a path is defined as any node sequence from some starting node 
 to any node in the tree along the parent-child connections. 
 The path must contain at least one node and does not need to go through the root.
 
 class Solution { // 拆成两个问题 经过改节点的最大距离 对比所有最大距离
   int max = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        if (root == null) {
            return 0;
        }
        dfs(root);
        return max;
    }
    public int dfs(TreeNode root) { // 后序遍历
        if (root == null) {
            return 0;
        }
        //计算左边分支最大值，左边分支如果为负数还不如不选择
        int leftMax = Math.max(0, dfs(root.left));
        //计算右边分支最大值，右边分支如果为负数还不如不选择
        int rightMax = Math.max(0, dfs(root.right));
        //left->root->right 作为路径与历史最大值做比较  // 更新遍历在当前节点时的最大路径和
        max = Math.max(max, root.val + leftMax + rightMax);
        //  // 选择以当前节点为根的含有最大值的路劲，左或右；返回给上一层递归的父节点作为路径
        return root.val + Math.max(leftMax, rightMax); // 不能左右同时返回
    }
    // https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/solution/shou-hui-tu-jie-hen-you-ya-de-yi-dao-dfsti-by-hyj8/
}
 
```

## 34-Find First and Last Position of Element in Sorted Array

```java
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]

class Solution {
   public int[] searchRange(int[] nums, int target) {
    int[] result = new int[2];
    result[0] = findFirst(nums, target);
    result[1] = findLast(nums, target);
    return result;
}

private int findFirst(int[] nums, int target){
    int idx = -1;
    int start = 0;
    int end = nums.length - 1;
    while(start <= end){
        int mid = (start + end) / 2;
        if(nums[mid] >= target){
            end = mid - 1;
        }else{
            start = mid + 1;
        }
        if(nums[mid] == target) idx = mid;
    }
    return idx;
}

private int findLast(int[] nums, int target){
    int idx = -1;
    int start = 0;
    int end = nums.length - 1;
    while(start <= end){
        int mid = (start + end) / 2;
        if(nums[mid] <= target){
            start = mid + 1;
        }else{
            end = mid - 1;
        }
        if(nums[mid] == target) idx = mid;
    }
    return idx;
}
}
```

## 426-Convert Binary Search Tree to Sorted Doubly Linked List

```java

class Solution {
    public Node treeToDoublyList(Node root) {
        if (root == null) return null;
        Stack<Node> stack = new Stack<>();
        Node cur = root;
        Node first = null; Node last = null;
        while (!stack.isEmpty() || cur != null) {
            while (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            Node node = stack.pop();
            if (first == null){
                first = node;
            }
            if (last != null) {
                last.right = node;
                node.left = last;
            }
            last = node;
            cur = node.right;
            
        }
        first.left = last;
        last.right = first;
        return first;
    }
}
```

## 

## 

## 

## 




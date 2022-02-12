# linkedin 25\~ 70

## 611 Valid Triangle Number

Given an integer array `nums`, return _the number of triplets chosen from the array that can make triangles if we take them as side lengths of a triangle_.

```
Input: nums = [2,2,3,4]
Output: 3
Explanation: Valid combinations are: 
2,3,4 (using the first 2)
2,3,4 (using the second 2)
2,2,3
// 不要与排列组合类题目混淆 固定三个 也可以是多指针
class Solution {
    public int triangleNumber(int[] nums) {
         int result = 0;
        if (nums.length < 3) return result;    
        Arrays.sort(nums);
        for (int i = 2; i < nums.length; i++) {
            int left = 0, right = i - 1;
            while (left < right) {
                if (nums[left] + nums[right] > nums[i]) {
                    result += (right - left);
                    right--;
                }
                else {
                    left++;
                }
            }
        }   
        return result;
    }
}

```

## 236 Lowest Common Ancestor of a Binary Tree

```

class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || p == root || q == root) return root;
        TreeNode left =  lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if(right == null) return left;
        // wrong answer if(left == right) return root;
        if(left == null) return right;
        return root;
        
    }
}
```

## 1650 Lowest Common Ancestor of a Binary Tree III

Given two nodes of a binary tree `p` and `q`, return _their lowest common ancestor (LCA)_.

Each node will have a reference to its parent node. The definition for `Node` is below:

```java
class Node {
    public int val;
    public Node left;
    public Node right;
    public Node parent;
}

class Solution {
    public Node lowestCommonAncestor(Node p, Node q) {
        if(p == null || q == null) return (p == null) ? q : p;
        HashSet<Node> pParents = new HashSet<>();
        while(p != null){
            pParents.add(p);
            p = p.parent;
        }
        while(!pParents.contains(q)){
            q = q.parent;
        } 
        return q;
    }
}
```

## 76 Minimum Window Substring

Given two strings `s` and `t` of lengths `m` and `n` respectively, return _the **minimum window substring** of_ `s` _such that every character in_ `t` _(**including duplicates**) is included in the window. If there is no such substring, return the empty string_ `""`_._

The testcases will be generated such that the answer is **unique**.

A **substring** is a contiguous sequence of characters within the string.

```
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B',
 and 'C' from string t.
 
 class Solution {
    public String minWindow(String s, String t) {
        if(s == null || t == null  || s.length() < t.length()) return "";
        HashMap<Character, Integer> window = new HashMap<>();
        HashMap<Character, Integer> need = new HashMap<>();
        for(Character tc : t.toCharArray()){
            need.put(tc, need.getOrDefault(tc, 0) + 1);
        }
        int left = 0; int right = 0; int valid = 0;
        int start = 0; int len = Integer.MAX_VALUE; 
        while(right < s.length()){
             char rc = s.charAt(right);
             right ++;
            
             if(need.containsKey(rc)){
                 window.put(rc, window.getOrDefault(rc, 0) + 1);
                 if(window.get(rc).equals(need.get(rc)) ){ 
                 // equlas not == 
                     valid ++;
                 }
             }
            while(valid == need.size()){
              if (right - left < len) {
                    start = left;
                    len = right - left;
                }
                char s2 = s.charAt(left);
                left ++;
                if (need.containsKey(s2)) {
                 // every time need containsKey not contains
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

## 973 K Closest Points to Origin

Given an array of `points` where `points[i] = [xi, yi]` represents a point on the **X-Y** plane and an integer `k`, return the `k` closest points to the origin `(0, 0)`.

The distance between two points on the **X-Y** plane is the Euclidean distance (i.e., `√(x1 - x2)2 + (y1 - y2)2`).

You may return the answer in **any order**. The answer is **guaranteed** to be **unique** (except for the order that it is in).

```

class Solution {
    public int[][] kClosest(int[][] points, int k) {
        PriorityQueue<int[]> pq = 
  new PriorityQueue<>((a, b) -> b[0]*b[0] + b[1]*b[1] - a[0]*a[0] - a[1]*a[1]);
        for (int i = 0; i < points.length; i ++) {
            pq.offer(points[i]);
            if(pq.size() > k) {
                pq.poll();
            }
        }
        int[][] res = new int[k][2];
        for(int i = k - 1; i >= 0; i --) {
            res[i] = pq.poll();
        }
        return res;  
    }
}
```

## 323 Number of Connected Components in an Undirected Graph

You have a graph of `n` nodes. You are given an integer `n` and an array `edges` where `edges[i] = [ai, bi]` indicates that there is an edge between `ai` and `bi` in the graph.

Return _the number of connected components in the graph_.

![](https://assets.leetcode.com/uploads/2021/03/14/conn1-graph.jpg)

```java
Input: n = 5, edges = [[0,1],[1,2],[3,4]]
Output: 2
//Union find

class Solution {
    public int countComponents(int n, int[][] edges) {
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        int components = n;
        for (int[] e : edges) {
            int p1 = findParent(parent, e[0]);
            int p2 = findParent(parent, e[1]);
            if (p1 != p2) {
                parent[p1] = p2; // Union 2 component
                components--;
            }
        }
        return components;
    }
    private int findParent(int[] parent, int i) {
        while (i != parent[i]) i = parent[i];
        return i; // Without Path Compression
    }
}

BFS

class Solution {
    public int countComponents(int n, int[][] edges) {
        int count = 0;
        List<List<Integer>> adjList = new ArrayList<>();
        
        boolean[] visited = new boolean[n];
        
        for (int i = 0; i < n; i++) {
            adjList.add(new ArrayList<>());
        }
        
        for (int[] edge : edges) {
            adjList.get(edge[0]).add(edge[1]);
            adjList.get(edge[1]).add(edge[0]);
        }
        
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                count++;
                bfs(i, visited, adjList);                
            }
        }
        return count;
    }
    private void bfs(int i, boolean[] visited, List<List<Integer>> adjList) {
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(i);
        while (!queue.isEmpty()) {
            int idx = queue.poll();
            visited[idx] = true;
            for (int next : adjList.get(idx)) {
                if (!visited[next]) {
                    queue.offer(next);
                }
            }
        }
    }
}


```

## 57 Insert Interval



You are given an array of non-overlapping intervals `intervals` where `intervals[i] = [starti, endi]` represent the start and the end of the `ith` interval and `intervals` is sorted in ascending order by `starti`. You are also given an interval `newInterval = [start, end]` that represents the start and end of another interval.

Insert `newInterval` into `intervals` such that `intervals` is still sorted in ascending order by `starti` and `intervals` still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return `intervals` _after the insertion_.

```
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]

class Solution {
    public int[][] insert(int[][] intervals, int[] newInterval) {
        int[][] res = new int[intervals.length + 1][2];
        int i = 0, k = 0;
        while (i < intervals.length && newInterval[0] > intervals[i][1]) {
            res[k++] = intervals[i];
            i++;
        }
        int[] tmp = new int[]{newInterval[0], newInterval[1]};
        while (i < intervals.length && newInterval[1] >= intervals[i][0]) {
            tmp[0] = Math.min(tmp[0], intervals[i][0]);
            tmp[1] = Math.max(tmp[1], intervals[i][1]);
            i++;
        }
        res[k++] = tmp;
        while (i < intervals.length) {
            res[k++] = intervals[i];
            i++;
        }
        return Arrays.copyOf(res,k); 
    }
}

/*在有序区间列表里插入一个区间，取得新区间的左边界newStart，右边界newEnd
遍历原区间
当newStart 大于 当前区间的右边界时
说明两个区间没有交集，不用合并，又因为原区间列表有序
所以直接将当前区间排入新的区间列表中
否则当newEnd 大于 当前区间的左边界时
两个区间有重合
所以将两个区间的最小左边界和最大右边界重新组合为新的区间，即为合并
将其排入新区间列表中
否则当区间列表没有遍历完时
将剩下的排入新区间列表中去*/


```

## 273 Integer to English Words

```
Convert a non-negative integer num to its English words representation.
Input: num = 123
Output: "One Hundred Twenty Three"

class Solution {
    public String numberToWords(int num) {
        if(num == 0) return "Zero";
        return helper(num);  
    }
    
    public String helper(int num ) {
    
        String[] words = new String[] {"", "One", "Two", "Three", "Four", 
        "Five", "Six", "Seven", "Eight", "Nine", "Ten",
  "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen",
   "Eighteen", "Nineteen"}; // Fifteen, Twelve Forty Nineteen Ninety Hundred
String[] words1 = new String[]{"","","Twenty ", "Thirty ", 
"Forty ", "Fifty ", "Sixty ",  "Seventy ", "Eighty ", "Ninety "};

       StringBuilder sb = new StringBuilder();
        if (num >= 1000000000) {
            sb.append(helper(num/1000000000)).append(" Billion ");
            num %= 1000000000; 
        }
        if (num >= 1000000) {
            sb.append(helper(num/1000000)).append(" Million ");
            num %= 1000000; 
        }
        if (num >= 1000) {
            sb.append(helper(num/1000)).append(" Thousand ");
            num %= 1000; 
        }
        if (num >= 100) {
            sb.append(helper(num/100)).append(" Hundred ");
            num %= 100; 
        }
        if (num >= 20) {
             sb.append(words1[num/10]).append(words[num%10]);
        } else {
          sb.append(words[num]);  
        }
        return sb.toString().trim();
    }
}

```

## 706 Design HashMap

Design a HashMap without using any built-in hash table libraries.

Implement the `MyHashMap` class:

* `MyHashMap()` initializes the object with an empty map.
* `void put(int key, int value)` inserts a `(key, value)` pair into the HashMap. If the `key` already exists in the map, update the corresponding `value`.
* `int get(int key)` returns the `value` to which the specified `key` is mapped, or `-1` if this map contains no mapping for the `key`.
* `void remove(key)` removes the `key` and its corresponding `value` if the map contains the mapping for the `key`.

```java
class MyHashMap {
        final ListNode[] nodes = new ListNode[10000];

        public void put(int key, int value) {
            int i = idx(key);
            if (nodes[i] == null)
                nodes[i] = new ListNode(-1, -1);
            ListNode prev = find(nodes[i], key);
            if (prev.next == null)
                prev.next = new ListNode(key, value);
            else prev.next.val = value;
        }

        public int get(int key) {
            int i = idx(key);
            if (nodes[i] == null)
                return -1;
            ListNode node = find(nodes[i], key);
            return node.next == null ? -1 : node.next.val; 
        }

        public void remove(int key) {
            int i = idx(key);
            if (nodes[i] == null) return;
            ListNode prev = find(nodes[i], key);
            if (prev.next == null) return;
            prev.next = prev.next.next;
        }

        int idx(int key) { return key % nodes.length;}

        ListNode find(ListNode bucket, int key) {
            ListNode node = bucket, prev = null;
            while (node != null && node.key != key) {
                prev = node;
                node = node.next;
            }
            return prev;
        }

        class ListNode {
            int key, val;
            ListNode next;

            ListNode(int key, int val) {
                this.key = key;
                this.val = val;
            }
        }
    }
```

## 72 Edit Distance

Given two strings `word1` and `word2`, return _the minimum number of operations required to convert `word1` to `word2`_.

You have the following three operations permitted on a word:

* Insert a character
* Delete a character
* Replace a character

```
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')

class Solution {
    public int minDistance(String word1, String word2) {
        int m = word1.length(); int n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i ++){
            dp[i][0] = i;
        }
        for (int j = 0; j <= n; j ++){
            dp[0][j] = j;
        }
        
        for (int i = 1; i <= m; i ++){
            for (int j = 1; j <= n; j ++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)){
                    dp[i][j] = dp[i -1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i][j - 1], 
                    Math.min(dp[i - 1][j],dp[i - 1][j - 1])) + 1;
                }
            }
        }
        return dp[m][n];
    }
}
```

## 69 Sqrt(x)

Given a non-negative integer `x`, compute and return _the square root of_ `x`.

Since the return type is an integer, the decimal digits are **truncated**, and only **the integer part** of the result is returned.

**Note:** You are not allowed to use any built-in exponent function or operator, such as `pow(x, 0.5)` or `x ** 0.5`.

```
Input: x = 4
Output: 2

Input: x = 8
Output: 2
Explanation: The square root of 8 is 2.82842..., 
and since the decimal part is truncated, 2 is returned.

class Solution {
    public int mySqrt(int x) {
        if (x <= 1) return x;
        int left = 1; int right = x; int result = 0;
        while (left <= right) {
          int  mid = left + (right - left) / 2;
            if (mid == x / mid) {
                return mid;
            } else if (mid < x / mid ) {
                result = mid;
                left = mid + 1;
            } else {
                
                right = mid - 1;
            }    
        }
         return result;
    }
}
```

## 50 Pow(x, n)

Implement [pow(x, n)](http://www.cplusplus.com/reference/valarray/pow/), which calculates `x` raised to the power `n` (i.e., `xn`).

```
Input: x = 2.00000, n = 10
Output: 1024.00000

class Solution {
    public double myPow(double x, int n) {
        if (n < 0) {
            x = 1/x;
            n = -n;
        }
        return myNewPow( x, n);
    }
    
    public double myNewPow(double x, int n) { 
    // double type need to keep not int.
        if (n == 0) return 1.00; // 1.00 not 1
        if (n == 1) return x;
        double half = myNewPow(x, n/2);
        if (n % 2 == 0){  
            return half * half;
        } else {
            return half * half * x;     
        }
    }
}


```

## 160 Intersection of Two Linked Lists

Given the heads of two singly linked-lists `headA` and `headB`, return _the node at which the two lists intersect_. If the two linked lists have no intersection at all, return `null`.



**Example 1:**

![](https://assets.leetcode.com/uploads/2021/03/05/160\_example\_1\_1.png)

```
Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], 
skipA = 2, skipB = 3
Output: Intersected at '8'
Explanation: The intersected node's value is 8 
(note that this must not be 0 if the two lists intersect).
From the head of A, it reads as [4,1,8,4,5]. 
From the head of B, it reads as [5,6,1,8,4,5]. 
There are 2 nodes before the intersected node in A; 
There are 3 nodes before the intersected node in B.

public class Solution {
   public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    if (headA == null || headB == null) return null;
    ListNode pA = headA, pB = headB;
    while (pA != pB) {
        pA = pA == null ? headB : pA.next;
        pB = pB == null ? headA : pB.next;
    }
    return pA;
}
}

```

## 34 Find First and Last Position of Element in Sorted Array



Given an array of integers `nums` sorted in non-decreasing order, find the starting and ending position of a given `target` value.

If `target` is not found in the array, return `[-1, -1]`.

You must write an algorithm with `O(log n)` runtime complexity.

```
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

## 103 Binary Tree Zigzag Level Order Traversal

```
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        boolean isForward = true;
        while(!queue.isEmpty()) {
            LinkedList<Integer> curLevel = new LinkedList<>();
            int size = queue.size();
            for(int i = 0; i < size; i ++) {
                TreeNode cur = queue.poll();
                if(isForward) {
                    curLevel.add(cur.val);
                } else {
                    curLevel.addFirst(cur.val);
                }
                if(cur.left != null) queue.offer(cur.left);
                if(cur.right != null) queue.offer(cur.right);
                
            }
            isForward = !isForward;  
            res.add(curLevel);
        }
        return res;
    }
}
```

## 449 Serialize and Deserialize BST

```
public class Codec {
  //use upper and lower boundaries to check whether we should add null
    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        serialize(root, sb);
        return sb.toString();
    }
    
    public void serialize(TreeNode root, StringBuilder sb) {
        if (root == null) return;
        sb.append(root.val).append(",");
        serialize(root.left, sb);
        serialize(root.right, sb);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data.isEmpty()) return null;
        Queue<String> q = new LinkedList<>(Arrays.asList(data.split(",")));
        return deserialize(q, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }
    
    public TreeNode deserialize(Queue<String> q, int lower, int upper) {
        if (q.isEmpty()) return null;
        String s = q.peek();
        int val = Integer.parseInt(s);
        if (val < lower || val > upper) return null;
        q.poll();
        TreeNode root = new TreeNode(val);
        root.left = deserialize(q, lower, val);
        root.right = deserialize(q, val, upper);
        return root;
    }
}
```

## 1197 Minimum Knight Moves

In an **infinite** chess board with coordinates from `-infinity` to `+infinity`, you have a **knight** at square `[0, 0]`.

A knight has 8 possible moves it can make, as illustrated below. Each move is two squares in a cardinal direction, then one square in an orthogonal direction.

![](https://assets.leetcode.com/uploads/2018/10/12/knight.png)

Return _the minimum number of steps needed to move the knight to the square_ `[x, y]`. It is guaranteed the answer exists.

```
Input: x = 2, y = 1
Output: 1
Explanation: [0, 0] → [2, 1]

class Solution {
    public int minKnightMoves(int x, int y) {
        // no need to work with negatives, sames # steps in any quadrant
        x = Math.abs(x);
        y = Math.abs(y);
        
        // special case dips negative, return early
        if (x == 1 && y == 1) return 2;
        
        Queue<int[]> queue = new LinkedList<>();
        boolean[][] visited = new boolean[x+3][y+3];

        queue.add(new int[]{0, 0});

        int steps = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();

            for (int i=0; i < size; i++) {
                int[] pos = queue.remove();
                int pX = pos[0];
                int pY = pos[1];             
                if (pX == x && pY == y) return steps;     
     // don't need to go beyond these points except for possible first move
                if (pX < 0 || pY < 0 || pX > x+2 || pY > y+2) continue;

                if (visited[pX][pY]) continue;
                visited[pX][pY] = true;

                queue.add(new int[]{pX+2, pY+1});
                queue.add(new int[]{pX+2, pY-1});
                queue.add(new int[]{pX+1, pY+2});
                queue.add(new int[]{pX+1, pY-2});
                queue.add(new int[]{pX-2, pY+1});
                queue.add(new int[]{pX-2, pY-1}); 
                queue.add(new int[]{pX-1, pY+2});
                queue.add(new int[]{pX-1, pY-2}); 
            }
            steps++;
        }
        return -1;
    }
}
```

953 Verifying an Alien Dictionary

In an alien language, surprisingly, they also use English lowercase letters, but possibly in a different `order`. The `order` of the alphabet is some permutation of lowercase letters.

Given a sequence of `words` written in the alien language, and the `order` of the alphabet, return `true` if and only if the given `words` are sorted lexicographically in this alien language.

```java
Input: words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz"
Output: true
Explanation: As 'h' comes before 'l' in this language, 
then the sequence is sorted.

class Solution {
     int[] dic = new int[26];
    public boolean isAlienSorted(String[] words, String order) {
        for (int i = 0; i < order.length(); i ++) {
            dic[order.charAt(i) - 'a'] = i;
        }
        
        for (int i = 1; i < words.length; i ++) {
            if (bigger(words[i - 1], words[i])){
                return false;
            }
        }
         
        return true;
    }
    
    public boolean bigger(String w1, String w2) {
        int m = w1.length(); int n = w2.length();
        for (int i = 0; i < Math.min(m, n); i ++){
            char c1 = w1.charAt(i); char c2 = w2.charAt(i);
            if(c1!=c2){
                return dic[c1 - 'a'] > dic[c2 -'a'];
            }
        }
        return m > n;
    }
}j
```

## 100 Same Tree

Given the roots of two binary trees `p` and `q`, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

```java
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p == null && q == null) return true;
        if(p == null || q == null) return false;
        if(p.val != q.val) return false; 
        boolean left = isSameTree(p.left, q.left);
        boolean right = isSameTree(p.right, q.right);
        return left && right;
        
    }
}
```

## 450 Delete Node in a BST

```
class Solution {
    public TreeNode deleteNode(TreeNode root, int key) {
        if(root == null){
            return null;
        }
        //当前节点值比key小，则需要删除当前节点的左子树中key对应的值，
        并保证二叉搜索树的性质不变
        if(key < root.val){
            root.left = deleteNode(root.left,key);
        }
        //当前节点值比key大，则需要删除当前节点的右子树中key对应的值，
        并保证二叉搜索树的性质不变
        else if(key > root.val){
            root.right = deleteNode(root.right,key);
        }
        //当前节点等于key，则需要删除当前节点，并保证二叉搜索树的性质不变
        else{
            //当前节点没有左子树
            if(root.left == null){
                return root.right;
            }
            //当前节点没有右子树
            else if(root.right == null){
                return root.left;
            }
            //当前节点既有左子树又有右子树
            else{
                TreeNode node = root.right;
                //找到当前节点右子树最左边的叶子结点
                while(node.left != null){
                    node = node.left;
                }
                //将root的左子树放到root的右子树的最下面的左叶子节点的左子树上
                node.left = root.left;
                return root.right;
            }
        }
        return root;
    }
}
```

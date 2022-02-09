# Linkedin 50\~100

## **5 : Longest Palindromic Substring**

```java
// Some code
Given a string s, return the longest palindromic substring in s.
Example 1:
Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.
Example 2:
Input: s = "cbbd"
Output: "bb"

class Solution {
   public String longestPalindrome(String s) {
    String res = "";
    for (int i = 0; i < s.length(); i++) {
        // 以 s[i] 为中心的最长回文子串
        String s1 = palindrome(s, i, i);
        // 以 s[i] 和 s[i+1] 为中心的最长回文子串
        String s2 = palindrome(s, i, i + 1);
        // res = longest(res, s1, s2)
        res = res.length() > s1.length() ? res : s1;
        res = res.length() > s2.length() ? res : s2;
    }
    return res;
}
    
    String palindrome(String s, int l, int r) {
    // 防止索引越界
    while (l >= 0 && r < s.length()
            && s.charAt(l) == s.charAt(r)) {
        // 向两边展开
        l--; r++;
    }
    // 返回以 s[l] 和 s[r] 为中心的最长回文串
    return s.substring(l + 1, r);
    }
}
```

## 701:  Insert into a Binary Search Tree

```java
You are given the root node of a binary search tree (BST) 
and a value to insert into the tree. 
Return the root node of the BST after the insertion. 
It is guaranteed that the new value does not exist in the original BST.

class Solution {
    TreeNode insertIntoBST(TreeNode root, int val) {
    // 找到空位置插入新节点
    if (root == null) return new TreeNode(val);
    // if (root.val == val)
    //     BST 中一般不会插入已存在元素
    if (root.val < val) 
        root.right = insertIntoBST(root.right, val);
    if (root.val > val) 
        root.left = insertIntoBST(root.left, val);
    return root;
}
}

```

## 26 Remove Duplicates from Sorted Array

```java
// Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
Explanation: Your function should return k = 5, 
with the first five elements of nums being 0, 1, 2, 3, and 4 respectively.
It does not matter what you leave beyond the returned k 
(hence they are underscores).

class Solution {
    public int removeDuplicates(int[] nums) {
        if (nums.length == 1) return 1;
        int index = 0; 
        for (int i = 1; i < nums.length; i ++) {
            if(nums[i] != nums[i - 1]){
                index ++;
                nums[index] = nums[i];
            }
        }
        return index + 1;
    }
}
```

## 39 Combination Sum

```java
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]

class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new LinkedList<>();
        List<Integer> temp = new LinkedList<>();
        combinationSumHelper(temp, res, candidates, target, 0);
        return res;
        
    }
    
    public void combinationSumHelper(List<Integer> temp, List<List<Integer>> res, 
                              int[] candidates, int target, int index ){
        if (target < 0) return;
        if (target == 0){
            res.add(new LinkedList(temp));
            return;
        }
        for (int i = index; i < candidates.length; i ++) {
            temp.add(candidates[i]);
        combinationSumHelper(temp, res, candidates, target - candidates[i],  i);
            temp.remove(temp.size() - 1);
        }
    }  
}
```

## 146 LRU cache

```java
class LRUCache {
    int cap;
    LinkedHashMap<Integer, Integer> cache = new LinkedHashMap<>();
    public LRUCache(int capacity) { 
        this.cap = capacity;
    }
    
    public int get(int key) {
        if (!cache.containsKey(key)) {
            return -1;
        }
        // 将 key 变为最近使用
        makeRecently(key);
        return cache.get(key);
    }
    
    public void put(int key, int val) {
        if (cache.containsKey(key)) {
            // 修改 key 的值
            cache.put(key, val);
            // 将 key 变为最近使用
            makeRecently(key);
            return;
        }
        
        if (cache.size() >= this.cap) {
            // 链表头部就是最久未使用的 key
            int oldestKey = cache.keySet().iterator().next();
            cache.remove(oldestKey);
        }
        // 将新的 key 添加链表尾部
        cache.put(key, val);
    }
    
    private void makeRecently(int key) {
        int val = cache.get(key);
        // 删除 key，重新插入到队尾
        cache.remove(key);
        cache.put(key, val);
    }
}
```

## 347 Top K Frequent Elements

```
Given an integer array nums and an integer k, return the k most frequent elements. 
You may return the answer in any order.
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for(int num: nums){
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        PriorityQueue<Map.Entry<Integer,Integer>> pq = 
        new PriorityQueue<>((a, b) -> a.getValue() - b.getValue());
        for(Map.Entry<Integer,Integer> entry : map.entrySet()){
            pq.offer(entry);
            if(pq.size() > k){
                pq.poll();
            }
        }
        int[] res = new int[k];
        for(int i = 0; i < res.length; i ++){
            res[i] = pq.poll().getKey();
        }
        return res;
    }
}
```

## 88 Merge Sorted Array

```
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
The result of the merge is [1,2,2,3,5,6] 
with the underlined elements coming from nums1.


class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int last = m + n - 1;
        int index1 = m - 1;
        int index2 = n - 1;
        while (index1 >= 0 && index2 >= 0){
            if (nums1[index1] > nums2[index2]){
                nums1[last--] = nums1[index1--];
            } else {
                nums1[last--] = nums2[index2--];
            }
        }
        while (index2 >= 0){
            nums1[last--] = nums2[index2--];
        }
        
        
    }
}
```

## 695 Max Area of Island

```
You are given an m x n binary matrix grid. 
An island is a group of 1's (representing land) connected 4-directionally 
(horizontal or vertical.) You may assume all four edges of the grid are 
surrounded by water.
The area of an island is the number of cells with a value 1 in the island.
Return the maximum area of an island in grid. If there is no island, return 0.

class Solution {
    public int maxAreaOfIsland(int[][] grid) {
        int row = grid.length;
        int col = grid[0].length;
        int max = 0;
        for (int i = 0; i < row; i ++){
            for (int j = 0; j < col; j ++){
                if (grid[i][j] == 1){
                    int area = getArea(grid, i , j);
                    max = Math.max(area, max);
                }
            }
        }
        return max;
    }
    
    public int getArea(int[][] grid, int i, int j){
        if (!inArea(grid, i, j)){
            return 0;
        }
        if (grid[i][j] != 1){
            return 0;
        }
        grid[i][j] = 2;
        int sum = 1 + 
         getArea(grid, i - 1, j) + 
         getArea(grid, i, j - 1) + 
         getArea(grid, i + 1, j) + 
         getArea(grid, i, j + 1);   
        return sum; 
    }
    
    public boolean inArea(int[][] grid, int i, int j){
        return i >= 0 && i < grid.length && j >= 0 && j < grid[0].length;
    }
}
```

## 560 Subarray Sum Equals K

```java
iven an array of integers nums and an integer k, 
return the total number of continuous subarrays whose sum equals to k.
Input: nums = [1,1,1], k = 2
Output: 2

class Solution {
    public int subarraySum(int[] nums, int k) {
        HashMap<Integer, Integer> map = new HashMap<>();
        int sum = 0;
        map.put(0, 1);
         int res = 0;
        for (int i = 0; i < nums.length; i ++) {
            sum += nums[i];
             if(map.containsKey(sum - k)) {
                res += map.get(sum - k);
            }
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return res;
    }
}
```

## 744 Find Smallest Letter Greater Than Target

```java
Given a characters array letters that is sorted in non-decreasing order 
and a character target, return the smallest character 
in the array that is larger than target.
Input: letters = ["c","f","j"], target = "a"
Output: "c"

 class Solution {
    public char nextGreatestLetter(char[] a, char x) {
        int n = a.length;

        //hi starts at 'n' rather than the usual 'n - 1'. 
        //It is because the terminal condition is 'lo < hi' and if hi starts from 'n - 1', 
        //we can never consider value at index 'n - 1'
        int lo = 0, hi = n;
        //Terminal condition is 'lo < hi', to avoid infinite loop when target is smaller than the first element
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (a[mid] > x)     hi = mid;
            else    lo = mid + 1;                 //a[mid] <= x
        }
 
        //Because lo can end up pointing to index 'n', in which case we return the first element
        return a[lo % n];
    }
   }
```

## 101 Symmetric Tree

```java
// Some code
class Solution {
 public boolean isSymmetric(TreeNode root) {
    return root==null || isSymmetricHelp(root.left, root.right);
}

private boolean isSymmetricHelp(TreeNode left, TreeNode right){
    if(left==null && right==null) return true;
    if(left == null || right == null) return false;
    if(left.val!=right.val) return false;
    return isSymmetricHelp(left.left, right.right) 
        && isSymmetricHelp(left.right, right.left);
}
}
```

## 151 Reverse Words in a String

Given an input string `s`, reverse the order of the **words**.

A **word** is defined as a sequence of non-space characters. The **words** in `s` will be separated by at least one space.

Return _a string of the words in reverse order concatenated by a single space._

**Note** that `s` may contain leading or trailing spaces or multiple spaces between two words. The returned string should only have a single space separating the words. Do not include any extra spaces.

```java
Input: s = "the sky is blue"
Output: "blue is sky the"
class Solution {
    public String reverseWords(String s) {
        if (s == null || s.length() == 0 || s.trim().isEmpty()) return "";
        String[] strs = s.trim().split(" ");
        int i = 0; int j = strs.length - 1;
        while (i < j){
            String temp = strs[i];
            strs[i ++] = strs[j];
            strs[j --] = temp;
        }
       StringBuilder sb = new StringBuilder();
       for (String str: strs){
           if(!str.trim().isEmpty()){
               sb.append(str).append(" "); 
           }
       } 
       return sb.toString().trim(); 
    }
}
```

## 61 Rotate List



Given the `head` of a linked list, rotate the list to the right by `k` places.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/11/13/rotate1.jpg)

```java
Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]
class Solution {
    public ListNode rotateRight(ListNode head, int k) {
        if(head == null || head.next == null) return head;
        ListNode fast = head; int len = 1;
        while(fast.next!=null){
            len++;
            fast = fast.next;
        }
        int move = k % len;
        if(move == 0) return head;
        
        ListNode slow = head;
        for(int i = 0; i < len - move -1 ; i ++){
          slow = slow.next;
        }
        
        fast.next = head;
        ListNode resNode = slow.next;
        slow.next = null;
        return resNode;    
    }
}
```

## 46 Permutations

```java
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new LinkedList<>();
        if (nums == null || nums.length == 0) return res;
        Deque<Integer> temp = new LinkedList<>();
        permuteHelper(res, temp, nums, 0);
        return res;
    }
    
    public void permuteHelper( List<List<Integer>> res, Deque<Integer> temp, int[] nums, int start){
        if (temp.size() == nums.length){
            res.add(new LinkedList(temp));
            return;
        }
        
        for (int i = 0; i < nums.length; i ++) {
            if (temp.contains(nums[i])) continue;
            temp.addLast(nums[i]);
            permuteHelper(res, temp, nums, i);
            temp.removeLast();
        } 
    }
}
```

## 128 Longest Consecutive Sequence



Given an unsorted array of integers `nums`, return _the length of the longest consecutive elements sequence._

You must write an algorithm that runs in `O(n)` time.

```java
Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. 
Therefore its length is 4.

class Solution {
    public int longestConsecutive(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        HashMap<Integer, Integer> map = new HashMap<>(); //k-v: 元素 - 元素位置
        int[] parent = new int[nums.length];
        
        for (int i = 0; i < nums.length; i++) {
            parent[i] = i;
            if (map.containsKey(nums[i])) {// 先把所有元素设为自己的leader，顺便除重
                continue;
            }
            map.put(nums[i], i);
        }
        
        for (int i = 0; i < nums.length; i++) {
            // 遇到相邻的元素，并且leader不一样，做union操作
            if (map.containsKey(nums[i] - 1) && 
            find(i, parent) != find(map.get(nums[i] - 1), parent)) {
                union(i, map.get(nums[i] - 1), parent);
            }
            if (map.containsKey(nums[i] + 1) && 
            find(i, parent) != find(map.get(nums[i] + 1), parent)) {
                union(i, map.get(nums[i] + 1), parent);
            }
        }
        
        // 找到parent中value最多的元素，代表某老大出现的次数最多，因此该集团的成员最多
        int maxLen = 0;
        int[] count = new int[parent.length];
        for (int val : map.values()) {
            count[find(val, parent)]++;
            maxLen = Math.max(maxLen, count[parent[val]]);
        }
        return maxLen;
    }
    
    // 以下为Union Find代码，
    private int find(int p, int[] parent) {
        if (p == parent[p]) {
            return parent[p];
        }
        parent[p] = find(parent[p], parent); // 路径压缩，到最终leader
        /**路径压缩的迭代写法
        while (p != parent[p]) {
            parent[p] = parent[parent[p]];
            p = parent[p];
        }
        */
        return parent[p];
    }
    private void union(int p, int q, int[] parent) {
        int f1 = find(p, parent);
        int f2 = find(q, parent);
        if (f1 != f2) {
            parent[f1] = f2;
        }
    }
}
 
```


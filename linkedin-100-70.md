# Linkedin 100 \~ 70

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
            //同一个元素可以多次用 所以继续用i, 如果不能多次用就是 i + 1;
            // permutaion no i.
        combinationSumHelper(temp, res, candidates, target - candidates[i],  i);
            temp.remove(temp.size() - 1);
        }
    }  
}
```

## 146 LRU cache

```java
class LRUCache {
    
    class Node{
        int key;
        int value;
        Node next;
        Node pre;
        Node(int key, int value){
            this.key = key;
            this.value = value;
        }
    }
    
    class DoubleList{
        Node head;
        Node tail;
        int size;
        DoubleList(){
            head = new Node(-1, -1);
            tail = new Node(-1,-1);
            head.next = tail;
            tail.pre = head;
            size = 0; 
        }
        
       public void addLast(Node node){
           tail.pre.next = node;
           node.pre = tail.pre;
           node.next = tail;
           tail.pre = node;
           size ++;
        }
        
        public void remove(Node node){
           node.pre.next = node.next;
            node.next.pre = node.pre;
            node.pre = null;
            node.next = null;
            size --;
        }
        
        public Node removeFirst(){
            if(head.next == tail) return null;
            Node first = head.next;
            remove(first);
            return first;
        }
        
        public int size(){
            return size;
        }
        
    }
    
        HashMap<Integer, Node> map = new HashMap<>();
        DoubleList cache = new DoubleList();
        int capacity;
    
    public LRUCache(int capacity) {
        this.capacity = capacity;
    }
    
    public int get(int key) {
        if(!map.containsKey(key)) return -1;
         int val = map.get(key).value;
         put(key, val);
        return val;
    }
    
    public void put(int key, int value) {
        if(map.containsKey(key)){
            Node cur = map.get(key);
            cache.remove(cur);
            cur.value = value;
            cache.addLast(cur);
            return;
        }
        if(capacity == cache.size()){
            Node first = cache.removeFirst();
            map.remove(first.key);
        }
        Node node = new Node(key, value);
        map.put(key, node);
        cache.addLast(node);
        
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

## 81 Search in Rotated Sorted Array II

There is an integer array `nums` sorted in non-decreasing order (not necessarily with **distinct** values).

Before being passed to your function, `nums` is **rotated** at an unknown pivot index `k` (`0 <= k < nums.length`) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]` (**0-indexed**). For example, `[0,1,2,4,4,4,5,6,6,7]` might be rotated at pivot index `5` and become `[4,5,6,6,7,0,1,2,4,4]`.

Given the array `nums` **after** the rotation and an integer `target`, return `true` _if_ `target` _is in_ `nums`_, or_ `false` _if it is not in_ `nums`_._

You must decrease the overall operation steps as much as possible.

```
class Solution {
    public boolean search(int[] nums, int target) {
         int start = 0, end = nums.length - 1, mid = -1;
        while(start <= end) {
            mid = (start + end) / 2;
            if (nums[mid] == target) {
                return true;
            }
            //If we know for sure right side is sorted or left side is unsorted
            if (nums[mid] < nums[end] || nums[mid] < nums[start]) {
                if (target > nums[mid] && target <= nums[end]) {
                    start = mid + 1;
                } else {
                    end = mid - 1;
                }
            //If we know for sure left side is sorted or right side is unsorted
            } else if (nums[mid] > nums[start] || nums[mid] > nums[end]) {
                if (target < nums[mid] && target >= nums[start]) {
                    end = mid - 1;
                } else {
                    start = mid + 1;
                }
     //If we get here, that means nums[start] == nums[mid] == nums[end], 
     //then shifting out
    //any of the two sides won't change the result 
    //but can help remove duplicate from
            //consideration, here we just use end-- but left++ works too
            } else {
                end--;
            }
        }     
        return false;
        
    }
}
```

## 56 Merge Intervals

Given an array of `intervals` where `intervals[i] = [starti, endi]`, merge all overlapping intervals, and return _an array of the non-overlapping intervals that cover all the intervals in the input_.

**Example 1:**

```
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
```

```
// Some code
class Solution {
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, (a,b) -> a[0] - b[0]); // -> not =>
        int k = 0;
        int i = 0;
        while (i < intervals.length) {
            int start = intervals[i][0];
            int end = intervals[i][1];
            while (i < intervals.length - 1 && end >= intervals[i + 1][0]){ 
            // test overlap need to first check end with next start
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

## 367 Valid Perfect Square

Given a **positive** integer _num_, write a function which returns True if _num_ is a perfect square else False.

**Follow up:** **Do not** use any built-in library function such as `sqrt`.

```
// Some code
Input: num = 16
Output: true
class Solution {
    public boolean isPerfectSquare(int num) {
        long left = 1, right = num;
        while (left <= right) {
            long mid = (left + right) / 2;
            if (mid * mid == num) return true; // check if mid is perfect square
            if (mid * mid < num) { // mid is small -> go right to increase mid
                left = mid + 1;
            } else {
                right = mid - 1; // mid is large -> to left to decrease mid
            }
        }
        return false;
    }
}
```

## 460 LFU

```
class LFUCache {

    // key 到 val 的映射，我们后文称为 KV 表
    HashMap<Integer, Integer> keyToVal;
    // key 到 freq 的映射，我们后文称为 KF 表
    HashMap<Integer, Integer> keyToFreq;
    // freq 到 key 列表的映射，我们后文称为 FK 表
    HashMap<Integer, LinkedHashSet<Integer>> freqToKeys;
    // 记录最小的频次
    int minFreq;
    // 记录 LFU 缓存的最大容量
    int cap;

    public LFUCache(int capacity) {
        keyToVal = new HashMap<>();
        keyToFreq = new HashMap<>();
        freqToKeys = new HashMap<>();
        this.cap = capacity;
        this.minFreq = 0;
    }

    public int get(int key) {
        if (!keyToVal.containsKey(key)) {
            return -1;
        }
        // 增加 key 对应的 freq
        increaseFreq(key);
        return keyToVal.get(key);
    }

    public void put(int key, int val) {
        if (this.cap <= 0) return;

        /* 若 key 已存在，修改对应的 val 即可 */
        if (keyToVal.containsKey(key)) {
            keyToVal.put(key, val);
            // key 对应的 freq 加一
            increaseFreq(key);
            return;
        }

        /* key 不存在，需要插入 */
        /* 容量已满的话需要淘汰一个 freq 最小的 key */
        if (this.cap <= keyToVal.size()) {
            removeMinFreqKey();
        }

        /* 插入 key 和 val，对应的 freq 为 1 */
        // 插入 KV 表
        keyToVal.put(key, val);
        // 插入 KF 表
        keyToFreq.put(key, 1);
        // 插入 FK 表
        freqToKeys.putIfAbsent(1, new LinkedHashSet<>());
        freqToKeys.get(1).add(key);
        // 插入新 key 后最小的 freq 肯定是 1
        this.minFreq = 1;
    }

    private void increaseFreq(int key) {
        int freq = keyToFreq.get(key);
        /* 更新 KF 表 */
        keyToFreq.put(key, freq + 1);
        /* 更新 FK 表 */
        // 将 key 从 freq 对应的列表中删除
        freqToKeys.get(freq).remove(key);
        // 将 key 加入 freq + 1 对应的列表中
        freqToKeys.putIfAbsent(freq + 1, new LinkedHashSet<>());
        freqToKeys.get(freq + 1).add(key);
        // 如果 freq 对应的列表空了，移除这个 freq
        if (freqToKeys.get(freq).isEmpty()) {
            freqToKeys.remove(freq);
            // 如果这个 freq 恰好是 minFreq，更新 minFreq
            if (freq == this.minFreq) {
                this.minFreq++;
            }
        }
    }

    private void removeMinFreqKey() {
        // freq 最小的 key 列表
        LinkedHashSet<Integer> keyList = freqToKeys.get(this.minFreq);
        // 其中最先被插入的那个 key 就是该被淘汰的 key
        int deletedKey = keyList.iterator().next();
        /* 更新 FK 表 */
        keyList.remove(deletedKey);
        if (keyList.isEmpty()) {
            freqToKeys.remove(this.minFreq);
            // 问：这里需要更新 minFreq 的值吗？
        }
        /* 更新 KV 表 */
        keyToVal.remove(deletedKey);
        /* 更新 KF 表 */
        keyToFreq.remove(deletedKey);
    }
}
```


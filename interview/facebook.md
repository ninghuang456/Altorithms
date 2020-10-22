---
description: FaceBook
---

# Frequency1~40

## 1570  Dot Product of Two Sparse Vectors

```java
class SparseVector {
  Map<Integer, Integer> indexMap = new HashMap<>();
  int n = 0;
  SparseVector(int[] nums) {
    for (int i = 0; i < nums.length; i++)
      if (nums[i] != 0)
        indexMap.put(i, nums[i]);
    n = nums.length;
  }
  
	// Return the dotProduct of two sparse vectors
  public int dotProduct(SparseVector vec) {
    if (indexMap.size() == 0 || vec.indexMap.size() == 0) return 0;
    if (indexMap.size() > vec.indexMap.size())
      return vec.dotProduct(this);
    int productSum = 0;
    for (Map.Entry<Integer, Integer> entry : indexMap.entrySet()) {
      int index = entry.getKey();
      Integer vecValue = vec.indexMap.get(index);
      if (vecValue == null) continue; 
      productSum += (entry.getValue() * vecValue);
    }
    return productSum;
  }
}
```

## 398  Random Pick Index

```java

public class Solution {

    int[] nums;
    Random rnd;

    public Solution(int[] nums) {
        this.nums = nums;
        this.rnd = new Random();
    }
    
    public int pick(int target) {
        int result = -1;
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != target)
                continue;
            if (rnd.nextInt(++count) == 0)
                result = i;
        }
        
        return result;
    }
}
```

## 34  Find First and Last Position of Element in Sorted Array

```java
//Input: nums = [5,7,7,8,8,10], target = 8
//Output: [3,4]

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

## 339 Nested List Weight Sum

```java
//Input: [[1,1],2,[1,1]]
//Output: 10 
//Explanation: Four 1's at depth 2, one 2 at depth 1.

class Solution {
    int result;
    public int depthSum(List<NestedInteger> nestedList) {
        result = 0;
        dfs(nestedList, 1);
        return result;
    }
    private void dfs(List<NestedInteger> nestedList, int depth) {
        for (NestedInteger ni : nestedList) {
            if (ni.isInteger()) {
                result += ni.getInteger() * depth;
            } else {
                dfs(ni.getList(), depth + 1);
            }
        }
    }
}
```

## 721 Accounts Merge

```java
//accounts = [["John", "johnsmith@mail.com", "john00@mail.com"], ["John", "johnnybravo@mail.com"], ["John", "johnsmith@mail.com", "john_newyork@mail.com"], ["Mary", "mary@mail.com"]]
//Output: [["John", 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com'],  ["John", "johnnybravo@mail.com"], ["Mary", "mary@mail.com"]]
class Solution {
    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        if (accounts.size() == 0) {
            return new ArrayList<>();
        }

        int n = accounts.size();
        UnionFind uf = new UnionFind(n);

        // Step 1: traverse all emails except names, 
        //if we have not seen an email before, put it with its index into map.
        // Otherwise, union the email to its parent index.
        Map<String, Integer> mailToIndex = new HashMap<>();
        for (int i = 0; i < n; i++) {
            for (int j = 1; j < accounts.get(i).size(); j++) {
                String curMail = accounts.get(i).get(j);
                if (mailToIndex.containsKey(curMail)) {
                    int preIndex = mailToIndex.get(curMail);
                    uf.union(preIndex, i);
                }
                else {
                    mailToIndex.put(curMail, i);
                }
            }
        }

        // Step 2: traverse every email list, find the parent of current list index and put all emails into the set list
        // that belongs to key of its parent index
        Map<Integer, Set<String>> disjointSet = new HashMap<>();
        for (int i = 0; i < n; i++) {
            // find parent index of current list index in parent array
            int parentIndex = uf.find(i);
            disjointSet.putIfAbsent(parentIndex, new HashSet<>());

            Set<String> curSet = disjointSet.get(parentIndex);
            for (int j = 1; j < accounts.get(i).size(); j++) {
                curSet.add(accounts.get(i).get(j));
            }
            disjointSet.put(parentIndex, curSet);
        }

        // step 3: traverse ket set of disjoint set group, retrieve all emails from each parent index, and then SORT
        // them, as well as adding the name at index 0 of every sublist
        List<List<String>> result = new ArrayList<>();
        for (int index : disjointSet.keySet()) {
            List<String> curList = new ArrayList<>();
            if (disjointSet.containsKey(index)) {
                curList.addAll(disjointSet.get(index));
            }
            Collections.sort(curList);
            curList.add(0, accounts.get(index).get(0));
            result.add(curList);
        }
        return result;
    }

    class UnionFind {
        int size;
        int[] parent;
        public UnionFind(int size) {
            this.size = size;
            this.parent = new int[size];

            for (int i = 0; i < size; i++) {
                parent[i] = i;
            }
        }

        public void union(int a, int b) {
            parent[find(a)] = parent[find(b)];
        }

        public int find(int x) {
            if (x != parent[x]) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        }
    }
}
```

## 340 Longest Substring with At Most K Distinct Characters

```java
// Input: s = "eceba", k = 2
//Output: 3
//Explanation: T is "ece" which its length is 3.
class Solution {
    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        if (k < 1 || s == null || s.length() == 0) return 0;
        HashMap<Character, Integer> window = new HashMap<>();
        int left = 0; int right = 0; int max = 0;
        while (right < s.length()){
            char r = s.charAt(right);
            right ++;
            window.put(r, window.getOrDefault(r, 0) + 1);
            while (window.size() > k){
                char l = s.charAt(left);
                left ++;
                window.put(l, window.get(l)- 1);
                if(window.get(l) == 0){
                    window.remove(l);
                }
            }
            max = Math.max(max, right - left);
        }
        return max;
    }
}
```

## 249 Group Shifted Strings

```java
//Input: ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"],
//Output: 
//[
//  ["abc","bcd","xyz"],
//  ["az","ba"],
//  ["acef"],
//  ["a","z"]
//]
public class Solution {
    public List<List<String>> groupStrings(String[] strings) {
        List<List<String>> result = new ArrayList<List<String>>();
        Map<String, List<String>> map = new HashMap<String, List<String>>();
        for (String str : strings) {
            int offset = str.charAt(0) - 'a';
            String key = "";
            for (int i = 0; i < str.length(); i++) {
                char c = (char) (str.charAt(i) - offset);
                if (c < 'a') {
                    c += 26;
                }
                key += c;
            }
            if (!map.containsKey(key)) {
                List<String> list = new ArrayList<String>();
                map.put(key, list);
            }
            map.get(key).add(str);
        }
        for (String key : map.keySet()) {
            List<String> list = map.get(key);
            Collections.sort(list);
            result.add(list);
        }
        return result;
    }
}
```

## 670 Maximum Swap

```java
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

## 270 Closest Binary Search Tree Value

```java
Input: root = [4,2,5,1,3], target = 3.714286

    4
   / \
  2   5
 / \
1   3

Output: 4

class Solution {
    public int closestValue(TreeNode root, double target) {
    int ret = root.val;   
    while(root != null){
        if(Math.abs(target - root.val) < Math.abs(target - ret)){
            ret = root.val;
        }      
        root = root.val > target? root.left: root.right;
    }     
    return ret;
  }
}
```

## 543 Diameter of Binary Tree

```java
Example:
Given a binary tree
          1
         / \
        2   3
       / \     
      4   5    
Return 3, which is the length of the path [4,2,1,3] or [5,2,1,3].

class Solution {
    int maxd = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        depth(root);
        return maxd;
    }
    public int depth(TreeNode node){
        if(node==null){
            return 0;
        }
        int Left = depth(node.left);
        int Right = depth(node.right);
        maxd=Math.max(Left+Right,maxd);
        //将每个节点最大直径(左子树深度+右子树深度)当前最大值比较并取大者
        return Math.max(Left,Right)+1;//返回节点深度
        // 
    }
}
// https://leetcode-cn.com/problems/diameter-of-binary-tree
///solution/er-cha-shu-de-zhi-jing-by-leetcode-solution/
```

## 621 Task Scheduler



Given a characters array `tasks`, representing the tasks a CPU needs to do, where each letter represents a different task. Tasks could be done in any order. Each task is done in one unit of time. For each unit of time, the CPU could complete either one task or just be idle.

However, there is a non-negative integer `n` that represents the cooldown period between two **same tasks** \(the same letter in the array\), that is that there must be at least `n` units of time between any two same tasks.

Return _the least number of units of times that the CPU will take to finish all the given tasks_.

**Example 1:**

```text
Input: tasks = ["A","A","A","B","B","B"], n = 2
Output: 8
Explanation: 
A -> B -> idle -> A -> B -> idle -> A -> B
There is at least 2 units of time between any two same tasks.
```

```java

class Solution {
       public int leastInterval(char[] tasks, int n) {
        if (tasks.length <= 1 || n < 1) return tasks.length;
        //步骤1
        int[] counts = new int[26];
        for (int i = 0; i < tasks.length; i++) {
            counts[tasks[i] - 'A']++;
        }
        //步骤2
        Arrays.sort(counts);
        int maxCount = counts[25];
        int retCount = (maxCount - 1) * (n + 1) + 1;
        int i = 24;
        //步骤3
        while (i >= 0 && counts[i] == maxCount) {
            retCount++;
            i--;
        }
        //步骤4
        return Math.max(retCount, tasks.length); 
        // 如果按照最长的排完之后，后面还有剩下的没有排的，
        //比如字符串序列式AAABBBCCCD，然后n=2的话，那拍好就是ABCABCABCD，
        //按照公式计算出来的结果是(3-1)*(3)+1+2=9，但是实际的序列应该是ABCABCABCD，应该是10，
        //所以通过求max来补充掉这个正好全排列但是还有多出去的情况
    }
}
```

## 314 Binary Tree Vertical Order Traversal

![](../.gitbook/assets/verticaltravel-2.jpg)

```java
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

## 689 Maximum Sum of 3 Non-Overlapping Subarrays

```java
//This is a more general DP solution, and it is similar to that buy and sell stock problem.

//dp[i][j] stands for in i th sum, the max non-overlap sum we can have from 0 to j
//id[i][j] stands for in i th sum, the first starting index for that sum.

class Solution {
    public int[] maxSumOfThreeSubarrays(int[] nums, int k) {
        int[][] dp = new int[4][nums.length + 1];
        int sum = 0;
        int[] accu = new int[nums.length + 1];
        for(int i = 0; i < nums.length; i++) {
            sum += nums[i];
            accu[i] = sum;
        }
        int[][] id = new int[4][nums.length + 1];
        int max = 0, inId = 0;
        for(int i = 1; i < 4; i++) {
            for(int j = k-1 ; j < nums.length; j++) {
           int tmpmax = j - k < 0 ? accu[j] : accu[j] - accu[j-k] + dp[i-1][j-k];
                if(j - k >= 0) {
                    dp[i][j] = dp[i][j-1];
                    id[i][j] = id[i][j-1];
                }
                if(j > 0 && tmpmax > dp[i][j-1]) {
                    dp[i][j] = tmpmax;
                    id[i][j] = j-k+1;
                }
            }
        }
        int[] res = new int[3];
        res[2] = id[3][nums.length-1];
        res[1] = id[2][res[2] - 1];
        res[0] = id[1][res[1] - 1];        
        return res;
    }
}
```

## 1026 Maximum Difference Between Node and Ancestor

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

## 986 Interval List Intersections

```java
Input: A = [[0,2],[5,10],[13,23],[24,25]], 
B = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]

class Solution {
    public int[][] intervalIntersection(int[][] A, int[][] B) {
        int i = 0; int j = 0; int k = 0; 
        int al = A.length; int bl = B.length; 
        int[][] res = new int[al + bl][2];
        while (i < a1 && j < b1){
            int a1 = A[i][0]; int a2 = A[i][1];
            int b1 = B[j][0]; int b2 = B[j][1];
            if (b1 <= a2 && b2 >= a1) {
                res[k][0] = Math.max(a1, b1);
                res[k][1] = Math.min(a2,b2);
                k ++;
            }
            if (a2 < b2){
                i ++;
            } else {
                j ++;
            }
        }
        return Arrays.copyOf(res, k);
        // ans.toArray(new int[0][]);
    }
}
```


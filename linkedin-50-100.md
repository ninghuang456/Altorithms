# Linkedin 50\~100

**5 : Longest Palindromic Substring**

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
}java
```

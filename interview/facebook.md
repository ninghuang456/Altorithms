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

        // Step 1: traverse all emails except names, if we have not seen an email before, put it with its index into map.
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


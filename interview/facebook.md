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




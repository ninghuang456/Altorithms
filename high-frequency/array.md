---
description: Array
---

# Array

## Array

1. Two Sum

```
class Solution {
    public int[] twoSum(int[] nums, int target) {
        if (nums == null || nums.length < 2) return new int[0];
        HashMap<Integer,Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
             if (map.containsKey(target - nums[i])){
                 int index = map.get(target - nums[i]);
                 return new int[]{index, i};   
             } else {
                 map.put(nums[i], i);
             }
        }
        return new int[0];
    }
}
```

### 

2.Median of Two Sorted Arrays

```text
public double findMedianSortedArrays(int[] A, int[] B) {
	    int m = A.length, n = B.length;
	    int l = (m + n + 1) / 2;
	    int r = (m + n + 2) / 2;
	    return (getkth(A, 0, B, 0, l) + getkth(A, 0, B, 0, r)) / 2.0;
	}

public double getkth(int[] A, int aStart, int[] B, int bStart, int k) {
	if (aStart > A.length - 1) return B[bStart + k - 1];            
	if (bStart > B.length - 1) return A[aStart + k - 1];                
	if (k == 1) return Math.min(A[aStart], B[bStart]);
	
	int aMid = Integer.MAX_VALUE, bMid = Integer.MAX_VALUE;
	if (aStart + k/2 - 1 < A.length) aMid = A[aStart + k/2 - 1]; 
	if (bStart + k/2 - 1 < B.length) bMid = B[bStart + k/2 - 1];        
	
	if (aMid < bMid) 
	    return getkth(A, aStart + k/2, B, bStart,       k - k/2);// Check: aRight + bLeft 
	else 
	    return getkth(A, aStart,       B, bStart + k/2, k - k/2);// Check: bRight + aLeft
}
```

3,Trapping Rain Water

Given _n_ non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.

![](https://assets.leetcode.com/uploads/2018/10/22/rainwatertrap.png)

```text
class Solution {
    public int trap(int[] A) {
        int a=0;
    int b=A.length-1;
    int max=0;
    int leftmax=0;
    int rightmax=0;
    while(a<=b){
        leftmax=Math.max(leftmax,A[a]);
        rightmax=Math.max(rightmax,A[b]);
        if(leftmax<rightmax){
            max+=(leftmax-A[a]);       // leftmax is smaller than rightmax, so the (leftmax-A[a]) water can be stored Keep track of the                                               //maximum height from both forward directions backward directions, call them leftmax and rightmax.
            a++;
        }
        else{
            max+=(rightmax-A[b]);
            b--;
        }
    }
    return max;
        
    }
}
```

3 Sum

```text
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList();
        if (nums == null || nums.length < 3) return result;
        Arrays.sort(nums);
        HashSet<List<Integer>> uniqueSet = new HashSet<>();
        
        for (int i = 0 ; i < nums.length; i++) {
           int j = i +1;
            int right = nums.length - 1;
            while (j < right){
                int sum = nums[i] + nums[j] + nums[right];
                if (sum == 0){
                    List<Integer> list = new ArrayList();
                    list.add(nums[i]);
                    list.add(nums[j]);
                    list.add(nums[right]);
                    if(uniqueSet.add(list)){
                    result.add(list);
                    }
                    right --;
                    j++;
                } else if (sum > 0) {
                    right --;
                } else {
                    j++;
                }
                 
            }
        }
        return result;
    }
}
```

5 Merge Intervals

Given a collection of intervals, merge all overlapping intervals.

**Example 1:**

```text
Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].

Input: [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.
```

```text
/**
 * Definition for an interval.
 * public class Interval {
 *     int start;
 *     int end;
 *     Interval() { start = 0; end = 0; }
 *     Interval(int s, int e) { start = s; end = e; }
 * }
 */
class Solution {
    public List<Interval> merge(List<Interval> intervals) {
        
          if (intervals.size() <= 1) {
            return intervals;
        }

        //Java8的lambda的comparator，把类匿名类作为参数
        intervals.sort((i1, i2) -> i1.start - i2.start);
        // intervals.sort((i1, i2) -> Integer.compare(i1.start, i2.start));

        List<Interval> result = new ArrayList<>();//OR Linkedlist

        //初始化
        int start = intervals.get(0).start;
        int end = intervals.get(0).end;

        for (Interval interval : intervals) {
            if (interval.start <= end) {//遍历到的interval的start小于之前的end，有重叠,先延伸
                end = Math.max(end, interval.end);
            } else {//没有重叠，将当前的interval加入到结果集
                result.add(new Interval(start, end));
                //加入之前的区间后，把start和end更新为当前遍历到的interval的start和end
                start = interval.start;
                end = interval.end;
            }
        }
        //最后的一个区间没有遍历到的interval比较，需要加入
        result.add(new Interval(start, end));

        return result;
        
    }
}
```

6 Maximum Subarray

Given an integer array `nums`, find the contiguous subarray \(containing at least one number\) which has the largest sum and return its sum.

**Example:**

```text
Input: [-2,1,-3,4,-1,2,1,-5,4],
Output: 6
```

```text
class Solution {
    public int maxSubArray(int[] a) {
         int maxSum = 0, thisSum = 0, max=a[0];
    for(int i=0; i<a.length; i++) {
        if(a[i]>max) max =a[i];
        thisSum += a[i];
        if(thisSum > maxSum)
            maxSum = thisSum;
        else if(thisSum < 0)
            thisSum = 0;
    }
    if (maxSum==0) return max;
    return maxSum;
        
    }
}
```

7 Best Time to Buy and Sell Stock

If you were only permitted to complete at most one transaction \(i.e., buy one and sell one share of the stock\), design an algorithm to find the maximum profit.

```text
class Solution {
    public int maxProfit(int[] prices) {
         if (prices.length == 0) {
			 return 0 ;
		 }		
		 int max = 0 ;
		 int sofarMin = prices[0] ;
	     for (int i = 0 ; i < prices.length ; ++i) {
	    	 if (prices[i] > sofarMin) {
	    		 max = Math.max(max, prices[i] - sofarMin) ;
	    	 } else{
	    		sofarMin = prices[i];  
	    	 }
	     }	     
	    return  max ;   
    }
}
```

8 Product of Array Except Self

 Given an array `nums` of _n_ integers where _n_ &gt; 1,  return an array `output` such that `output[i]` is equal to the product of all the elements of `nums` except `nums[i]`.

```text
Input:  [1,2,3,4]
Output: [24,12,8,6]
```

```text
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int len = nums.length;
        int[] front = new int[len], back = new int[len], result = new int[len];
        //初始化乘积为1
        front[0] = 1;
        back[len - 1] = 1;
        //计算i元素前面所有元素的乘积
        for (int i = 1; i < len; i++) {
            front[i] = front[i - 1] * nums[i - 1];
        }
        //计算i元素后面所有元素的乘积
        for (int i = len - 2; i >= 0; i--) {
            back[i] = back[i + 1] * nums[i + 1];
        }

        //二者相乘
        for (int i = 0; i < len; i++) {
            result[i] = front[i] * back[i];
        }
        return result;
        
    }
}
```

9 Container With Most Water

Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate \(i, ai\). nvertical lines are drawn such that the two endpoints of line i is at \(i, ai\) and \(i, 0\). Find two lines, which together with x-axis forms a container, such that the container contains the most water.

**Note:** You may not slant the container and n is at least 2.

![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/17/question_11.jpg)

```text
class Solution {
    public int maxArea(int[] height) {
        int result = 0, left = 0, right = height.length - 1;
        while (left < right){
            int high = Math.min(height[left], height[right]);
            int width = right - left; // don't mess up right and left.
            int temp = high * width;
            result = Math.max(result, temp);
            if (height[left] < height[right]){
                left ++;
            }else {
                right --;
            } 
        }
        return result;
    }
}
```

10 Search in Rotated Sorted Array

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

\(i.e., `[0,1,2,4,5,6,7]` might become `[4,5,6,7,0,1,2]`\).

You are given a target value to search. If found in the array return its index, otherwise return `-1`.

You may assume no duplicate exists in the array.

```text
class Solution {
    public int search(int[] nums, int target) {
 if (nums == null || nums.length == 0) {
            return -1;
        }
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < nums[right]) {//left到mid顺序乱了，mid到right的部分没有受到rotate影响，这时候先检查mid到right的有序的这部分
                if (target > nums[mid] && target <= nums[right]) {//target在mid到right有序的这部分，需要两个条件加起来才确定是在这个部分
                    left = mid + 1;
                } else {//target在left到mid乱序的这部分
                    right = mid - 1;
                }
            } else {//nums[mid] > nums[right]，mid到right的顺序乱了，而从left到mid是有顺序的，这时候先检查left到mid有序的这部分
                if (target >= nums[left] && target < nums[mid]) {//target在left到mid之间有序的这部分，需要两个条件加起来才确定是在这个部分
                    right = mid - 1;
                } else {//target在mid到right之间乱序的部分
                    left = mid + 1;
                }
            }
        }
        return -1;
    }
}
```

11 Next Permutation

Implement **next permutation**, which rearranges numbers into the lexicographically next greater permutation of numbers.

If such arrangement is not possible, it must rearrange it as the lowest possible order \(ie, sorted in ascending order\).

The replacement must be [**in-place**](http://en.wikipedia.org/wiki/In-place_algorithm) and use only constant extra memory.

Here are some examples. Inputs are in the left-hand column and its corresponding outputs are in the right-hand column.

`1,2,3` → `1,3,2`  
`3,2,1` → `1,2,3`  
`1,1,5` → `1,5,1`

```text
class Solution {
    public void nextPermutation(int[] nums) {
         if (nums == null || nums.length == 0) return;

      int n = nums.length, i = n - 2, j = n - 1;
      while (i >= 0 && nums[i] >= nums[i + 1]) {//从后向前找到一个非增长的元素，位置为i
         i--;
      }
      if (i >= 0) {//全逆序的数组不会进入这个循环，i的位置为-1
         while (nums[j] <= nums[i]) {//从后向前找到第一个比i位置元素大的元素，肯定有这个数字
            j--;
         }
         swap(nums, i, j);
      }
      reverse(nums, i + 1, n - 1);//i位置后面的数组元素进行翻转
   }
   private void swap(int[] nums, int i, int j) {
      int temp = nums[i];
      nums[i] = nums[j];
      nums[j] = temp;
   }
   private void reverse(int[] nums, int i, int j) {
      while (i < j) {
         swap (nums, i, j);
         i++;
         j--;
      }
        
    }
}
```

### 


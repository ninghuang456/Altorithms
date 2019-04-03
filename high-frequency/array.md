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

排列Arrangement是从N个不同元素中取出M个元素，当N == M时，就是全排列Permutation。如果能够知道下一个排列Next Permutation是什么，也就知道了全排列是什么。全排列的问题在算法上没有特别的，但是要理清思路，还得刻意练习才行。

假如排列是{2,3,6,5,4,1}，求下一个排列的基本步骤是这样：  
1\) 先从后往前看，找到第一个不是依次增长的数，记录下位置p。比如现在的例子就应该是3，对应的位置是p==1；现在在p位置的数字跟从后向前的数字进行比较_，_找到第一个比p位置的数大的数，然后两个调换位置，比如例子中的4。把3和4调换位置后得到{2,4,6,5,3,1}。最后把p之后的所有数字倒序，得到{2,4,1,3,5,6}，即是要求的下一个排列；  
2\) 如果从后向前看的时候，上面的数字都是依次增长的，那么说明这是最后一个排列，下一个就是第一个，把所有数字翻转过来即可\(比如{6,5,4,3,2,1}下一个是{1,2,3,4,5,6}\)；

最坏情况需要扫描数组三次，所以时间复杂度是O\(3\*n\)=O\(n\)，空间复杂度是O\(1\)。

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

12 Game of life

According to the [Wikipedia's article](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life): "The **Game of Life**, also known simply as **Life**, is a cellular automaton devised by the British mathematician John Horton Conway in 1970."

Given a board with m by n cells, each cell has an initial state live \(1\) or dead \(0\). Each cell interacts with its [eight neighbors](https://en.wikipedia.org/wiki/Moore_neighborhood) \(horizontal, vertical, diagonal\) using the following four rules \(taken from the above Wikipedia article\):

1. Any live cell with fewer than two live neighbors dies, as if caused by under-population.
2. Any live cell with two or three live neighbors lives on to the next generation.
3. Any live cell with more than three live neighbors dies, as if by over-population..
4. Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.

Write a function to compute the next state \(after one update\) of the board given its current state. The next state is created by applying the above rules simultaneously to every cell in the current state, where births and deaths occur simultaneously.

**Example:**

```text
Input: 
[
  [0,1,0],
  [0,0,1],
  [1,1,1],
  [0,0,0]
]
Output: 
[
  [0,0,0],
  [1,0,1],
  [0,1,1],
  [0,1,0]
]
```

**Follow up**:

1. Could you solve it in-place? Remember that the board needs to be updated at the same time: You cannot update some cells first and then use their updated values to update other cells.
2. In this question, we represent the board using a 2D array. In principle, the board is infinite, which would cause problems when the active area encroaches the border of the array. How would you address these problems?

### **题意和分析**

参考[这里](http://www.cnblogs.com/grandyang/p/4854466.html)和[这里](https://segmentfault.com/a/1190000003819277)，用二维数组来表示细胞，1代表活细胞，0代表死细胞按照题意，每个细胞满足以下条件：

1. 如果活细胞周围八个位置的活细胞数少于两个，则该位置活细胞死亡

2. 如果活细胞周围八个位置有两个或三个活细胞，则该位置活细胞仍然存活

3. 如果活细胞周围八个位置有超过三个活细胞，则该位置活细胞死亡

4. 如果死细胞周围正好有三个活细胞，则该位置死细胞复活

要求计算给定的二维数组的下一个状态，在in-place位置更新，所以就不能新建一个相同大小的数组，只能更新原有数组，但是题目中要求所有的位置必须被同时更新，不能分批更新，但是在循环程序中我们还是一个位置一个位置更新的，那么当一个位置更新了，这个位置成为其他位置的neighbor时，我们怎么知道其未更新的状态呢，我们可以使用状态机转换：

状态0： 死细胞转为死细胞

状态1： 活细胞转为活细胞

状态2： 活细胞转为死细胞

状态3： 死细胞转为活细胞

对所有状态对2取余，那么状态0和2就变成死细胞，状态1和3就是活细胞。因此先对原数组进行逐个扫描，对于每一个位置，扫描其周围八个位置，如果遇到状态1或2，就计数器累加1，扫完8个邻居，如果少于两个活细胞或者大于三个活细胞，而且当前位置是活细胞的话，标记状态2，而如果有三个活细胞且当前是死细胞的话，标记状态3。完成一遍扫描后再对数据扫描一遍，对2取余。

```text
class Solution {
	public void gameOfLife(int[][] board) {
		if (board == null || board.length == 0 || board[0].length == 0) {
			return;
		}
		int m = board.length;
		int n = board[0].length;

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				int lives = livesNeighbors(board, m, n, i, j);
				if (board[i][j] == 1 && lives >= 2 && lives <= 3) {
					board[i][j] = 3;
				}
				if (board[i][j] == 0 && lives == 3) {
					board[i][j] = 2;
				}
			}
		}

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				board[i][j] >>= 1;//除以2
			}
		}
	}
	public int livesNeighbors(int[][] board, int m, int n, int i,int j) {
		int lives = 0;
		for (int x = Math.max(i - 1, 0); x <= Math.min(i + 1, m - 1); x++) {//Math.max和Math.min处理边界问题
			for (int y = Math.max(j - 1, 0); y <= Math.min(j + 1, n - 1); y++) {
				lives += board[x][y] & 1;
			}
		}
		lives -= board[i][j] & 1;
		return lives;
	}
}
```

13 Subarray Sum Equals K

Given an array of integers and an integer **k**, you need to find the total number of continuous subarrays whose sum equals to **k**.

**Example 1:**  


```text
Input:nums = [1,1,1], k = 2
Output: 2
```

**Note:**  


1. The length of the array is in range \[1, 20,000\].
2. The range of numbers in the array is \[-1000, 1000\] and the range of the integer **k** is \[-1e7, 1e7\].

### 题意和分析

这道题是找到所有的子序列的个数，这些子序列的元素的和等于给定的一个target。暴力解法就是两层循环，所有的子序列都计算一遍，找到所有sum\[i, j\] = k的子序列，O\(n^2\)。

我们需要找到sum\[i, j\]，如果我们知道sum\[0, i-1\]和sum\[0, j\]，这样一减就知道sum\[i, j\]是否等于k，换句话说，sum\[j\] - sum\[i\]的话，nums\[i, j\]之间数字的和就是k，比如sum\[j\]跟sum\[i\]一样，那么nums\[i, j\]这段加起来就是0。result += map.get\(sum - k\)这句比较难懂，这个意思是如果sum - k多次等于一个值，那么前面每一个nums\[i\]位置到这里的subarray都算是一个可计入记过的subarray，相当于是需要记得之前有多少个相同的值。

做法就是遍历这个数组，计算current的sum并且把所有的sum都存到一个HashMap里面。举例说明：

Array = {3,4,7,2,-3,1,4,2}，k= 7，如果遇到二者相减（sum - k）等于7，或者sum本身等于7或者7的倍数，subarray的count均+1，（注意黑体字）

* 循环初始map - {{0,1}}， sum - 0， result - 0；
* 第一此循环遇到3，map - {{0, 1}, {3, 1}}；sum - 3；result - 0；sum - k = -4；
* 第二次循环遇到4，map - {{0,1}, {3,1}, {**7,1**}}；sum - 7；result - 1；sum - k = 0；
* 第三次循环遇到7，map - {{0,1}, {3,1}, {**7, 1**}, {**14, 1**}}；sum - 14；result - 2；sum - k = 7；
* 第四次循环遇到2，map - {{0,1}, {3,1}, {7,1}, {14,1}, {16,1}}；sum - 16；result - 2；sum - k = 9；
* 第五次循环遇到-3，map - {{0,1}, {3,1}, {7,1}, {14,1}, {16,1}, {13,1}}；sum - 13；result - 2；sum - k = 6；
* 第六次循环遇到1，map - {{0,1}, {3,1}, {7,1}, {14,**2**}, {16,1}, {13,1}}；sum - 14；result - 3；sum - k = 7；
* 第七次循环遇到4，map - {{0,1}, {3,1}, {7,1}, {14,2}, {16,1}, {13,1}, {18,1}}；sum - 18；result - 3； sum - k = 11；
* 第八次循环遇到2，map - {{0,1}, {3,1}, {7,1}, {14,2}, {16,1}, **{13,1}**, {18,1}, {**20,1**}}，sum - 20；result - 4；sum - k = 13；
* 循环结束

  

Time：O\(n\)；Space：O\(n\)。

### 代码

暴力解法O\(n^2\)

```java
class Solution {
    public int subarraySum(int[] nums, int k) {
        int count = 0;
        int[] sum = new int[nums.length + 1];
        sum[0] = 0;
        for (int i = 1; i <= nums.length; i++) {
            sum[i] = sum[i - 1] + nums[i - 1];
        }
        for (int start = 0; start < nums.length; start++) {
            for (int end = start + 1; end <= nums.length; end++) {
                if (sum[end] - sum[start] == k)
                    count++;
            }
        }
        return count;
    }
}
```

优化成O\(n\)的解法

```java
class Solution {
    public int subarraySum(int[] nums, int k) {
        int sum = 0, result = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);//把以0为key的pair放进去，从sum为0开始

        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];//在之前sum的基础上，遇到当前的元素就更新sum，然后检查hashmap看之前出现过没有
            if (map.containsKey(sum - k)) {
                result += map.get(sum - k);//记得之前有几个这样的值
            }
            map.put(sum, map.getOrDefault(sum
```

14 Spiral Matrix

### **原题概述**

Given a matrix of _m_ x _n_ elements \(_m_ rows, _n_ columns\), return all elements of the matrix in spiral order.

**Example 1:**

```text
Input:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
Output: [1,2,3,6,9,8,7,4,5]
```

**Example 2:**

```text
Input:
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]
```

### **题意和分析**

要求由外层向内层螺旋打印数组，只能一行一列地打印，先往右，再往下，再往左，最后往上，用四个变量来记录打印的位置，下一轮从新的打印位置开始。

### **代码**

```java
class Solution {
   public List<Integer> spiralOrder(int[][] matrix) {

      List<Integer> result = new ArrayList<>();
      if (matrix == null || matrix.length == 0 || matrix[0].length == 0) return result;

      int rowBegin = 0;
      int colEnd = matrix[0].length - 1;
      int colBegin = 0;
      int rowEnd = matrix.length - 1;

      while (rowBegin <= rowEnd && colBegin <= colEnd) {
         //从左向右
         for (int i = colBegin; i <= colEnd; i++) {
            result.add(matrix[rowBegin][i]);
         }
         rowBegin++;

         //从上到下
         for (int i = rowBegin; i <= rowEnd; i++) {
            result.add(matrix[i][colEnd]);
         }
         colEnd--;

         //从右到左
         if (rowBegin <= rowEnd) {//这里检查防止行已经打印完重复打印
            for (int i = colEnd; i >= colBegin; i--) {
               result.add(matrix[rowEnd][i]);
            }
         }
         rowEnd--;

         //从下到上
         if (colBegin <= colEnd) {//这里检查防止列已经打印完重复打印
            for (int i = rowEnd; i >= rowBegin; i--) {
               result.add(matrix[i][colBegin]);
            }
         }
         colBegin++;

      }
      return result;
   }
}
```

15 Word Search

Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

**Example:**

```text
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false.
```

### **题意和分析**

这道题是比较明显的深搜

递归，回溯和DFS的区别

> 递归是一种算法结构，回溯是一种算法思想 
>
> 一个递归就是在函数中调用函数本身来解决问题 回溯就是通过不同的尝试来生成问题的解，有点类似于穷举，但是和穷举不同的是回溯会“剪枝”，意思就是对已经知道错误的结果没必要再枚举接下来的答案了，比如一个有序数列1,2,3,4,5，我要找和为5的所有集合，从前往后搜索我选了1，然后2，然后选3 的时候发现和已经大于预期，那么4,5肯定也不行，这就是一种对搜索过程的优化。
>
> 深度优先搜索（DFS）对于某一种数据结构来说，一般是树（搜索树是起记录路径和状态判断的作用），对于回溯和DFS，其主要的区别是，回溯法在求解过程中不保留完整的树结构，而深度优先搜索则记下完整的搜索树。
>
> 为了减少存储空间，在深度优先搜索中，用标志的方法记录访问过的状态，这种处理方法使得深度优先搜索法与回溯法没什么区别了。

如同上面的比较，DFS有两种经典的做法，一是用跟原本二维数组同等大小的数组来记录是否visited过，其中元素为boolean， 如果二维数组board的当前字符和目标字符串word对应的字符相等，则对其上下左右四个邻字符分别调用DFS的递归函数，只要有一个返回true，那么就表示可以找到对应的字符串，否则就不能找到；第二是对第一种做法空间上的优化，每次用一个char来记录当前二维数组里面的char，在递归调用前用一个特殊的字符，比如‘\#’，来代替当前字符说明已经检查过了，然后再递归调用后再改回来方便下次检查。

### **代码**

```java
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
```

优化空间

```java
class Solution {
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0 || board[0].length == 0) {
            return false;
        }
        int m = board.length, n = board[0].length;
        int index = 0;//字符串的索引
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (dfs(board, word, index, i, j)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean dfs(char[][] board, String word, int index, int i, int j) {
        if (index == word.length()) {//找完了
            return true;
        }
        int m = board.length, n = board[0].length;
        if (i < 0 || j < 0 || i >= m || j >= n
                || board[i][j] != word.charAt(index)) {//两个字符不相等
            return false;
        }
        char temp = board[i][j];//临时存一下当前的字符
        board[i][j] = '#';
        boolean result = (dfs(board, word, index + 1, i - 1, j)//左
                || dfs(board, word, index + 1, i + 1, j)//右
                || dfs(board, word, index + 1, i, j - 1)//上
                || dfs(board, word, index + 1, i, j + 1));//下
        board[i][j] = temp;//修改回来
        return result;
    }
}
```




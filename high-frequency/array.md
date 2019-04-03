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

16 Task Scheduler

Given a char array representing tasks CPU need to do. It contains capital letters A to Z where different letters represent different tasks.Tasks could be done without original order. Each task could be done in one interval. For each interval, CPU could finish one task or just be idle.

However, there is a non-negative cooling interval **n** that means between two **same tasks**, there must be at least n intervals that CPU are doing different tasks or just be idle.

You need to return the **least** number of intervals the CPU will take to finish all the given tasks.

**Example 1:**

```text
Input: tasks = ["A","A","A","B","B","B"], n = 2
Output: 8
Explanation: A -> B -> idle -> A -> B -> idle -> A -> B.
```

**Note:**

1. The number of tasks is in the range \[1, 10000\].
2. The integer n is in the range \[0, 100\].

#### 题意和分析

安排CPU的任务，规定在两个相同任务之间至少隔n个时间点，求时间总长。思路比较多，这里的做法是建立一个优先队列，然后把统计好的个数都存入优先队列中，那么大的次数会在队列的前面。这题还是要分块，每块能装n+1个任务，装任务是从优先队列中取，每个任务取一个，装到一个临时数组中，然后遍历取出的任务，对于每个任务，将其哈希表映射的次数减1，如果减1后，次数仍大于0，则将此任务次数再次排入队列中，遍历完后如果队列不为空，说明该块全部被填满，则结果加上n+1。我们之前在队列中取任务是用个变量count来记录取出任务的个数，我们想取出n+1个，如果队列中任务数少于n+1个，那就用count来记录真实取出的个数，当队列为空时，就加上count的个数。

#### 代码

```java
class Solution {
    public int leastInterval(char[] tasks, int n) {
        Map<Character, Integer> counts = new HashMap<>();
        for (char t : tasks) {
            counts.put(t, counts.getOrDefault(t, 0) + 1);
        }
        PriorityQueue<Integer> pq = new PriorityQueue<Integer>((a, b) -> b - a);
        pq.addAll(counts.values());

        int alltime = 0;
        int cycle = n + 1;
        while (!pq.isEmpty()) {
            int worktime = 0;
            List<Integer> tmp = new ArrayList<Integer>();
            for (int i = 0; i < cycle; i++) {
                if (!pq.isEmpty()) {
                    tmp.add(pq.poll());
                    worktime++;
                }
            }
            for (int cnt : tmp) {
                if (--cnt > 0) {
                    pq.offer(cnt);
                }
            }
            alltime += !pq.isEmpty() ? cycle : worktime;
        }

        return alltime;
    }
}
```

17 Maximal Rectangle

Given a 2D binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

**Example:**

```text
Input:
[
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
Output: 6
```

### 题意和分析

The DP solution proceeds row by row, starting from the first row. Let the maximal rectangle area at row i and column j be computed by \[right\(i,j\) - left\(i,j\)\]\*height\(i,j\).

All the 3 variables left, right, and height can be determined by the information from previous row, and also information from the current row. So it can be regarded as a DP solution. The transition equations are:

> left\(i,j\) = max\(left\(i-1,j\), cur\_left\), cur\_left can be determined from the current row

> right\(i,j\) = min\(right\(i-1,j\), cur\_right\), cur\_right can be determined from the current row

> height\(i,j\) = height\(i-1,j\) + 1, if matrix\[i\]\[j\]=='1';

> height\(i,j\) = 0, if matrix\[i\]\[j\]=='0'

### 代码

```java
class Solution {
    public int maximalRectangle(char[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0) return 0;
        int m = matrix.length, n = matrix[0].length, maxArea = 0;
        int[] left = new int[n];
        int[] right = new int[n];
        int[] height = new int[n];
        Arrays.fill(right, n - 1);
        for (int i = 0; i < m; i++) {
            int rB = n - 1;
            for (int j = n - 1; j >= 0; j--) {
                if (matrix[i][j] == '1') {
                    right[j] = Math.min(right[j], rB);
                } else {
                    right[j] = n - 1;
                    rB = j - 1;
                }
            }
            int lB = 0;
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1') {
                    left[j] = Math.max(left[j], lB);
                    height[j]++;
                    maxArea = Math.max(maxArea, height[j] * (right[j] - left[j] + 1));
                } else {
                    height[j] = 0;
                    left[j] = 0;
                    lB = j + 1;
                }
            }
        }
        return maxArea;
    }
}
```

18 First Missing Positive

Given an unsorted integer array, find the smallest missing positive integer.

**Example 1:**

```text
Input: [1,2,0]
Output: 3
```

**Example 2:**

```text
Input: [3,4,-1,1]
Output: 2
```

**Example 3:**

```text
Input: [7,8,9,11,12]
Output: 1
```

**Note:**

Your algorithm should run in _O_\(_n_\) time and uses constant extra space.

### **题意和分析**

给一个数组，返回第一个缺失的正数，要求线性时间复杂度O\(n\)和常量空间O\(1\)，因此一般的排序方法是不能用的，另外用空间也有要求所以利用额外空间例如HashMap和HashSet也不能用了；只能in-place来做，遍历数组，把1放到nums\[0\]处，把2放到nums\[1\]处，如果nums\[i\] &gt; 0（**负数和0不用管**），同时nums\[i\]为整数且不大于n（**因为缺失的第一个正数值最大就是数组的长度n，不可能超过**），同时nums\[i\]不等于nums\[nums\[i\] - 1\]的话（**桶排序的思想，对应的数字应该放在对应的位置上**），则交换nums\[i\]和nums\[nums\[i\] - 1\]的位置；然后再遍历一遍，遇到nums\[i\] != i + 1即为第一个缺失的正数。

### **代码**

```java
class Solution {
    public int firstMissingPositive(int[] nums) {
        if (nums == null || nums.length == 0) return 1;
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            while (nums[i] > 0 && nums[i] <= n && nums[i] != nums[nums[i] - 1]) {
                int temp = nums[nums[i] - 1];
                nums[nums[i] - 1] = nums[i];
                nums[i] = temp;
            }
        }
        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) return i + 1;
        }

        //从1开始连续出现
        return n + 1;
    }
}
```

19 Merge Sorted Array

Given two sorted integer arrays _nums1_ and _nums2_, merge _nums2_ into _nums1_ as one sorted array.

**Note:**

* The number of elements initialized in _nums1_ and _nums2_ are _m_ and _n_ respectively.
* You may assume that _nums1_ has enough space \(size that is greater or equal to _m_ + _n_\) to hold additional elements from _nums2_.

**Example:**

```text
Input:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

Output: [1,2,2,3,5,6]
```

### 题意与分析

两个排序好的数组nums1和nums2，按照顺序全部元素都放到第一个数组当中，保证装得下所以不用担心nums1数组空间的问题。大致的思路就是在nums1中最后一个位置开始（通过原先两个数组的参数代表有效长度，相加得来），把大的数填在后面，这样就不会覆盖nums1前面的数字了，注意别越界就行。

Time：O\(m + n\)；Space：O\(1\)；

这道题很有可能和[Merge Two Sorted Lists](https://guilindev.gitbook.io/interview/leetcode/ji-chu-shu-ju-jie-gou-zai-suan-fa-zhong-de-ying-yong/linkedlist/he-bing-liang-ge-you-xu-lie-biao)一起问。

### 代码

```java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int length = m + n;

        while (n > 0) { //表示nums2里面的元素还没有加完
            //从最后一位开始检查，每次循环前移一位
            length--;
            if (m == 0 || nums1[m - 1] < nums2[n - 1]) {//m=0表示只剩nums2的元素了，直接加入就好
                n--;//从nums2最后一位开始
                nums1[length] = nums2[n];
            } else {//m != 0 &&　nums1[m - 1] > nums2[n - 1]
                m--;//从nums1最后一位开始
                nums1[length] = nums1[m];
            }
        }
        //这样写可以短一点: nums1[--length] = (m == 0 || nums[m - 1] < nums[n - 1]) ? nums2[--n] : nums1[--m];
    }
}
```

20 Move Zeroes

### 原题概述

Given an array `nums`, write a function to move all `0`'s to the end of it while maintaining the relative order of the non-zero elements.

**Example:**

```text
Input: [0,1,0,3,12]
Output: [1,3,12,0,0]
```

**Note**:

1. You must do this **in-place** without making a copy of the array.
2. Minimize the total number of operations.

### 题意和分析

给一个数组，把其中的0挪到最后面去，非0元素的相对顺序不能变，不能另外开一个数组；使用两个指针，从0位置开始查，找到不为0的元素后，与另外一个指针交换值，直到末尾。

### 代码

```java
class Solution {
    public void moveZeroes(int[] nums) {
        for (int left = 0, right = 0; right < nums.length; right++) {
            if (nums[right] != 0) {//把非0的元素全部换到前面来
                int temp = nums[right];
                nums[right] = nums[left];
                nums[left] = temp;
                left++;//挪动指针从非0元素到下一位
            }
        }
    }
}
```

21 Rotate Image

You are given an _n_ x _n_ 2D matrix representing an image.

Rotate the image by 90 degrees \(clockwise\).

**Note:**

You have to rotate the image [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm), which means you have to modify the input 2D matrix directly. **DO NOT** allocate another 2D matrix and do the rotation.

**Example 1:**

```text
Given input matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

rotate the input matrix in-place such that it becomes:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]
```

**Example 2:**

```text
Given input matrix =
[
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
], 

rotate the input matrix in-place such that it becomes:
[
  [15,13, 2, 5],
  [14, 3, 4, 1],
  [12, 6, 8, 9],
  [16, 7,10,11]
]
```

### 题意和分析

计算机里图片的本质是矩阵，旋转矩阵即是旋转图片，有很多方法可以旋转矩阵，我自己比较好理解的两种办法是：

1）首先对原数组取其转置矩阵（行列互换），然后把每行的数字翻转可得到结果，如下所示\(其中蓝色数字表示翻转轴\)：

1  2  3　　　 　　 1  4  7　　　　　  7  4  1

4  5  6　　--&gt;　　 2  5  8　　 --&gt;  　  8  5  2　　

7  8  9 　　　 　　3  6  9　　　　      9  6  3

2）首先以从对角线为轴翻转，然后再以x轴中线上下翻转即可得到结果：

1  2  3　　　 　　 9  6  3　　　　　  7  4  1

4  5  6　　--&gt;　　 8  5  2　　 --&gt;   　 8  5  2　　

7  8  9 　　　 　　7  4  1　　　　　  9  6  3

3）每次循环换四个数字：

1  2  3                 7  2  1                  7  4  1

4  5  6      --&gt;      4  5  6　　 --&gt;  　 8  5  2　　

7  8  9                 9  8  3　　　　　 9  6  3

### 代码

转置矩阵的办法

```java
class Solution {
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {//j = i不用重复转置
                //转换为转置矩阵transport matrix
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
        //逐行将元素翻转
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n/2; j++) {//注意这里是j < n/2，没有=
                int temp = matrix[i][j];
                matrix[i][j] = matrix[i][n - j - 1];
                matrix[i][n - j - 1] = temp;
            }
        }
    }
}
```

对角线翻转的办法

```java
class Solution {
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        //以对角线为轴翻转
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - j][n - 1 - i];
                matrix[n - 1 - j][n - 1 - i] = temp;
            }
        }
        //以x轴中线上下翻转
        for (int i = 0; i < n / 2; i++) {
            for (int j = 0; j < n; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - i][j];
                matrix[n - 1 - i][j] = temp;
            }
        }
    }
}
```

22 Combination Sum

Given _a_ **set** of candidate numbers \(`candidates`\) **\(without duplicates\)** and a target number \(`target`\), find all unique combinations in `candidates` where the candidate numbers sums to `target`.

The **same** repeated number may be chosen from `candidates` unlimited number of times.

**Note:**

* All numbers \(including `target`\) will be positive integers.
* The solution set must not contain duplicate combinations.

**Example 1:**

```text
Input: candidates = [2,3,6,7], target = 7,
A solution set is:
[
  [7],
  [2,2,3]
]
```

**Example 2:**

```text
Input: candidates = [2,3,5], target = 8,
A solution set is:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
```

### **题意和分析**

给一个没有重复元素的数组，找出里面的元素加起来等于target的所有组合，原数组的元素可以利用多次。这种求所有组合的情况通常都是另外写一个方法来做递归求得（[这里是这类型题的总结](https://leetcode.com/problems/combination-sum/discuss/16502/A-general-approach-to-backtracking-questions-in-Java-%28Subsets-Permutations-Combination-Sum-Palindrome-Partitioning%29)）。

### **代码**

```java
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (candidates == null || candidates.length == 0) {
            return result;
        }
        List<Integer> oneRecord = new ArrayList<>();
        Arrays.sort(candidates);
        backtrack(candidates, target, 0, oneRecord, result);
        return result;
    }

    private void backtrack(int[] candidates, int remian, int index, List<Integer> oneRecord, List<List<Integer>> result) {
        if (remian < 0 ) {//组合不合适
            return;
        } else if (remian == 0) {//找到合适的组合
            result.add(new ArrayList<>(oneRecord));
        } else {
            for (int i = index; i < candidates.length; i++) {
                oneRecord.add(candidates[i]);
                backtrack(candidates, remian - candidates[i], i, oneRecord, result);
                oneRecord.remove(oneRecord.size() - 1);//按照索引移除所有元素
            }
        }
    }
}
```

23 Max Area of Island

Given a non-empty 2D array `grid` of 0's and 1's, an **island** is a group of `1`'s \(representing land\) connected 4-directionally \(horizontal or vertical.\) You may assume all four edges of the grid are surrounded by water.

Find the maximum area of an island in the given 2D array. \(If there is no island, the maximum area is 0.\)

**Example 1:**

```text
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]
```

Given the above grid, return `6`. Note the answer is not 11, because the island must be connected 4-directionally.

**Example 2:**

```text
[[0,0,0,0,0,0,0,0]]
```

Given the above grid, return `0`.

**Note:** The length of each dimension in the given `grid` does not exceed 50.

### 题意和分析

找二维数组中最大的岛的面积，岛为1，是典型DFS的实现。

### 代码

```java
class Solution {
   public int maxAreaOfIsland(int[][] grid) {
      int maxArea = 0;
      for (int i = 0; i < grid.length; i++) {
         for (int j = 0; j < grid[0].length; j++) {
            if (grid[i][j] == 1) {
               maxArea = Math.max(maxArea, dfs(grid, i, j));
            }
         }
      }

      return maxArea;
   }
   private int dfs(int[][] grid, int i, int j) {
      if (i >= 0 && i < grid.length && j >= 0 && j < grid[0].length && grid[i][j] == 1) {
         grid[i][j] = 0;//计算过了
         return 1 + dfs(grid, i - 1, j) + dfs(grid, i + 1, j) + dfs(grid, i, j - 1) + dfs(grid, i, j + 1);//记得前面的1是本身，需要加上
      }
      return 0;//当前元素为0，不是island，直接返回
   }
}
```

24 Word Ladder II

Given two words \(_beginWord_ and _endWord_\), and a dictionary's word list, find all shortest transformation sequence\(s\) from _beginWord_ to _endWord_, such that:

1. Only one letter can be changed at a time
2. Each transformed word must exist in the word list. Note that _beginWord_ is _not_ a transformed word.

**Note:**

* Return an empty list if there is no such transformation sequence.
* All words have the same length.
* All words contain only lowercase alphabetic characters.
* You may assume no duplicates in the word list.
* You may assume _beginWord_ and _endWord_ are non-empty and are not the same.

**Example 1:**

```text
Input:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

Output:
[
  ["hit","hot","dot","dog","cog"],
  ["hit","hot","lot","log","cog"]
]
```

**Example 2:**

```text
Input:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]

Output: []

Explanation: The endWord "cog" is not in wordList, therefore no possible transformation.
```

```text
The basic idea is:

1). Use BFS to find the shortest distance between start and end, tracing the distance of crossing nodes from start node to end node, and store node's next level neighbors to HashMap;

2). Use DFS to output paths with the same distance as the shortest distance from distance HashMap: compare if the distance of the next level node equals the distance of the current node + 1.

public List<List<String>> findLadders(String start, String end, List<String> wordList) {
   HashSet<String> dict = new HashSet<String>(wordList);
   List<List<String>> res = new ArrayList<List<String>>();         
   HashMap<String, ArrayList<String>> nodeNeighbors = new HashMap<String, ArrayList<String>>();// Neighbors for every node
   HashMap<String, Integer> distance = new HashMap<String, Integer>();// Distance of every node from the start node
   ArrayList<String> solution = new ArrayList<String>();

   dict.add(start);          
   bfs(start, end, dict, nodeNeighbors, distance);                 
   dfs(start, end, dict, nodeNeighbors, distance, solution, res);   
   return res;
}

// BFS: Trace every node's distance from the start node (level by level).
private void bfs(String start, String end, Set<String> dict, HashMap<String, ArrayList<String>> nodeNeighbors, HashMap<String, Integer> distance) {
  for (String str : dict)
      nodeNeighbors.put(str, new ArrayList<String>());

  Queue<String> queue = new LinkedList<String>();
  queue.offer(start);
  distance.put(start, 0);

  while (!queue.isEmpty()) {
      int count = queue.size();
      boolean foundEnd = false;
      for (int i = 0; i < count; i++) {
          String cur = queue.poll();
          int curDistance = distance.get(cur);                
          ArrayList<String> neighbors = getNeighbors(cur, dict);

          for (String neighbor : neighbors) {
              nodeNeighbors.get(cur).add(neighbor);
              if (!distance.containsKey(neighbor)) {// Check if visited
                  distance.put(neighbor, curDistance + 1);
                  if (end.equals(neighbor))// Found the shortest path
                      foundEnd = true;
                  else
                      queue.offer(neighbor);
                  }
              }
          }

          if (foundEnd)
              break;
      }
  }

// Find all next level nodes.    
private ArrayList<String> getNeighbors(String node, Set<String> dict) {
  ArrayList<String> res = new ArrayList<String>();
  char chs[] = node.toCharArray();

  for (char ch ='a'; ch <= 'z'; ch++) {
      for (int i = 0; i < chs.length; i++) {
          if (chs[i] == ch) continue;
          char old_ch = chs[i];
          chs[i] = ch;
          if (dict.contains(String.valueOf(chs))) {
              res.add(String.valueOf(chs));
          }
          chs[i] = old_ch;
      }

  }
  return res;
}

// DFS: output all paths with the shortest distance.
private void dfs(String cur, String end, Set<String> dict, HashMap<String, ArrayList<String>> nodeNeighbors, HashMap<String, Integer> distance, ArrayList<String> solution, List<List<String>> res) {
    solution.add(cur);
    if (end.equals(cur)) {
       res.add(new ArrayList<String>(solution));
    } else {
       for (String next : nodeNeighbors.get(cur)) {            
            if (distance.get(next) == distance.get(cur) + 1) {
                 dfs(next, end, dict, nodeNeighbors, distance, solution, res);
            }
        }
    }           
   solution.remove(solution.size() - 1);
}
```

25 Longest Consecutive Sequence

Given an unsorted array of integers, find the length of the longest consecutive elements sequence.

Your algorithm should run in O\(_n_\) complexity.

**Example:**

```text
Input: [100, 4, 200, 1, 3, 2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
```

### **题意和分析**

没有排序的数组里面寻找最长的子序列，要求时间复杂度是O\(n\)，没有空间复杂度的要求，于是可以用一个HashSet，把数组里面所有的元素放入到set里面，然后遍历数组，对每个元素都进行移除操作，同时用两个指针prev和next求出当前元素的构成连续数列的前面和后面一个数，继续检查prev和next是否在set中存在，如果存在就继续移除，最后用next - prev - 1（因为两个指针指向的元素在set中不存在的时候才停止移除，所以有-1），对每个元素都进行这样的操作后求出连续序列最大的。

 也可以采用HashMap来做，刚开始map为空，然后遍历所有数组中的元素，如果该数字不在map中，那么分别检查前后两个数字是否在map中，如果在，则返回其哈希表中映射值，若不在，则返回0，将prev+next+1作为当前数字的映射，并更新result结果，然后更新num-left和num-right的映射值。

### **代码**

HashSet

```java
class Solution {
   public int longestConsecutive(int[] nums) {
      if (nums == null || nums.length == 0) return 0;

      HashSet<Integer> set = new HashSet<>();
      int result = 0;

      //将数组里的所有元素放到HashSet里面
      for (int num : nums) set.add(num);

      for (int num : nums) {
         if (set.remove(num)) {//Java的remove方法是有返回值的，同样add也有
            int prev = num - 1, next = num + 1;
            while (set.remove(prev)) prev--;
            while (set.remove(next)) next++;

            result = Math.max(result, next - prev - 1);
         }
      }
      return result;
   }
}
```

HashMap

```java
class Solution {
   public int longestConsecutive(int[] nums) {
      if (nums == null || nums.length == 0) return 0;

      HashMap<Integer, Integer> map = new HashMap<>();
      int result = 0;
      for (int num : nums) {
         if (!map.containsKey(num)) {
            //注意这里的prev和next是元素在HashMap中的索引值
            int prev = map.containsKey(num - 1) ? map.get(num - 1) : 0;
            int next = map.containsKey(num + 1) ? map.get(num + 1) : 0;

            int sum = prev + next + 1;
            map.put(num, sum);

            result = Math.max(result, sum);

            map.put(num - prev, sum);
            map.put(num + next, sum);
         }
      }

      return result;
   }
}
```

26.  3 Sum Closest

Given an array `nums` of _n_ integers and an integer `target`, find three integers in `nums` such that the sum is closest to `target`. Return the sum of the three integers. You may assume that each input would have exactly one solution.

**Example:**

```text
Given array nums = [-1, 2, 1, -4], and target = 1.

The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
```

### 题意和分析

找到数组中的一个triplet三个数的和距离target的差值相比其它triplets来说是最小的，并返回这个triplet的和，因为不是返回所有可能的triplets，所以不需要去重了。这是上一道题的延伸，思路类似，先排序，然后确定一个index，剩下的两个indices两头扫描。

 同样，时间复杂度: O\(nlogn\) + O\(n^2\) = O\(n^2\)；空间复杂度O\(n\)。

### 代码

```java
class Solution {
    public int threeSumClosest(int[] nums, int target) {
        if (nums == null || nums.length < 3) {
            return 0;
        }
        
        int result = 0; 
        int min = Integer.MAX_VALUE;
        
        Arrays.sort(nums);
        
        for (int first = 0; first <= nums.length - 3; first++) {
            int second = first + 1, third = nums.length - 1;
            while (second < third) {//循环一轮找到最小的min
                int sum = nums[first] + nums[second] + nums[third];
                if (Math.abs(sum - target) < min) {
                    min = Math.abs(sum - target);
                    result = sum;
                }
                
                if (sum == target) {//等于target那就是差距最小了，直接返回
                    return sum;
                } else if (sum < target) {
                    second++;
                } else {
                    third--;
                }
            }
        }
        return result;
    }
}
```

27. Construct Binary Tree from Preorder and Inorder Traversal

Given preorder and inorder traversal of a tree, construct the binary tree.

**Note:**  
You may assume that duplicates do not exist in the tree.

For example, given

```text
preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
```

Return the following binary tree:

```text
    3
   / \
  9  20
    /  \
   15   7
```

### 题意和分析

这道题用先序和中序来建立二叉树，先序的顺序第一个肯定是root，所以二叉树的根结点可以确定，由于题目中说了没有相同的元素，所以利用先序的根我们可以找到这个根在中序的位置，并且在中序的数组中根结点为中心拆分成左右两部分，然后又用我们熟悉的递归调用就可以重建二叉树了

### 代码

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return buildTree(preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1);
    }

    private TreeNode buildTree(int[] preorder, int pLeft, int pRight, int[] inorder, int iLeft, int iRight) {
        if (pLeft > pRight || iLeft > iRight) {
            return null;
        }
        int i = 0;
        for (i = iLeft; i <= iRight; i++) {
            if (preorder[pLeft] == inorder[i]) {
                break;
            }
        }

        TreeNode cur = new TreeNode(preorder[pLeft]);
        cur.left = buildTree(preorder, pLeft + 1, pLeft + i - iLeft, inorder, iLeft, i - 1);
        cur.right = buildTree(preorder, pLeft + i - iLeft +1, pRight, inorder, i + 1, iRight);

        return cur;
    }
}
```

28. Jump Game

Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.

**Example 1:**

```text
Input: [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
```

**Example 2:**

```text
Input: [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum
             jump length is 0, which makes it impossible to reach the last index.
```

### **题意和分析**

这道题首先可以用DP， 维护一个一位数组dp，其中dp\[i\]表示达到i位置时剩余的步数，到达当前位置跟上一个位置（不是前一个位置）的剩余步数和数字（能达到的最远位置）有关，下一个位置的剩余步数（dp值）就等于当前的这个较大值减去1，因为需要花一个跳力到达下一个位置，所以状态转移方程：dp\[i\] = max\(dp\[i - 1\], nums\[i - 1\]\) - 1，如果当某一个时刻dp数组的值为负了，说明无法抵达当前位置，则直接返回false，最后判断**数**组最后一位是否为非负数即可知道是否能抵达该位置。

Greedy的做法会更优，因为其实没有必要用维护一维数组的方式来对每一步的剩余步数进行关注，只知道是否能到达末尾就行了，只需维护一个变量reach，来记录当前步最远能到达的位置坐标，初始为0，遍历整个数组，如果当前的坐标大于reach或者reach已经到达最后一个位置或超出，就跳出循环；否则就更更新reach为当前reach的值和i + nums\[i\]中的较大值。

### **代码**

DP

```java
class Solution {
    public boolean canJump(int[] nums) {
        int len = nums.length;
        int [] dp = new int[len];
        Arrays.fill(dp, 0);

        for (int i = 1; i < len; i++) {
            dp[i] = Math.max(dp[i - 1], nums[i - 1]) - 1;
            if (dp[i] < 0) {
                return false;
            }
        }
        return dp[len - 1] >= 0;
    }
}
```

Greedy

```java
class Solution {
    public boolean canJump(int[] nums) {
        int len = nums.length;
        int reach = 0;
        for (int i = 0; i < len; i++) {
            if (i > reach || reach >= len - 1) {
                break;
            }
            reach = Math.max(reach, i + nums[i]);
        }
        return reach >= len - 1;
    }
}
```

29. Sub set

Given a set of **distinct** integers, _nums_, return all possible subsets \(the power set\).

**Note:** The solution set must not contain duplicate subsets.

**Example:**

```text
Input: nums = [1,2,3]
Output:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]

```

```text
backtracking:
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);//需要排序，这样才知道哪些数字已经用过
        backtrack(result, new ArrayList<>(), nums, 0);//初始传入0作为起始点
        return result;
    }
    private void backtrack(List<List<Integer>> result, List<Integer> tempList, int[] nums, int start) {
        result.add(new ArrayList<>(tempList));//注意添加的方式，是新建一个ArrayList对象
        for (int i = start; i < nums.length; i++) {
            tempList.add(nums[i]);
            backtrack(result, tempList, nums, i + 1);
            tempList.remove(tempList.size() - 1);//每一轮需要清空
        }
    }
}
```


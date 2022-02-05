# Array2

31\. Jump Game II

Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Your goal is to reach the last index in the minimum number of jumps.

**Example:**

```
Input: [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2.
    Jump 1 step from index 0 to 1, then 3 steps to the last index.
```

```
class Solution {
    public int jump(int[] nums) {
        int step = 0;
        int last = 0;//上一步能到达的最远距离
        int curr = 0;//当前结点最远能覆盖的距离
        for (int i = 0; i < nums.length; i++) {
            if (i > last) {//上一步不能到达，得再跳一次了
                last = curr;
                step++;
            }
            curr = Math.max(curr, i + nums[i]);
        }
        return step;
    }
}
```

32 Two Sum II

Given an array of integers that is already _**sorted in ascending order**_, find two numbers such that they add up to a specific target number.

The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2.

**Note:**

* Your returned answers (both index1 and index2) are not zero-based.
* You may assume that each input would have _exactly_ one solution and you may not use the _same_ element twice.

**Example:**

```
Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore index1 = 1, index2 = 2.
```

### 题意和分析

是上一道题1-2Sum的变种，按照第二种办法先排序再两个指针找的情况，因为input是sorted的数组，所以连排序都不用了，而且也不用一个额外的数组来存排序之前的indices，相比之下是更简单了。

Time：O(n)，Space：O(1)。

的当然，跟上一道题一样，也可以用HashMap来解，无论输入的array是否sorted,

Time: O(n),  Space: O(n)

### 代码

```java
class Solution {
    public int[] twoSum(int[] numbers, int target) {
        int[] result = new int[2];
        
        int left = 0, right = numbers.length - 1;
        while (left < right) {
            if (numbers[left] + numbers[right] < target) {
                left++;
            } else if (numbers[left] + numbers[right] > target) {
                right--;
            } else {
                break;//遇到合适的pair直接break，省点时间，有唯一解的情况
            }
        }
        
        //因为两个indices不是zero-based，所以+1，从1开始数
        result[0] = left + 1;
        result[1] = right + 1;
        
        return result;
    }
}
```

33 Four Sums

Given an array `nums` of _n_ integers and an integer `target`, are there elements _a_, _b_, _c_, and _d_ in `nums` such that _a_ + _b_ + _c_ + _d_ = `target`? Find all unique quadruplets in the array which gives the sum of `target`.

**Note:**

The solution set must not contain duplicate quadruplets.

**Example:**

```
Given array nums = [1, 0, -1, 0, -2, 2], and target = 0.

A solution set is:
[
  [-1,  0, 0, 1],
  [-2, -1, 1, 2],
  [-2,  0, 0, 2]
]
```

### 题意和分析

找到Quadruplets的四个数，加起来等于target的值，思路跟3Sum一样，再在外面套一层循环，最后的解决方案不能有重复，所以依然得去重。

时间复杂度: O(nlogn) + O(n^3)  = O(n^3)；空间复杂度O(n)。

### 代码

```java
class Solution {
    public List<List<Integer>> fourSum(int[] nums, int target) {
        ArrayList<List<Integer>> result = new ArrayList<List<Integer>>();
        HashSet<ArrayList<Integer>> noDuplicateQuad = new HashSet<ArrayList<Integer>>();
        
        Arrays.sort(nums);
        
        for (int first = 0; first <= nums.length - 4; first++) {
            for (int second = first + 1; second <= nums.length - 3; second++) {
                int third = second + 1;
                int fourth = nums.length - 1;
                
                while (third < fourth) {
                    int sum = nums[first] + nums[second] + nums[third] + nums[fourth];
                    if (sum < target) {
                        third++;
                    } else if (sum > target) {
                        fourth--;
                    } else { //找到了一个合适的Quadruplet
                        ArrayList<Integer> oneQuadruplet = new ArrayList<>();
                        oneQuadruplet.add(nums[first]);
                        oneQuadruplet.add(nums[second]);
                        oneQuadruplet.add(nums[third]);
                        oneQuadruplet.add(nums[fourth]);
                        
                        if (!noDuplicateQuad.contains(oneQuadruplet)) {
                            noDuplicateQuad.add(oneQuadruplet);
                            result.add(oneQuadruplet);
                        }
                        
                        //这里是两个indices同时移动，因为排过序了后找到的是已经等于target了，所以只移动一个index的话是不会再找到非重复的Quadruplet的
                        third++;
                        fourth--;
                    }
                }
            }
        }
        return result;
    }
}
```

34.Missing Number

Given an array containing n distinct numbers taken from `0, 1, 2, ..., n`, find the one that is missing from the array.

**Example 1:**

```
Input: [3,0,1]
Output: 2
```

**Example 2:**

```
Input: [9,6,4,2,3,5,7,0,1]
Output: 8
```

**Note**:\
Your algorithm should run in linear runtime complexity. Could you implement it using only constant extra space complexity?

### 题意和分析

给一个未排序的数组，里面有n个数字，范围从0到n(本来有n+1个数字)，找到缺失的那个数，要求线型时间复杂度和常数空间。

我们可以用等差数列的求和公司n \* (n - 1) / 2求得一个值Sum，然后遍历整个数组，将每个元素值从Sum里面减掉，最后剩下的数字就是缺失的数字。

这里我们还是用位操作异或XOR来做，因为0到n之间少了一个数，因为 a^b^b =a，异或两次等于原来的数，将这个少了一个数的数组和0到n之间完整的数组异或XOR一下，那么相同的数字都变为0了，最后剩下的就是缺失了的那个数字了。比如5==101 ^ 6==110 == 011；数组\[0,1,3,4]，result初始为4，循环的值分别为4^0^0=4，4^1^1=4，4^2^3=5，5^3^4=2，最后2作为缺失的数字返回。

### 代码

```java
class Solution {
    public int missingNumber(int[] nums) {
        int result = nums.length;
        for (int i = 0; i < nums.length; i++) {
            result = result ^ i ^ nums[i]; // a^b^b = a
        }
        return result;
    }
}
```

面试中有可能给定的数组是排好序的，那么就用二分查找法来做，找出中间元素nums\[mid]的下标mid，然后用元素值nums\[mid]和下标值mid之间做对比，如果元素值大于下标值，则说明缺失的数字在左边，此时将right赋为mid，反之则将left赋为mid+1。

35 Summary Ranges

Given a sorted integer array without duplicates, return the summary of its ranges.

**Example 1:**

```
Input:  [0,1,2,4,5,7]
Output: ["0->2","4->5","7"]
Explanation: 0,1,2 form a continuous range; 4,5 form a continuous range.
```

**Example 2:**

```
Input:  [0,2,3,4,6,8,9]
Output: ["0","2->4","6","8->9"]
Explanation: 2,3,4 form a continuous range; 8,9 form a continuous range.
```

### **题意和分析**

这道题的思路比较简单，从左边找到右边即可，注意判断一下如果是连续的元素就用"->"，否则就是元素自己，注意下边界就行。

### **代码**

```java
class Solution {
   public List<String> summaryRanges(int[] nums) {
      List<String> result = new ArrayList<>();
      int len = nums.length;
      for (int i = 0; i < len; i++) {
         int temp = nums[i];
         while (i + 1 < len && nums[i + 1] - nums[i] == 1) {//是连续元素
            i++;
         }
         if (nums[i] != temp) {//多于一个元素
            result.add(temp + "->" + nums[i]);
         } else {//只有当前元素自己
            result.add(nums[i] + "");
         }
      }
      return result;
   }
}
```

36 Degree of an Array

Given a non-empty array of non-negative integers `nums`, the **degree** of this array is defined as the maximum frequency of any one of its elements.

Your task is to find the smallest possible length of a (contiguous) subarray of `nums`, that has the same degree as `nums`.

**Example 1:**\


```
Input: [1, 2, 2, 3, 1]
Output: 2
Explanation: 
The input array has a degree of 2 because both elements 1 and 2 appear twice.
Of the subarrays that have the same degree:
[1, 2, 2, 3, 1], [1, 2, 2, 3], [2, 2, 3, 1], [1, 2, 2], [2, 2, 3], [2, 2]
The shortest length is 2. So return 2.
```

**Example 2:**\


```
Input: [1,2,2,3,1,4,2]
Output: 6
```

### **题意和分析**

给一个整数数组，规定degree是这个数组的一个或者多个出现次数最多的元素，求出degree一样的最短子数组的长度。比较直观的做法是先用一个HashMap来统计哪些元素属于degree，同时用另外一个HashMap来统计每个元素的初始出现位置和最后出现位置（第一次出现就更新起始位置，接下来只更新结束位置），这样一次遍历后就知道了数组的degree有哪些元素和每个元素的的起始终止位置，把起始和终止位置相减然后+1，就是拥有同样degree的子数组的长度了；由于degree的元素可能有多个，因此根据degree的数值需要遍历所有的degree元素找到最小的子字符串长度。

另外起始可以把两个HashMap合成一个，key是元素，value是数组分别包含三个元素 - 出现次数，起始位置和终止位置。

### **代码**

```java
class Solution {
    public int findShortestSubArray(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int degree = 0, result = Integer.MAX_VALUE;
        //存元素和出现的次数
        HashMap<Integer, int[]> map = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            if (!map.containsKey(nums[i])) {//第一次出现，更新出现次数，和初始结束位置
                map.put(nums[i], new int[]{1, i, i});
            } else {//之前已经出现过，更新出现次数和结束位置
                int[] temp = map.get(nums[i]);
                temp[0]++;
                temp[2] = i;
            }
        }

        for (int[] value : map.values()) {
            if (value[0] > degree) {//更新degree
                degree = value[0];
                result = value[2] - value[1] + 1;//计算当前元素的起始终止位置
            } else if (value[0] == degree) {
                result = Math.min(result, value[2] - value[1] + 1);//找到最小的长度
            }
        }
        return result;
    }
}
```

37.Find the Duplicate Number

Given an array nums containing n + 1 integers where each integer is between 1 and n (inclusive), prove that at least one duplicate number must exist. Assume that there is only one duplicate number, find the duplicate one.

**Example 1:**

```
Input: [1,3,4,2,2]
Output: 2
```

**Example 2:**

```
Input: [3,1,3,4,2]
Output: 3
```

**Note:**

1. You **must not** modify the array (assume the array is read only).
2. You must use only constant, O(1) extra space.
3. Your runtime complexity should be less than _O_(_n_2).
4. There is only one duplicate number in the array, but it could be repeated more than once.

### **题意和分析**

n+1长度的数组，找出唯一的那个有重复的数字，可能重复多次，不能修改数组并要求常量空间和小于平方级线型时间复杂度（不能用额外空间来统计出现的次数，不能排序，也不能套两个循环来暴力破解）。

1）二分查找，利用数组的元素的值在区间\[1, n]的特点进行搜索，首先求出中间的索引mid，然后遍历整个数组，统计所有小于等于索引mid的元素的个数，如果元素个数大于mid索引，则说明重复值在\[mid+1, n]这些索引之间，因为“较小的数比较多”，反之，重复值应在\[1, mid-1]之间（“较大的数比较多”），然后依次类推，直到搜索完成，此时的low就是我们要求的重复值；

2）双指针，数组元素的范围是\[1, n]，利用数组元素和坐标的转换来形成一个闭环，利用快慢指针找到重复的值，参考[这里](http://bookshadow.com/weblog/2015/09/28/leetcode-find-duplicate-number/)[这里](https://leetcode.com/problems/find-the-duplicate-number/discuss/72845/Java-O\(n\)-time-and-O\(1\)-space-solution.-Similar-to-find-loop-in-linkedlist.)。

### **代码**

二分查找

```java
class Solution {
    public int findDuplicate(int[] nums) {
        int low = 1, high = nums.length - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            int count = 0;
            for (int num : nums) {
                if (num <= mid) {
                    count++;
                }
            }
            if (count <= mid) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return low;
    }
}
```

双指针

```java
class Solution {
    public int findDuplicate(int[] nums) {
        int len = nums.length, slow = len, fast = len;
        do {
            slow = nums[slow - 1];
            fast = nums[nums[fast-1] - 1];
        } while (slow != fast);
        slow = len;
        while (slow != fast) {
            slow = nums[slow - 1];
            fast = nums[fast - 1];
        }
        return slow;
    }
```

38 Maximum Product Subarray

Given an integer array `nums`, find the contiguous subarray within an array (containing at least one number) which has the largest product.

**Example 1:**

```
Input: [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
```

**Example 2:**

```
Input: [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
```



滚动数组

```java
class Solution {
    public int maxProduct(int[] nums) {
        int maxProduct = nums[0], temp = 0;
        for (int i = 1, max = maxProduct, min = maxProduct; i < nums.length; i++) {
            if (nums[i] < 0) {
                temp = min;
                min = max;
                max = temp;
            }
            max = Math.max(nums[i], max * nums[i]);
            min = Math.min(nums[i], min * nums[i]);
            maxProduct = Math.max(maxProduct, max);
        }
        return maxProduct;
    }
}
```

DP的解法

```java
class Solution {
    public int maxProduct(int[] nums) {
        int max = nums[0];
        int prevMin = nums[0], prevMax = nums[0];
        int curMin, curMax;
        for (int i = 1; i < nums.length; i++) {
            curMin = Math.min(Math.min(prevMax * nums[i], prevMin * nums[i]), nums[i]);
            curMax = Math.max(Math.max(prevMax * nums[i], prevMin * nums[i]), nums[i]);
            prevMin = curMin;
            prevMax = curMax;
            max = Math.max(curMax, max);
        }
        return max;
    }
}
```

39 Minimum path sum

Given a _m_ x _n_ grid filled with non-negative numbers, find a path from top left to bottom right which _minimizes_ the sum of all numbers along its path.

**Note:** You can only move either down or right at any point in time.

**Example:**

```
Input:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
Output: 7
Explanation: Because the path 1→3→1→1→1 minimizes the sum.
```

### **题意和分析**

给一个二维数组，从左上到右下找一条经过的元素加起来最小的path，返回所有元素加起来的和。全局最优，用DP， dp\[i]\[j] = grid\[i]\[j] + min(dp\[i - 1]\[j])，所有路径经过的元素之和等于当前元素的值加上上一个可到达的元素的总和最小值。

### **代码**

```java
class Solution {
    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }
        int m = grid.length, n = grid[0].length;
        int[][] dp = new int[m][n];

        //初始化第一个值
        dp[0][0] = grid[0][0];
        //初始化第一行和第一列
        for (int i = 1; i < m; i++) {
            dp[i][0] = grid[i][0] + dp[i-1][0];//当前值+上一步的最小值
        }
        for (int j = 1; j < n; j++) {
            dp[0][j] = grid[0][j] + dp[0][j-1];
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = grid[i][j] + Math.min(dp[i-1][j], dp[i][j-1]);
            }
        }
        return dp[m-1][n-1];
    }
}
```

# Amazon Frequency 1~ 30

## 1: Two Sum

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int[] res = new int[2];
        if (nums == null || nums.length == 0 ) return res;
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i ++) {
            if (map.containsKey(target - nums[i])){
               int index1 = map.get(target - nums[i]);
               res[0] = index1;
               res[1] = i; 
            } else {
                map.put(nums[i], i);
            }
        }
        
        return res;
    }
}
```

## 42-Trapping Rain Water

```java
class Solution {
    public int trap(int[] height) {
        if (height == null || height.length == 0) return 0;
        int left = 0; int right = height.length - 1;
        int leftMax = height[left];
        int rightMax = height[right];
        int sum = 0;
        while (left <= right){
           leftMax = Math.max(leftMax, height[left]);
           rightMax = Math.max(rightMax, height[right]);
           if (leftMax < rightMax){
               sum += leftMax - height[left];
               left ++;
           } else {
               sum += rightMax - height[right];
               right --;
           }
            
        }
        return sum;
    }
}
```




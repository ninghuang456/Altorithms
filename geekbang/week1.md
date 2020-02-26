# week1

Array, LinkedList

```text
1:Two sum
class Solution {
    public int[] twoSum(int[] nums, int target) {
        if (nums == null || nums.length < 2) return new int[0];
        HashMap<Integer,Integer> map = new HashMap<Integer,Integer>();
        for (int i = 0; i < nums.length; i ++) {
            if (map.containsKey(target - nums[i])) {
                return new int[]{map.get(target - nums[i]),i};
            }
            map.put(nums[i],i);
        }
        return new int[0];        
    }
}

2:Merge Two Sorted Lists   
class Solution {
   public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null || l2 == null) {
            return l1 == null ? l2 : l1;
        }
        ListNode head = null;
        if (l1.val < l2.val) {
            head = l1;
            head.next = mergeTwoLists(l1.next, l2);
        } else {
            head = l2;
            head.next = mergeTwoLists(l1,l2.next);
        }  
} }

3:Remove Duplicates from Sorted Array    
class Solution {
    public int removeDuplicates(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        int total = 1;
        for (int i = 1; i < nums.length; i ++) {
            if (nums[i] > nums[i - 1]) {
                nums[total] = nums[i];
                total ++;
            }    }
        return total;    
    }  }

4:Plus One  
class Solution {
    public int[] plusOne(int[] digits) {
        if (digits == null || digits.length == 0) return new int[0];
        for (int i = digits.length - 1; i >= 0; i --) { // i -- not i ++ starting from digits.length - 1;
            if (digits[i] < 9) {
                digits[i] = digits[i] + 1;
                return digits;
            }
            digits[i] = 0;
        }
        int[] result = new int[digits.length + 1];
            result[0] = 1;
         return result;
    }
}

```


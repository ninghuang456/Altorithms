# 11~20 classic

```text
11  Add Two Numbers
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
         if (l1 == null || l2 == null) {
            return l1 == null ? l2 : l1;
        }
        ListNode dum = new ListNode(-1);
        ListNode cur = dum;
        int res = 0;
       
        while(l1 != null || l2 != null) {
            int sum = res;
          if (l1 != null ){
              sum += l1.val;
              l1 = l1.next;
            }
          if (l2 != null) {
              sum += l2.val;
              l2 = l2.next;
          }
          res = sum / 10;
          int val = sum % 10;
          cur.next = new ListNode(val);         
          cur = cur.next; 
          
        }
        if (res != 0) {//循环结束后最后判断下是否还有进位
            cur.next = new ListNode(res);
        }
        return dum.next;
    }
         
}

12. Merge Two Sorted Lists

class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null|| l2 == null){
            return (l1 == null) ? l2 : l1;
        }
        ListNode dumy = new ListNode(-1);
        ListNode cur = dumy;
        while (l1 != null && l2!= null){
            if (l1.val >= l2.val){
                cur.next = l2;
                l2 = l2.next;
            } else {
                cur.next = l1;
                l1 = l1.next;
            }
            cur = cur.next;
        }
        cur.next = (l1 == null) ? l2 : l1;
      
        return dumy.next;
    }
}
recursion
public ListNode mergeTwoLists(ListNode l1, ListNode l2){
		if(l1 == null) return l2;
		if(l2 == null) return l1;
		if(l1.val < l2.val){
			l1.next = mergeTwoLists(l1.next, l2);
			return l1;
		} else{
			l2.next = mergeTwoLists(l1, l2.next);
			return l2;
		}
}

13. Merge k Sorted Lists
class Solution {
      public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        return dac(lists, 0, lists.length - 1);
    }
    private ListNode dac (ListNode[] lists, int left, int right) {
        if (left == right) {
            return lists[left];
        } 
        if (left < right) {
            int mid = left + (right - left) / 2;
            ListNode list1 = dac(lists, left, mid);
            ListNode list2 = dac(lists, mid + 1, right); 
            ListNode merge = mergeTwoLists(list1, list2);
            return merge;
        } else {
            return null;
        }
    }
    
    private ListNode mergeTwoLists (ListNode List1, ListNode List2) {
        if (List1 == null) {
            return List2;
        }
        if (List2 == null) {
            return List1;
        }
        if (List1.val < List2.val) {
            // not ListNode list1 = mergeTwoLists(List1.next , List2);
            List1.next = mergeTwoLists(List1.next , List2);
            return List1;
        } else {
            List2.next = mergeTwoLists(List1 , List2.next);
            return List2;
        }
    }
}

14.Reverse Linked List
recursion
class Solution {
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null){
            return head;
        }
        ListNode newHead = reverseList(head.next);
        head.next.next = head; 
        head.next = null;// don't forget this one
        return newHead;
    }
}

Iteration
class Solution {
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode pre = null;
        ListNode cur = head;
        
        while (cur!= null){
            ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;//need return pre not cur; not cur
    }
}

15.Longest Substring Without Repeating Characters

Given a string, find the length of the longest substring without repeating characters.
Input: "abcabcbb"
Output: 3 
Explanation: The answer is "abc", with the length of 3. 

class Solution {
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.length() == 0) return 0;
        Set<Character> hset = new HashSet<>();
        int left = 0;
        int right = 0;
        int max = 0;
        
        while(right < s.length()){
            if (!hset.contains(s.charAt(right))){
                hset.add(s.charAt(right));
                max = Math.max(max, hset.size());
                right ++;
            } else {
                hset.remove(s.charAt(left));
                left ++;
            }
        }
        return max;
        // "abccacefsas"
    }
}

16.Pow(x, n)
class Solution {
    public double myPow(double x, int n) {
      
        if (n < 0) {
            x = 1/x;
            n = -n; 
        }     
        return mynewpow(x, n);  
    }
    
     public double mynewpow(double x, int n) {
         if (n ==1) return x;
         if (n ==0) return 1.00;
         double result = mynewpow(x,n/2);
         if (n%2 == 0) {
             return result * result;
         } else {
             return result * result * x;
         }
     }
}

17. 3Sum
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


18.Container With Most Water
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

19. Linked List Cycle
public class Solution {
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode slow = head;
        ListNode fast = head;
        while ( fast != null && fast.next!=null){
             slow = slow.next;
             fast = fast.next.next;
            if (fast == slow) {
                return true;
            }
        }
        return false;
    }
}

20. Subarray Sum Equals K
Given an array of integers and an integer k, you need to find the total number of continuous subarrays whose sum equals to k.
class Solution {
    public int subarraySum(int[] nums, int k) {
        int sum = 0, result = 0;
        Map<Integer, Integer> preSum = new HashMap<>();
        preSum.put(0, 1);
        
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            if (preSum.containsKey(sum - k)) {
                result += preSum.get(sum - k);
            }
            preSum.put(sum, preSum.getOrDefault(sum, 0) + 1);
        }        
        return result;        
    }
}

```


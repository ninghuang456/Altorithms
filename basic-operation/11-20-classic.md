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


```


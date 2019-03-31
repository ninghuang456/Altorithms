# LinkedList

1 Add Two Numbers

ou are given two **non-empty** linked lists representing two non-negative integers. The digits are stored in **reverse order** and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

**Example:**

```text
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
```

```text
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
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
```

2 Copy List with Random Pointer

A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.

Return a [**deep copy**](https://en.wikipedia.org/wiki/Object_copying#Deep_copy) of the list.

**Example 1:**

![](https://discuss.leetcode.com/uploads/files/1470150906153-2yxeznm.png)

```text
Input:
{"$id":"1","next":{"$id":"2","next":null,"random":{"$ref":"2"},"val":2},"random":{"$ref":"2"},"val":1}

Explanation:
Node 1's value is 1, both of its next and random pointer points to Node 2.
Node 2's value is 2, its next pointer points to null and its random pointer points to itself.
```

```text
class Node {
    public int val;
    public Node next;
    public Node random;
    public Node() {}
    public Node(int _val,Node _next,Node _random) {
        val = _val;
        next = _next;
        random = _random;
    }
};
*/
class Solution {
    public Node copyRandomList(Node head) {      
        Map<Node, Node> map = new HashMap<>();      
        Node cur = head;
        while (cur != null) {
            map.put(cur, new Node(cur.val));
            cur = cur.next;
        }       
        cur = head;
        while (cur != null) {
            map.get(cur).next = map.get(cur.next);
            map.get(cur).random = map.get(cur.random);
            cur = cur.next;
        }
        return map.get(head); 
    }
}
```

3 Merge Two Sorted Lists

Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.

```text
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
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
```

4 Reverse Linked List

```text
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
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
```

5 Merge k Sorted Lists

```text
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
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
```

6 Linked List Cycle

Given a linked list, determine if it has a cycle in it.

```text
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
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
```


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



* Reverse Nodes in k-Group

  原题概述

  Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.

  k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes in the end should remain as it is.

  Example:

  Given this linked list: 1-&gt;2-&gt;3-&gt;4-&gt;5

  For k = 2, you should return: 2-&gt;1-&gt;4-&gt;3-&gt;5

  For k = 3, you should return: 3-&gt;2-&gt;1-&gt;4-&gt;5

  Note:

  •    Only constant extra memory is allowed.

  •    You may not alter the values in the list's nodes, only nodes itself may be changed.

  题意和分析

  这道题要求以k个结点为一组进行翻转，实际上是把原链表分成若干小段，然后分别对其进行翻转，以题目中给的例子来看，对于给定链表1-&gt;2-&gt;3-&gt;4-&gt;5，一般在处理链表问题时，我们大多时候都会在开头再加一个dummy node，因为翻转链表时头结点可能会变化，为了记录当前最新的头结点的位置而引入的dummy node，那么我们加入dummy node后的链表变为-1-&gt;1-&gt;2-&gt;3-&gt;4-&gt;5，如果k为3的话，我们的目标是将1,2,3翻转一下，那么我们需要一些指针，pre和next分别指向要翻转的链表的前后的位置，然后翻转后pre的位置更新到如下新的位置： 

  -1-&gt;1-&gt;2-&gt;3-&gt;4-&gt;5

  \|           \|

  pre         next

  -1-&gt;3-&gt;2-&gt;1-&gt;4-&gt;5

  ```text
      |  |
     pre next
  ```

以此类推，只要next走过k个节点，就可以进行局部翻转了。 也可以使用递归来做，我们用head记录每段的开始位置，current记录结束位置的下一个节点，然后我们调用reverse函数来将这段翻转，然后得到一个newHead，原来的head就变成了末尾，这时候后面接上递归调用下一段得到的新节点，返回newHead即可。

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
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode curr = head;
        int count = 0;
        while (curr != null && count != k) { // find the k+1 node
            curr = curr.next;
            count++;
        }
        if (count == k) { // if k+1 node is found
            curr = reverseKGroup(curr, k); // reverse list with k+1 node as head
            // head - head-pointer to direct part, 
            // curr - head-pointer to reversed part;
            while (count-- > 0) { // reverse current k-group: 
                ListNode tmp = head.next; // tmp - next head in direct part
                head.next = curr; // preappending "direct" head to the reversed list 
                curr = head; // move head of reversed part to a new node
                head = tmp; // move "direct" head to the next node in direct part
            }
            head = curr;
        }
        return head;
    }
}

```


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


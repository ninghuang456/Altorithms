# MicroSoft

## 146: LRU Cache

```java
class Node {
        int key;
        int value;
        Node pre;
        Node next;
        
        public Node (int key, int value){
            this.key = key;
            this.value = value;
        }
    }

class DoubleList {
    Node head;
    Node tail;
    int size;
    
    public DoubleList(int size){
         this.size = size;
        head = new Node(-1,-1);
        tail = new Node(-1,-1);
        head.next = tail;
        tail.pre = head;
       
    }
    
    public void remove(Node node){
        node.pre.next = node.next;
        node.next.pre = node.pre;
        size --;
    }
    
    public void addFirst(Node node){
        node.next = head.next;
        node.pre = head;
        head.next.pre = node;
        head.next = node;
        size ++;
    }
    
    public Node removeLast(){
        if (tail.pre == head){
            return null;
        }
        Node node = tail.pre;
        remove(node);
        return node;
    }
    
    public int getSize(){
        return this.size;
    }
}

class LRUCache {
    HashMap<Integer, Node> map;
    DoubleList cache;
    int capacity;
    
    public LRUCache(int capacity) {
        this.capacity = capacity;
        map = new HashMap<>();
        cache = new DoubleList(0);
    }
    
    public int get(int key) {
        if (!map.containsKey(key)) return -1;
        Node node = map.get(key);
        int value = node.value;
        put(key, value);
        return value;    
    }
    
    public void put(int key, int value) {
        if(map.containsKey(key)){
            Node cur = map.get(key);
            cache.remove(cur);
            cur.value = value;
            map.put(key,cur);
            cache.addFirst(cur);
            return;
        }
        if(cache.getSize() == capacity){
            Node last = cache.removeLast();
            map.remove(last.key);
        }
        Node nodeAdd = new Node(key, value);
        map.put(key, nodeAdd);
        cache.addFirst(nodeAdd);   
    }
}
```

## 545: Boundary of binary search tree

```java
class Solution {

    public List<Integer> boundaryOfBinaryTree(TreeNode root) {
        List<Integer> result = new LinkedList<>();
        if (root == null) {
            return result;
        }
        
        if (!isLeaf(root)) {
            result.add(root.val);
        }
        
        // 先左边界，再叶子节点，最后右边界
        leftTrunk(root.left, result); // 从root的左孩子开始判断左边界
        addLeaves(root, result); // 从root开始判断叶子节点
        rightTrunk(root.right, result); // 从root的右孩子开始判断右边界

        return result;
    }

    // 左边界，没有左孩子，需要有右孩子，等同于不是叶子节点的情况下，没有左孩子
    private void leftTrunk(TreeNode root, List<Integer> result) {
        if (root == null) {
            return;
        }
        if (!isLeaf(root)) {
            result.add(root.val);
        }
        if (root.left != null) {
            leftTrunk(root.left, result);
        } else {
            leftTrunk(root.right, result);
        }
    }

    // 右边界，没有右孩子，需要有左孩子，等同于不是叶子节点的情况下，没有右孩子
    private void rightTrunk(TreeNode root, List<Integer> result) {
        if (root == null) {
            return;
        }
        if (root.right != null) {
            rightTrunk(root.right, result);
        } else {
            rightTrunk(root.left, result);
        }
        if (!isLeaf(root)) {
            result.add(root.val);
        }
    }

    // 叶子节点，左右孩子都没有
    private void addLeaves(TreeNode node, List<Integer> result) {
        if (node == null) {
            return;
        }
        if (isLeaf(node)) {
            result.add(node.val);
        } else {
            addLeaves(node.left, result);
            addLeaves(node.right, result);
        }
    }

    private boolean isLeaf(TreeNode node) {
        return node.left == null && node.right == null;
    }
}
```

## 460: LFU Cache

```java
public class LFUCache {
    HashMap<Integer, Integer> vals;
    HashMap<Integer, Integer> counts;
    HashMap<Integer, LinkedHashSet<Integer>> lists;
    int cap;
    int min = -1;
    public LFUCache(int capacity) {
        cap = capacity;
        vals = new HashMap<>();
        counts = new HashMap<>();
        lists = new HashMap<>();
        lists.put(1, new LinkedHashSet<>());
    }
    
    public int get(int key) {
        if(!vals.containsKey(key))
            return -1;
        int count = counts.get(key);
        counts.put(key, count+1);
        lists.get(count).remove(key);
        if(count==min && lists.get(count).size()==0)
            min++;
        if(!lists.containsKey(count+1))
            lists.put(count+1, new LinkedHashSet<>());
        lists.get(count+1).add(key);
        return vals.get(key);
    }
    
    public void set(int key, int value) {
        if(cap<=0)
            return;
        if(vals.containsKey(key)) {
            vals.put(key, value);
            get(key);
            return;
        } 
        if(vals.size() >= cap) {
            int evit = lists.get(min).iterator().next();
            lists.get(min).remove(evit);
            vals.remove(evit);
        }
        vals.put(key, value);
        counts.put(key, 1);
        min = 1;
        lists.get(1).add(key);
    }
}
```

## 1647: Minimum Deletions to Make Character Frequencies Unique

```java
A string s is called good if there are no two different characters in s that have 
the same frequency.
Given a string s, return the minimum number of characters you need to delete to 
make s good.
The frequency of a character in a string is the number of times it appears in 
the string. For example, in the string "aab", the frequency of 'a' is 2, 
while the frequency of 'b' is 1.

Example 1:
Input: s = "aab"
Output: 0
Explanation: s is already good.

Example 2:
Input: s = "aaabbbcc"
Output: 2
Explanation: You can delete two 'b's resulting in the good string "aaabcc".
Another way it to delete one 'b' and one 'c' resulting in the good string "aaabbc".

Example 3:
Input: s = "ceabaacb"
Output: 2
Explanation: You can delete both 'c's resulting in the good string "eabaab".
Note that we only care about characters that are still in the string at the end 
(i.e. frequency of 0 is ignored).

public int minDeletions(String s) {
	int freq[] = new int[26];
	for (char c : s.toCharArray())
		freq[c - 'a']++;
	Arrays.sort(freq);
	int keep = freq[25], prev = keep;
	for (int i = 24; i >= 0 && freq[i] != 0 && prev != 0; i--) {
		prev = Math.min(freq[i], prev - 1);
		keep += prev;
	}
	return s.length() - keep;
}

****************************************************************************
public int minDeletions(String s) {
	int freq[] = new int[26];
	for (char c : s.toCharArray())
		freq[c - 'a']++;
	Arrays.sort(freq);
	int keep = freq[25], prev = keep;
	for (int i = 24; i >= 0 && freq[i] != 0 && prev != 0; i--) {
		prev = Math.min(freq[i], prev - 1);
		keep += prev;
	}
	return s.length() - keep;
}
Complexity
Time: step 1 is O(n), other steps are O(1) (sorting/checking 26 numbers) 
- overall O(n)
space: using array of size 26 - O(1)
```

## 277 - Find the Celebrity

Suppose you are at a party with `n` people \(labeled from `0` to `n - 1`\), and among them, there may exist one celebrity. The definition of a celebrity is that all the other `n - 1` people know him/her, but he/she does not know any of them.

Now you want to find out who the celebrity is or verify that there is not one. The only thing you are allowed to do is to ask questions like: "Hi, A. Do you know B?" to get information about whether A knows B. You need to find out the celebrity \(or verify there is not one\) by asking as few questions as possible \(in the asymptotic sense\).

You are given a helper function `bool knows(a, b)` which tells you whether A knows B. Implement a function `int findCelebrity(n)`. There will be exactly one celebrity if he/she is in the party. Return the celebrity's label if there is a celebrity in the party. If there is no celebrity, return `-1`.

Input: graph = \[\[1,1,0\],\[0,1,0\],\[1,1,1\]\]  Output: 1 Explanation: There are three persons labeled with 0, 1 and 2. graph\[i\]\[j\] = 1 means person i knows person j, otherwise graph\[i\]\[j\] = 0 means person i does not know person j. The celebrity is the person labeled as 1 because both 0 and 2 know him but 1 does not know anybody.

![](.gitbook/assets/277_example_1_bold.png)

```java
/* The knows API is defined in the parent class Relation.
      boolean knows(int a, int b); */

public class Solution extends Relation {
    public int findCelebrity(int n) {
        if (n < 0) {
            return -1;
        } 
        
        // 找到出度为0的人，也就是不认识任何别人的人
        int candidate = 0;
        for (int i = 0; i < n; i++) {
            if (knows(candidate, i)) {
                candidate = i;
            }
        }
        
        // 看看是不是所有人都认识他 和 是否他不认识所有人
        for (int i = 0; i < n; i++) {
            if (i != candidate) { // 自己不和自己比较
                if (!knows(i, candidate)) { // 有人不认识他
                    return -1;
                }
                if (knows(candidate, i)) {// 他认识某人
                    return -1;
                }
            }
        }
        return candidate;
    }
}
```

## 151: Reverse Words in a String

```java
Input: s = "the sky is blue"
Output: "blue is sky the"

class Solution {
    public String reverseWords(String s) {
        if (s == null || s.length() == 0 || s.trim().isEmpty()) return "";
        String[] strs = s.trim().split(" ");
        int i = 0; int j = strs.length - 1;
        while (i < j){
            String temp = strs[i];
            strs[i ++] = strs[j];
            strs[j --] = temp;
        }
       StringBuilder sb = new StringBuilder();
       for (String str: strs){
           if(!str.trim().isEmpty()){
               sb.append(str).append(" "); 
           }
       } 
       return sb.toString().trim(); 
    }
}
```

## 1578: Minimum Deletion Cost to Avoid Repeating Letters

```java
Given a string s and an array of integers cost where cost[i] is the cost of 
deleting the ith character in s.
Return the minimum cost of deletions such that there are no two identical 
letters next to each other.
Notice that you will delete the chosen characters at the same time, 
in other words, after deleting a character, the costs of deleting other
 characters will not change.

Input: s = "abaac", cost = [1,2,3,4,5]
Output: 3
Explanation: Delete the letter "a" with cost 3 to get "abac" 
(String without two identical letters next to each other).

Explanation
For each group of continuous same characters,
we need cost = sum_cost(group) - max_cost(group)
Complexity
Time O(N)
Space O(1)

    public int minCost(String s, int[] cost) {
        int res = 0, max_cost = 0, sum_cost = 0, n = s.length();
        for (int i = 0; i < n; ++i) {
            if (i > 0 && s.charAt(i) != s.charAt(i - 1)) {
                res += sum_cost - max_cost;
                sum_cost = max_cost = 0;
            }
            sum_cost += cost[i];
            max_cost = Math.max(max_cost, cost[i]);
        }
        res += sum_cost - max_cost;
        return res;
    }
```


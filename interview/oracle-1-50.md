# Oracle 1 ~ 40

## 99 - Recover Binary Search Tree

```java
You are given the root of a binary search tree (BST), where exactly two nodes 
of the tree were swapped by mistake. Recover the tree without 
changing its structure.
Follow up: A solution using O(n) space is pretty straight forward. 
Could you devise a constant space solution? 

// o(n)
class Solution {
    private TreeNode first;
    private TreeNode second;
    private TreeNode pre;
    public void recoverTree(TreeNode root) {
        if(root==null) return;
        first = null;
        second = null;
        pre = null;
        inorder(root);
        int temp = first.val;
        first.val = second.val;
        second.val = temp;
    }
    
    private void inorder(TreeNode root){
        if(root==null) return;
        inorder(root.left);
        
        if(first==null && (pre==null ||pre.val>=root.val)){
            first = pre;
        }
        if(first!=null && pre.val>=root.val){
            second = root;
        }
        pre = root;
        inorder(root.right);
    }
}

//莫里斯遍历 只花O(1)空间复杂度
public void morrisTraversal(TreeNode root){
		TreeNode temp = null;
		while(root!=null){
			if(root.left!=null){
				// connect threading for root
				temp = root.left;
				while(temp.right!=null && temp.right != root)
					temp = temp.right;
				// the threading already exists
				if(temp.right!=null){
					temp.right = null;
					System.out.println(root.val);
					root = root.right;
				}else{
					// construct the threading
					temp.right = root;
					root = root.left;
				}
			}else{
				System.out.println(root.val);
				root = root.right;
			}
		}
	}
	//题解：
	public void recoverTree(TreeNode root) {
        TreeNode pre = null;
        TreeNode first = null, second = null;
        // Morris Traversal
        TreeNode temp = null;
		while(root!=null){
			if(root.left!=null){
				// connect threading for root
				temp = root.left;
				while(temp.right!=null && temp.right != root)
					temp = temp.right;
				// the threading already exists
				if(temp.right!=null){
				    if(pre!=null && pre.val > root.val){
				        if(first==null){first = pre;second = root;}
				        else{second = root;}
				    }
				    pre = root;
				    
					temp.right = null;
					root = root.right;
				}else{
					// construct the threading
					temp.right = root;
					root = root.left;
				}
			}else{
				if(pre!=null && pre.val > root.val){
				    if(first==null){first = pre;second = root;}
				    else{second = root;}
				}
				pre = root;
				root = root.right;
			}
		}
		// swap two node values;
		if(first!= null && second != null){
		    int t = first.val;
		    first.val = second.val;
		    second.val = t;
		}
    }
```

## 981：Time Based Key-Value Store

```java
Create a timebased key-value store class TimeMap, that supports two operations.
1. set(string key, string value, int timestamp)
Stores the key and value, along with the given timestamp.
2. get(string key, int timestamp)
Returns a value such that set(key, value, timestamp_prev) was called previously,
 with timestamp_prev <= timestamp.
If there are multiple such values, it returns the one with the largest 
timestamp_prev.
If there are no values, it returns the empty string ("").

class TimeMap {
    class Node {
        String value;
        int timestamp;
        Node(String value, int timestamp) {
            this.value = value;
            this.timestamp = timestamp;
        }
    }  
    Map<String, List<Node>> map;
    /** Initialize your data structure here. */
    public TimeMap() {
        map = new HashMap();
    }
    public void set(String key, String value, int timestamp) {
        map.putIfAbsent(key, new ArrayList());
        map.get(key).add(new Node(value, timestamp));
    }
    
    public String get(String key, int timestamp) {
        List<Node> nodes = map.get(key);
        if (nodes == null) return "";
        
        int left = 0, right = nodes.size() - 1;
        while (left + 1 < right) {
            int mid = left + (right - left) / 2;
            Node node = nodes.get(mid);
            if (node.timestamp == timestamp) {
                return node.value;
            } else if (node.timestamp < timestamp) {
                left = mid;
            } else {
                right = mid;
            }
        }
        if (nodes.get(right).timestamp <= timestamp) 
             return nodes.get(right).value;
        else if (nodes.get(left).timestamp <= timestamp) 
              return nodes.get(left).value;
        return "";
    }
}
```

## 554 - Brick Wall

```java
There is a brick wall in front of you. The wall is rectangular and has
several rows of bricks. The bricks have the same height but different width. 
You want to draw a vertical line from the top to the bottom and cross the 
least bricks.
Input: [[1,2,2,1],
        [3,1,2],
        [1,3,2],
        [2,4],
        [3,1,2],
        [1,3,1,1]]

Output: 2

class Solution {
    public int leastBricks(List<List<Integer>> wall) {
        Map<Integer, Integer> map = new HashMap();     
        int count = 0;
        for (List<Integer> row : wall) {
            int sum = 0;
            for (int i = 0; i < row.size() - 1; i++) {
                sum += row.get(i);
                map.put(sum, map.getOrDefault(sum, 0) + 1);
                count = Math.max(count, map.get(sum));
            }
        }    
        return wall.size() - count;
    }
}
//1:通过哈希，将每次的砖的长度逐渐的累加，相同的长度就是缝隙相同的地方；
//2: 当遍历完所有的砖之后，重复最多的就是缝隙重合最多的，总的层数减去该值
//，就是从该缝隙穿过时，穿过的砖的数量；
```

## 272 ： Closest Binary Search Tree Value II

```java
Given a non-empty binary search tree and a target value, find k values in the 
BST that are closest to the target.
Note: Given target value is a floating point.
You may assume k is always valid, that is: k ≤ total nodes.
You are guaranteed to have only one unique set of k values in 
the BST that are closest to the target.
Example:
Input: root = [4,2,5,1,3], target = 3.714286, and k = 2
    4
   / \
  2   5
 / \
1   3
Output: [4,3]

//O(N)解法，
class Solution {
    public List<Integer> closestKValues(TreeNode root, double target, int k) {
        Deque<Integer> dq = new LinkedList<>();
        inorder(root, dq);
        while (dq.size() > k) {
            if (Math.abs(dq.peekFirst()-target)> Math.abs(dq.peekLast() - target))
                dq.pollFirst();
            else 
                dq.pollLast();
        }
        return new ArrayList<Integer>(dq);
    }
    
    public void inorder(TreeNode root, Deque<Integer> dq) {
        if (root == null) return;
        inorder(root.left, dq);
        dq.add(root.val);
        inorder(root.right, dq);
    }
}

// o()
class Solution {
    public List<Integer> closestKValues(TreeNode root, double target, int k) {
        Stack<TreeNode> smaller = new Stack<>();
        Stack<TreeNode> larger = new Stack<>();
        pushSmaller(root, target, smaller);
        pushLarger(root, target, larger);
        
        List<Integer> res = new ArrayList<>();
        TreeNode cur = null;
        while (res.size() < k) {
            if (smaller.isEmpty() || (!larger.isEmpty() && larger.peek().val - target < target - smaller.peek().val)) {
                cur = larger.pop();
                res.add(cur.val);
                pushLarger(cur.right, target, larger);
            } else {
                cur = smaller.pop();
                res.add(cur.val);
                pushSmaller(cur.left, target, smaller);
            }
        }
        return res;    
    }
    
    private void pushSmaller(TreeNode node, double target,  Stack<TreeNode> stack) {
        while (node != null) {
            if (node.val < target) {
                stack.push(node);
                node = node.right;
            } else {
                node = node.left;
            }
        }
    }
    
    private void pushLarger(TreeNode node, double target, Stack<TreeNode> stack) {
        while (node != null) {
            if (node.val >= target) {
                stack.push(node);
                node = node.left;
            } else {
                node = node.right;
            }
        }
    }
}
```

## 285 - Inorder Successor in BST

```java
Given a binary search tree and a node in it, find the in-order successor
 of that node in the BST.
The successor of a node p is the node with the smallest key greater than p.val.

public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
    TreeNode res = null;
    while(root!=null) {
        if(root.val > p.val) {
        	res = root;
        	root = root.left;
        }
        else root = root.right;
    }
    return res;
}

// predecessors， smaller
public TreeNode inorderPredecessors(TreeNode root, TreeNode p) {
    TreeNode res = null;
    while(root!=null) {
        if(root.val < p.val) {
        	res = root;
        	root = root.right;
        }
        else root = root.left;           
    }
    return res;
}


```

## 1190 - Reverse Substrings Between Each Pair of Parentheses

```java
You are given a string s that consists of lower case English letters and brackets. 
Reverse the strings in each pair of matching parentheses, starting from the 
innermost one.Your result should not contain any brackets.
Example 1:
Input: s = "(abcd)"
Output: "dcba"

Example 2:
Input: s = "(u(love)i)"
Output: "iloveu"
Explanation: The substring "love" is reversed first, 
then the whole string is reversed.

class Solution {
    public String reverseParentheses(String s) {
        int n = s.length();
        Stack<Integer> stack = new Stack<>();
        int[] pair = new int[n];

        //先去找匹配的括号
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else if (s.charAt(i) == ')') {
                int j = stack.pop();
                pair[i] = j;
                pair[j] = i;
            }
        }

        StringBuilder res = new StringBuilder();
        // i是当前位置 | d是方向,1就是向右穿
        for (int i = 0, d = 1; i < n; i+=d) {
            if (s.charAt(i) == '(' || s.charAt(i) == ')') {
                // 如果碰到括号，那么去他对应的括号，并且将方向置反
                i = pair[i];
                d = -d;
            } else {
                res.append(s.charAt(i));
            }
        }
        return res.toString();
    }
}
```

## 200 - Number of Islands

```java
class Solution {
    
    public int numIslands(char[][] grid) {
        int[][] directions = {{-1, 0}, {0, -1}, {1, 0}, {0, 1}};
        int r = grid.length;
        int c = grid[0].length;
        boolean[] marked = new boolean[r * c];
        int count = 0;
        for (int i = 0; i < r; i ++) {
            for (int j = 0; j < c; j ++) {
             if (!marked[i * c + j] && grid[i][j] == '1') {
                 count++;
                 int index = i * c + j;
                 LinkedList<Integer> queue = new LinkedList<>();
                 queue.offer(index);
                 marked[i * c + j] = true;
                 while (!queue.isEmpty()) {
                     int cur = queue.poll();
                     int curX = cur / c;
                     int curY = cur % c;
                     for (int k = 0; k < 4; k ++) {
                        int nextX = curX + directions[k][0];
                        int  nextY = curY + directions[k][1];
                         if (inArea(r,c,nextX, nextY) && grid[nextX][nextY] == '1' && !marked[nextX * c + nextY]){
                             queue.offer(nextX * c + nextY);
                             marked[nextX * c + nextY] = true;
                         }
                     }
                  }   
               } 
            }
        }
        
        return count;
        
    }
    
    public boolean inArea(int r, int c, int i, int j) {
        return i >= 0 && i < r && j >= 0 && j < c;
    }
}

//dfs
class Solution {
    int[][] dis = new int[][]{{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) return 0;
        int total = 0;
        for (int r = 0; r < grid.length; r ++) {
            for (int c = 0; c < grid[0].length; c++) {
                if (grid[r][c] == '1') {
                    searchArea(grid, r , c);
                    total ++;
                }
            }
        }
        return total;
    }
    
    private void searchArea(char[][] grid, int r, int c){
        if(!inArea(grid,r,c)){
            return;
        }
        if(grid[r][c] != '1'){
            return;
        }
        grid[r][c] = '2';
        for (int i = 0; i < 4; i ++){
            int nextR = r + dis[i][0];
            int nextC = c + dis[i][1];
            searchArea(grid, nextR, nextC);
        }
            
        
    }
    
    private boolean inArea(char[][] grid, int r, int c){
        return r >= 0 && r < grid.length && c >= 0 && c < grid[0].length;
    }
}
```

## 158: Read N Characters Given Read4 II - Call multiple times

```java
public class Solution extends Reader4 {
     int i = 0;
     int size = 0;
     char[] buf4 = new char[4];
    public int read(char[] buf, int n) {
        int index = 0;
        while (index < n){
            if(size == 0){
                size = read4(buf4);
                if(size == 0)
                    break;
            }
            while (index < n && i < size){
               buf[index ++] = buf4[i++];
            }
            if (i == size){
                i = 0;
                size = 0;
            }
            
            
        }
        return index;
    }
}
```

## 146- LRU Cache

```java
 class Node {
    int key; int value;
    Node next; Node pre;
    Node (int key, int value){
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
     
     public void remove(Node node) {
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
    

    
    public Node removeLast() {
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
        if (!map.containsKey(key)) {return -1;}
        Node node = map.get(key);
        int value = node.value;
        put(key, value);
        return value;
    }
    
    public void put(int key, int value) {
        if(map.containsKey(key)){
            Node node = map.get(key);
            cache.remove(node);
            node.value = value;
            map.put(key, node);
            cache.addFirst(node);
            return;
        }
        if (cache.getSize() == capacity){
            Node node = cache.removeLast();
            map.remove(node.key);
        }
        Node nodeAdd = new Node(key, value);
        map.put(key, nodeAdd);
        cache.addFirst(nodeAdd);   
    }
}

```

## 150- Evaluate Reverse Polish Notation

```java
Evaluate the value of an arithmetic expression in Reverse Polish Notation.
Valid operators are +, -, *, /. Each operand may be an integer or another expression.

Example 1:
Input: ["2", "1", "+", "3", "*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9


class Solution {
   public  int evalRPN(String[] tokens) {
        Stack<Integer> st = new Stack<>();
        for (int i = 0; i < tokens.length; i ++) {
            if("+-*/".contains(tokens[i])){
         //   if(tokens[i].equals("+")  || tokens[i].equals("-")...不能是 “==”){
                int num2 = st.pop();
                int num1 = st.pop();
                int cur = getNum(num1, num2, tokens[i]);
                st.push(cur);
            } else {
                st.push(Integer.parseInt(tokens[i]));
            }
        }
        return st.pop();
    }

    public  int getNum(int num1, int num2, String operator){
        switch(operator){
            case "+":
                return num1 + num2;
            case "-":
                return num1 - num2;
            case "*":
                return num1 * num2;
            case "/":
                return num1 / num2;
            default :
                return 0;
        }
    }
}
```

## 706：Design HashMap

```java
class MyHashMap {
        final ListNode[] nodes = new ListNode[10000];

        public void put(int key, int value) {
            int i = idx(key);
            if (nodes[i] == null)
                nodes[i] = new ListNode(-1, -1);
            ListNode prev = find(nodes[i], key);
            if (prev.next == null)
                prev.next = new ListNode(key, value);
            else prev.next.val = value;
        }

        public int get(int key) {
            int i = idx(key);
            if (nodes[i] == null)
                return -1;
            ListNode node = find(nodes[i], key);
            return node.next == null ? -1 : node.next.val; 
        }

        public void remove(int key) {
            int i = idx(key);
            if (nodes[i] == null) return;
            ListNode prev = find(nodes[i], key);
            if (prev.next == null) return;
            prev.next = prev.next.next;
        }

        int idx(int key) { return key % nodes.length;}

        ListNode find(ListNode bucket, int key) {
            ListNode node = bucket, prev = null;
            while (node != null && node.key != key) {
                prev = node;
                node = node.next;
            }
            return prev;
        }

        class ListNode {
            int key, val;
            ListNode next;

            ListNode(int key, int val) {
                this.key = key;
                this.val = val;
            }
        }
    }
```

## 347 - Top K Frequent Elements

```java
Given a non-empty array of integers, return the k most frequent elements.
Example 1:
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for(int num: nums){
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        PriorityQueue<Map.Entry<Integer,Integer>> pq = 
                   new PriorityQueue<>((a, b) -> a.getValue() - b.getValue());
                   // 求最大的K个元素用小顶堆因为要把最小的不断从堆顶去掉
        for(Map.Entry<Integer,Integer> entry : map.entrySet()){ //not map.EntrySet
            pq.offer(entry);
            if(pq.size() > k){
                pq.poll();
            }
        }
        int[] res = new int[k];
        for(int i = 0; i < res.length; i ++){
            res[i] = pq.poll().getKey(); // getKey() in here not getValue();
        }
        return res;
    }
}
```

## 1047：Remove All Adjacent Duplicates In String

```java
Input: "abbaca"
Output: "ca"
Explanation: 
For example, in "abbaca" we could remove "bb" since the letters are adjacent 
and equal, and this is the only possible move.  The result of this move is 
that the string is "aaca", of which only "aa" is possible, so the 
final string is "ca".
//往回看的经常要用到STACK 类似括号这样
public String removeDuplicates(String S) {
        Stack<Character> stack = new Stack<>();
        for(char s : S.toCharArray()){
            if(!stack.isEmpty() && stack.peek() == s)
                stack.pop();
            else
                stack.push(s);
        }
        StringBuilder sb = new StringBuilder();
        for(char s : stack) sb.append(s);
        return sb.toString();
  }
    
```

## 253 - Meeting Rooms II

```java
Given an array of meeting time intervals consisting of start and 
end times [[s1,e1],[s2,e2],...] (si < ei), find the minimum number of 
conference rooms required. 是不是最后一个输入的时候 当前需要多少房间。
Example
Input: [[0, 30],[5, 10],[15, 20]]
Output: 2

//优先队列
class Solution {
    public int minMeetingRooms(int[][] intervals) {
        Arrays.sort(intervals, (a,b) -> a[0] - b[0]);
        PriorityQueue<int[]> pq = new PriorityQueue<>((a,b) -> a[1] - b[1]);
        for(int i = 0; i < intervals.length; i ++) {
            int[] top = pq.peek();
            if(!pq.isEmpty() && intervals[i][0] >= top[1]){
                pq.poll();
            }
            pq.offer(intervals[i]);
        }
        return pq.size();
    }
}
// 扫描线 
//在会议开始和结束的地方会改变状态


```

## 23- Merge k Sorted Lists

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        
       // if(lists == null || lists.length == 0) return null;
        PriorityQueue<ListNode> pq = new PriorityQueue<>((l1,l2)-> l1.val - l2.val);
        ListNode head = new ListNode(-1);
        ListNode cur = head;
        for(int i = 0; i < lists.length; i ++){
            if(lists[i] != null){
             pq.offer(lists[i]);   
            }
        }
        while(!pq.isEmpty()){
            ListNode temp = pq.poll();
            cur.next = temp;
            cur = cur.next;
            if(temp.next != null){
                pq.offer(temp.next);
            }
        }
        return head.next;
    }
}
```

## 73-Set Matrix Zeroes

```java
Given an m x n matrix. If an element is 0, set its entire row and column to 0. 
Do it in-place.
Follow up:
A straight forward solution using O(mn) space is probably a bad idea.
A simple improvement uses O(m + n) space, but still not the best solution.
Could you devise a constant space solution?

思路:
思路一: 用 O(m+n)额外空间
两遍扫matrix,第一遍用集合记录哪些行,哪些列有0;第二遍置0
思路二: 用O(1)空间
关键思想: 用matrix第一行和第一列记录该行该列是否有0,作为标志位
但是对于第一行,和第一列要设置一个标志位,为了防止自己这一行(一列)也有0的情况

class Solution {
    public void setZeroes(int[][] matrix) {
        int row = matrix.length;
        int col = matrix[0].length;
        boolean row0_flag = false;
        boolean col0_flag = false;
        // 第一行是否有零
        for (int j = 0; j < col; j++) {
            if (matrix[0][j] == 0) {
                row0_flag = true;
                break;
            }
        }
        // 第一列是否有零
        for (int i = 0; i < row; i++) {
            if (matrix[i][0] == 0) {
                col0_flag = true;
                break;
            }
        }
        // 把第一行第一列作为标志位
        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = matrix[0][j] = 0;
                }
            }
        }
        // 置0
        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (row0_flag) {
            for (int j = 0; j < col; j++) {
                matrix[0][j] = 0;
            }
        }
        if (col0_flag) {
            for (int i = 0; i < row; i++) {
                matrix[i][0] = 0;
            }
        } 
    }
}

//简化版
class Solution {
    public void setZeroes(int[][] matrix) {
        boolean col0_flag = false;
        int row = matrix.length;
        int col = matrix[0].length;
        for (int i = 0; i < row; i++) {
            if (matrix[i][0] == 0) col0_flag = true;
            for (int j = 1; j < col; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = matrix[0][j] = 0;
                }
            }
        }
        for (int i = row - 1; i >= 0; i--) {
            for (int j = col - 1; j >= 1; j--) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
            if (col0_flag) matrix[i][0] = 0;
        }
    }
}


```

## 1242- Web Crawler Multithreaded

```java

```

## 236-Lowest Common Ancestor of a Binary Tree

```java
class Solution {
   public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if( root == p || root == q || root == null)
            return root;
        TreeNode left = lowestCommonAncestor( root.left,  p,  q);
        TreeNode right = lowestCommonAncestor( root.right,  p,  q);
        if(left == null)
            return right;
        else if (right == null)
            return left;
        else
            return root;
    }
}
```

## 380-Insert Delete GetRandom O\(1\)

```java
public class RandomizedSet {
    ArrayList<Integer> nums;
    HashMap<Integer, Integer> locs;
    java.util.Random rand = new java.util.Random();
    /** Initialize your data structure here. */
    public RandomizedSet() {
        nums = new ArrayList<Integer>();
        locs = new HashMap<Integer, Integer>();
    }
    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
        boolean contain = locs.containsKey(val);
        if ( contain ) return false;
        locs.put( val, nums.size());
        nums.add(val);
        return true;
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
        boolean contain = locs.containsKey(val);
        if ( ! contain ) return false;
        int loc = locs.get(val);
        if (loc < nums.size() - 1 ) { // not the last one than swap the last one with this val
            int lastone = nums.get(nums.size() - 1 );
            nums.set( loc , lastone );
            locs.put(lastone, loc);
        }
        locs.remove(val);
        nums.remove(nums.size() - 1);
        return true;
    }
    
    /** Get a random element from the set. */
    public int getRandom() {
        return nums.get( rand.nextInt(nums.size()) );
    }
}
```

## 93-Restore IP Addresses

```java
Given a string s containing only digits, return all possible valid IP addresses 
that can be obtained from s. You can return them in any order.
A valid IP address consists of exactly four integers, each integer is between 0
 and 255, separated by single dots and cannot have leading zeros. 
 For example, "0.1.2.201" and "192.168.1.1" are valid IP addresses 
 and "0.011.255.245", "192.168.1.312" and "192.168@1.1" are invalid IP addresses. 
Example 1:
Input: s = "25525511135"
Output: ["255.255.11.135","255.255.111.35"]

public List<String> restoreIpAddresses(String s) {
        List<String> result = new ArrayList<>();
        StringBuilder ip = new StringBuilder();

        for (int a = 1; a < 4; a++) {
            for (int b = 1; b < 4; b++) {
                for (int c = 1; c < 4; c++) {
                    for (int d = 1; d < 4; d++) {
                        /*
                         * 1、保障下面subString不会越界
                         * 2、保障截取的字符串与输入字符串长度相同
                         * //1、2比较好理解，3比较有意思
                         * 3、不能保障截取的字符串转成int后与输入字符串长度相同
                         * 如：字符串010010，a=1，b=1，c=1，d=3，对应字符串0，1，0，010
                         * 转成int后seg1=0，seg2=1，seg3=0，seg4=10
                         * //所以需要下面这处判断if (ip.length() == s.length() + 3)
                         */
            if (a + b + c + d == s.length()) {
                int seg1 = Integer.parseInt(s.substring(0, a));
                int seg2 = Integer.parseInt(s.substring(a, a + b));
                int seg3 = Integer.parseInt(s.substring(a + b, a + b + c));
                int seg4 = Integer.parseInt(s.substring(a + b + c, a + b + c + d));
                // 四个段数值满足0~255
                if (seg1 <= 255 && seg2 <= 255 && seg3 <= 255 && seg4 <= 255) {
                        ip.append(seg1).append(".").append(seg2).append(".").
                                append(seg3).append(".").append(seg4);
                                // 保障截取的字符串转成int后与输入字符串长度相同
                                if (ip.length() == s.length() + 3) {
                                    result.add(ip.toString());
                                }
                                ip.delete(0, ip.length());
                            }
                        }
                    }
                }
            }
        }
        return result;
    }
    
    // DFS
这道题也是一个用DFS找所有的可能性的问题。这样一串数字：25525511135
我们可以对他进行切分，但是根据IP的性质，有些切分就是明显不可能的：
比如011.11.11.11, 这个的问题是以0开头了，不合法，直接跳过。
比如257.11.11.11, 257大于IP里的最大数 255了，不合法，直接跳过。
然后我们把这一层切分后剩下的字符串传到下一层，继续去找。
直到最后我们得到4个section为止，把这个结果存到我们的result list。

public List<String> restoreIpAddresses(String s) {
    List<String> list = new ArrayList();
    if(s.length() > 12) return list;
      
    helper(s, list, 0, "", 0);  
    return list;
  }
  
  void helper(String s, List<String> list, int pos, String res, int sec){
    if(sec == 4 && pos == s.length()) {
      list.add(res);
      return;
    }  
      
    for(int i = 1; i <= 3; i++){
      if(pos + i > s.length()) break;  
      String section = s.substring(pos, pos + i);
      if(section.length() > 1 && section.startsWith("0") ||
       Integer.parseInt(section) >= 256)  break;
      helper(s, list, pos + i, sec == 0 ? section : res + "." + section, sec + 1);
    }  
  }
```

## 362- Design Hit Counter

```java
Design a hit counter which counts the number of hits received in the past 5 minutes.
Each function accepts a timestamp parameter (in seconds granularity) 
and you may assume that calls are being made to the system in chronological 
order (ie, the timestamp is monotonically increasing). You may assume that 
the earliest timestamp starts at 1.
It is possible that several hits arrive roughly at the same time.
Example:
HitCounter counter = new HitCounter();
// hit at timestamp 1.
counter.hit(1);
// hit at timestamp 2.
counter.hit(2);
// hit at timestamp 3.
counter.hit(3);
// get hits at timestamp 4, should return 3.
counter.getHits(4);
// hit at timestamp 300.
counter.hit(300);
// get hits at timestamp 300, should return 4.
counter.getHits(300);
// get hits at timestamp 301, should return 3.
counter.getHits(301); 

public class HitCounter {
    private int[] times;
    private int[] hits;
    /** Initialize your data structure here. */
    public HitCounter() {
        times = new int[300];
        hits = new int[300];
    }
    
    /** Record a hit.
        @param timestamp - The current timestamp (in seconds granularity). */
    public void hit(int timestamp) {
        int index = timestamp % 300;
        if (times[index] != timestamp) {
            times[index] = timestamp;
            hits[index] = 1;
        } else {
            hits[index]++;
        }
    }
    
    /** Return the number of hits in the past 5 minutes.
        @param timestamp - The current timestamp (in seconds granularity). */
    public int getHits(int timestamp) {
        int total = 0;
        for (int i = 0; i < 300; i++) {
            if (timestamp - times[i] < 300) {
                total += hits[i];
            }
        }
        return total;
    }
}
```

## 394-Decode String

```java
Example 1:
Input: s = "3[a]2[bc]"
Output: "aaabcbc"

Example 2:
Input: s = "3[a2[c]]"
Output: "accaccacc"

public class Solution {
    private int pos = 0;
    public String decodeString(String s) {
        StringBuilder sb = new StringBuilder();
        String num = "";
        for (int i = pos; i < s.length(); i++) {
            if (s.charAt(i) != '[' && s.charAt(i) != ']' && !Character.isDigit(s.charAt(i))) {
                sb.append(s.charAt(i));
            } else if (Character.isDigit(s.charAt(i))) {
                num += s.charAt(i);
            } else if (s.charAt(i) == '[') {
                pos = i + 1;
                String next = decodeString(s);
                for (int n = Integer.valueOf(num); n > 0; n--) sb.append(next);
                num = "";
                i = pos;
            } else if (s.charAt(i) == ']') {
                pos = i;
                return sb.toString();
            }
        }
        return sb.toString();
    }
}
```

## 37-Sudoku Solver

```java
Write a program to solve a Sudoku puzzle by filling the empty cells.
A sudoku solution must satisfy all of the following rules:
Each of the digits 1-9 must occur exactly once in each row.
Each of the digits 1-9 must occur exactly once in each column.
Each of the digits 1-9 must occur exactly once in each of 
the 9 3x3 sub-boxes of the grid.
The '.' character indicates empty cells.

class Solution {
    public void solveSudoku(char[][] board) {
        if(board == null || board.length == 0) 
            return;
        solve(board);
    }
    
    public boolean solve(char[][] board){
        for(int i = 0; i <board.length; i ++){
            for(int j = 0; j <board[0].length; j++){
                if(board[i][j] == '.'){
                    for(char c = '1'; c <= '9'; c++){
                        if(isValid(board,i,j,c)){
                            board[i][j]= c;
                            if(solve(board))
                                return true;
                            else 
                                board[i][j] = '.';
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }
    
    private boolean isValid(char[][] board, int row, int col, char c) {
        for (int i = 0; i < 9; i ++) {
            if(board[i][col] != '.' && board[i][col] == c) return false; 
            // check row
            if(board[row][i]!= '.' && board[row][i]== c) return false; 
            // check column
            if(board[3 * (row/3) + i/3][3*(col/3) + i % 3] != '.' 
            && board[3 * (row/3) + i/3][3*(col/3) + i%3] == c) return false;
        }
        return true;
    }
}

```

## 692- Top K Frequent Words

```java
Input: ["i", "love", "leetcode", "i", "love", "coding"], k = 2
Output: ["i", "love"]
Explanation: "i" and "love" are the two most frequent words.
Note that "i" comes before "love" due to a lower alphabetical order.

class Solution {
    public List<String> topKFrequent(String[] words, int k) {
        
        List<String> result = new LinkedList<>();
        Map<String, Integer> map = new HashMap<>();
        for(int i=0; i<words.length; i++)
        {
            if(map.containsKey(words[i]))
                map.put(words[i], map.get(words[i])+1);
            else
                map.put(words[i], 1);
        }
        
        PriorityQueue<Map.Entry<String, Integer>> pq = new PriorityQueue<>(
                 (a,b) -> a.getValue()==b.getValue() ?
                  b.getKey().compareTo(a.getKey()) : a.getValue()-b.getValue()
        );
        
        for(Map.Entry<String, Integer> entry: map.entrySet())
        {
            pq.offer(entry);
            if(pq.size()>k)
                pq.poll();
        }

        while(!pq.isEmpty())
            result.add(0, pq.poll().getKey());
        
        return result;
    }
}
```

## 273-Integer to English Words

```java
Input: num = 123
Output: "One Hundred Twenty Three"
class Solution {
    public String numberToWords(int num) {
        if(num == 0) return "Zero";
        return helper(num);  
    }
    
    public String helper(int num ) {
        String[] words = new String[] {"", "One", "Two", "Three", "Four", "Five",
         "Six", "Seven", "Eight", "Nine", "Ten",
   "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen",
        "Eighteen", "Nineteen"}; // Fifteen, Twelve Forty Nineteen Ninety Hundred
        String[] words1 = new String[]{"","","Twenty ", "Thirty ", 
        "Forty ", "Fifty ", "Sixty ",  "Seventy ", "Eighty ", "Ninety "};
       StringBuilder sb = new StringBuilder();
        if (num >= 1000000000) {
            sb.append(helper(num/1000000000)).append(" Billion ");
            num %= 1000000000; 
        }
        if (num >= 1000000) {
            sb.append(helper(num/1000000)).append(" Million ");
            num %= 1000000; 
        }
        if (num >= 1000) {
            sb.append(helper(num/1000)).append(" Thousand ");
            num %= 1000; 
        }
        if (num >= 100) {
            sb.append(helper(num/100)).append(" Hundred ");
            num %= 100; 
        }
        if (num >= 20) {
             sb.append(words1[num/10]).append(words[num%10]);
        } else {
          sb.append(words[num]);  
        }
        
        return sb.toString().trim();
        
    }
}
```

## 

## 

## 




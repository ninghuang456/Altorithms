# Oracle 1 ~ 60

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

## 

## 




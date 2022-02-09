# Linkedin 1 \~ 50

## 244 Shortest Word Distance II

```
Input
["WordDistance", "shortest", "shortest"]
[[["practice", "makes", "perfect", "coding", "makes"]], ["coding", "practice"], ["makes", "coding"]]
Output
[null, 3, 1]

Explanation
WordDistance wordDistance = new WordDistance(["practice", "makes", "perfect", "coding", "makes"]);
wordDistance.shortest("coding", "practice"); // return 3
wordDistance.shortest("makes", "coding");    // return 1

class WordDistance {
    
    private Map<String, List<Integer>> map;

    public WordDistance(String[] words) {
        // 构造器中先把单词（可能有重复）的各自下标进行预处理
        map = new HashMap<String, List<Integer>>();
        for(int i = 0; i < words.length; i++) {
            String word = words[i];
            if(map.containsKey(word)) {
                map.get(word).add(i);
            } else {
                List<Integer> list = new ArrayList<Integer>();
                list.add(i);
                map.put(word, list);
            }
        }
    }

    public int shortest(String word1, String word2) {
        // 对比两个list之间各元素的最小差值
        List<Integer> list1 = map.get(word1);
        List<Integer> list2 = map.get(word2);
        int distance = Integer.MAX_VALUE;
        // list 1 and list 2 already sorted!
        for(int i = 0, j = 0; i < list1.size() && j < list2.size(); ) {
            int index1 = list1.get(i), index2 = list2.get(j);
            //distance = Math.min(distance, Math.abs(index1 - index2));
            if(index1 < index2) {
                distance = Math.min(distance, index2 - index1);
                i++;
            } else {
                distance = Math.min(distance, index1 - index2);
                j++;
            }
        }
        return distance;
    }
}
```

## 272: Closest Binary Search Tree Value II

```java
Given the root of a binary search tree, a target value, and an integer k, 
return the k values in the BST that are closest to the target. 
You may return the answer in any order.

Input: root = [4,2,5,1,3], target = 3.714286, k = 2
Output: [4,3]

class Solution {
  public List<Integer> closestKValues(TreeNode root, double target, int k) {
  List<Integer> res = new ArrayList<>();

  Stack<Integer> s1 = new Stack<>(); // predecessors
  Stack<Integer> s2 = new Stack<>(); // successors

  inorder(root, target, false, s1);
  inorder(root, target, true, s2);
  
  while (k-- > 0) {
    if (s1.isEmpty())
      res.add(s2.pop());
    else if (s2.isEmpty())
      res.add(s1.pop());
    else if (Math.abs(s1.peek() - target) < Math.abs(s2.peek() - target))
      res.add(s1.pop());
    else
      res.add(s2.pop());
  }
  
  return res;
}

// inorder traversal  inorder traversal gives us sorted predecessors, 
whereas reverse-inorder traversal gives us sorted successors.
void inorder(TreeNode root, double target, boolean reverse, 
                                           Stack<Integer> stack) {
  if (root == null) return;

  inorder(reverse ? root.right : root.left, target, reverse, stack);
  // early terminate, no need to traverse the whole tree
  if ((reverse && root.val <= target) || (!reverse && root.val > target)) return;
  // track the value of current node
  stack.push(root.val);
  inorder(reverse ? root.left : root.right, target, reverse, stack);
}
}


```

## 364 Nested List Weight Sum II



You are given a nested list of integers `nestedList`. Each element is either an integer or a list whose elements may also be integers or other lists.

The **depth** of an integer is the number of lists that it is inside of. For example, the nested list `[1,[2,2],[[3],2],1]` has each integer's value set to its **depth**. Let `maxDepth` be the **maximum depth** of any integer.

The **weight** of an integer is `maxDepth - (the depth of the integer) + 1`.

Return _the sum of each integer in_ `nestedList` _multiplied by its **weight**_.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/03/27/nestedlistweightsumiiex1.png)

```java
Input: nestedList = [[1,1],2,[1,1]]
Output: 8
Explanation: Four 1's with a weight of 1, one 2 with a weight of 2.
1*1 + 1*1 + 2*2 + 1*1 + 1*1 = 8

class Solution {
   public int depthSumInverse(List<NestedInteger> nestedList) {
        if (nestedList == null) return 0;
        Queue<NestedInteger> queue = new LinkedList<NestedInteger>();
        int prev = 0;
        int total = 0;
        for (NestedInteger next: nestedList) {
            queue.offer(next);
        }
        
        while (!queue.isEmpty()) {
            int size = queue.size();
            int levelSum = 0;
            for (int i = 0; i < size; i++) {
                NestedInteger current = queue.poll();
                if (current.isInteger()) levelSum += current.getInteger();
                List<NestedInteger> nextList = current.getList();
                if (nextList != null) {
                    for (NestedInteger next: nextList) {
                        queue.offer(next);
                    }
                }
            }
            prev += levelSum;
            total += prev;
        }
        return total;
    }
}

```

## 339 Nested List Weight Sum

&#x20;

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/01/14/nestedlistweightsumex1.png)

```java
You are given a nested list of integers nestedList. 
Each element is either an integer or a list
 whose elements may also be integers or other lists.
The depth of an integer is the number of lists that it is inside of. 
For example, the nested list [1,[2,2],[[3],2],1] 
has each integer's value set to its depth.
Return the sum of each integer in nestedList multiplied by its depth.

class Solution {
    int result;
    public int depthSum(List<NestedInteger> nestedList) {
        result = 0;
        dfs(nestedList, 1);
        return result;
    }
    private void dfs(List<NestedInteger> nestedList, int depth) {
        for (NestedInteger ni : nestedList) {
            if (ni.isInteger()) {
                result += ni.getInteger() * depth;
            } else {
                dfs(ni.getList(), depth + 1);
            }
        }
    }
}java

```

## 156 Binary Tree Upside Down



Given the `root` of a binary tree, turn the tree upside down and return _the new root_.

You can turn a binary tree upside down with the following steps:

1. The original left child becomes the new root.
2. The original root becomes the new right child.
3. The original right child becomes the new left child.

![](https://assets.leetcode.com/uploads/2020/08/29/main.jpg)

The mentioned steps are done level by level. It is **guaranteed** that every right node has a sibling (a left node with the same parent) and has no children.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/08/29/updown.jpg)

```java
Input: root = [1,2,3,4,5]
Output: [4,5,2,null,null,3,1]
// solution 1:
public TreeNode upsideDownBinaryTree(TreeNode root) {
    if(root == null || root.left == null) {
        return root;
    }
    
    TreeNode newRoot = upsideDownBinaryTree(root.left);
    root.left.left = root.right;   // node 2 left children
    root.left.right = root;         // node 2 right children
    root.left = null;
    root.right = null;
    return newRoot;
}

// solution 2:
public TreeNode upsideDownBinaryTree(TreeNode root) {
    TreeNode curr = root;
    TreeNode next = null;
    TreeNode temp = null;
    TreeNode prev = null;
    
    while(curr != null) {
        next = curr.left;
        // swapping nodes now, need temp to keep the previous right child
        curr.left = temp;
        temp = curr.right;
        curr.right = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}  

```

## 360 Sort Transformed Array

![](.gitbook/assets/image.png)

```
Given a sorted integer array nums and three integers a, b and c, 
apply a quadratic function of the 
form f(x) = ax2 + bx + c to each element nums[i] in the array, 
and return the array in a sorted order.
Example 1:
Input: nums = [-4,-2,2,4], a = 1, b = 3, c = 5
Output: [3,9,15,33]
Example 2:
Input: nums = [-4,-2,2,4], a = -1, b = 3, c = 5
Output: [-23,-5,1,7]

顶点： X = -b / 2a; Y = f(x);
如果“a”(x2的系数)是正的，那么抛物线的开口就是朝上的，反之就是朝下的。 
也就是把开口朝上的抛物线上下颠倒。


public class Solution {
    public int[] sortTransformedArray(int[] nums, int a, int b, int c) {
        int n = nums.length;
        int[] sorted = new int[n];
        int i = 0, j = n - 1;
        int index = a >= 0 ? n - 1 : 0;
        while (i <= j) {
            if (a >= 0) {
                sorted[index--] = 
                quad(nums[i], a, b, c) >= quad(nums[j], a, b, c) ? 
                quad(nums[i++], a, b, c) : quad(nums[j--], a, b, c);
            } else {
                sorted[index++] = 
                quad(nums[i], a, b, c) >= quad(nums[j], a, b, c) ? 
                quad(nums[j--], a, b, c) : quad(nums[i++], a, b, c);
            }
        }
        return sorted;
    }
    
    private int quad(int x, int a, int b, int c) {
        return a * x * x + b * x + c;
    }
}

class Solution {
    public int[] sortTransformedArray(int[] nums, int a, int b, int c) {
        if (nums.length == 0 || nums == null)
            return new int[0];
        int n = nums.length;
        int[] res = new int[n];
        if (a == 0) {
            for (int i = 0; i < n; i++) {
                int cur = b >= 0 ? nums[i] : nums[n - 1 - i];
                res[i] = b * cur + c;
            }
            return res;
        }
        //sort based on distance to pivot
        double pivot = (double) -b / (2 * a);
        int[] distSorted = new int[n];
        int lo = 0, hi = n - 1, end = n - 1;
        while (lo <= hi) { 
            double d1 = pivot - nums[lo], d2 = nums[hi] - pivot;
            if (d1 > d2) {
                distSorted[end--] = nums[lo++];
            } else {
                distSorted[end--] = nums[hi--];
            }
        }
        //populate res based on distSorted, and also a
        for (int i = 0; i < n; i++) {
            int cur = a > 0 ? distSorted[i] : distSorted[n - 1 - i];
            res[i] = a * cur * cur + b * cur + c;
        }
        return res;
    }
}
```

## 297: Serialize and Deserialize Binary Tree

```java
public class Codec {
    // Encodes a tree to a single string.
    public String spliter = ",";
    public String nulval = "null";
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        if(root == null) return sb.append(nulval).toString();
        serializeHelper(root,sb);
        return sb.toString();
    }
    
    public void serializeHelper(TreeNode node, StringBuilder sb) {
        if(node == null) {
            sb.append(nulval).append(spliter);
            return;
        }
        sb.append(node.val).append(spliter);
        serializeHelper(node.left, sb);
        serializeHelper(node.right, sb);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Queue<String> queue = new LinkedList<>();
        queue.addAll(Arrays.asList(data.split(spliter)));
        return deserializeHelper(queue);
        
    }
    public TreeNode deserializeHelper(Queue<String> queue) {
        String cur = queue.poll();
        if (cur.equals(nulval)) {
            return null;
        } 
        TreeNode node = new TreeNode(Integer.valueOf(cur));
        node.left = deserializeHelper(queue);
        node.right = deserializeHelper(queue);
        return node; 
    }
}
}

```

## 366 Find Leaves of Binary Tree



Given the `root` of a binary tree, collect a tree's nodes as if you were doing this:

* Collect all the leaf nodes.
* Remove all the leaf nodes.
* Repeat until the tree is empty.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/03/16/remleaves-tree.jpg)

```
Input: root = [1,2,3,4,5]
Output: [[4,5,3],[2],[1]]
Explanation:
[[3,5,4],[2],[1]] and [[3,4,5],[2],[1]] are also considered correct answers since per each level it does not matter the order on which elements are returned.
```

```
class Solution {
    
    List<List<Integer>> res = new ArrayList<>();
    HashMap<Integer, List<Integer>> map = new HashMap<>();
    
    public List<List<Integer>> findLeaves(TreeNode root) {
        if(root == null) return res;
        findMaxDistance(root);
        for(int i = 0; i < map.size(); i ++){
            res.add(map.get(i));
        }
        return res;
    }
    
    public int findMaxDistance(TreeNode root){
        if(root == null) return -1;
        int left = findMaxDistance(root.left);
        int right = findMaxDistance(root.right);
        int cur = Math.max(left, right) + 1;
        if(!map.containsKey(cur)){
            List<Integer> curLevel = new ArrayList<>();
            curLevel.add(root.val);
            map.put(cur, curLevel);
        } else {
            map.get(cur).add(root.val);
        }
        
        return cur;
        
    }
}
```

## 380 Insert Delete GetRandom O(1)

Implement the `RandomizedSet` class:

* `RandomizedSet()` Initializes the `RandomizedSet` object.
* `bool insert(int val)` Inserts an item `val` into the set if not present. Returns `true` if the item was not present, `false` otherwise.
* `bool remove(int val)` Removes an item `val` from the set if present. Returns `true` if the item was present, `false` otherwise.
* `int getRandom()` Returns a random element from the current set of elements (it's guaranteed that at least one element exists when this method is called). Each element must have the **same probability** of being returned.

You must implement the functions of the class such that each function works in **average** `O(1)` time complexity.

**Example 1:**

```
Input
["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
[[], [1], [2], [2], [], [1], [2], []]
Output
[null, true, false, true, 2, true, false, 2]
```

\


```
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

## 2040 Kth Smallest Product of Two Sorted Arrays

Given two **sorted 0-indexed** integer arrays `nums1` and `nums2` as well as an integer `k`, return _the_ `kth` _(**1-based**) smallest product of_ `nums1[i] * nums2[j]` _where_ `0 <= i < nums1.length` _and_ `0 <= j < nums2.length`.

```
Input: nums1 = [2,5], nums2 = [3,4], k = 2
Output: 8
Explanation: The 2 smallest products are:
- nums1[0] * nums2[0] = 2 * 3 = 6
- nums1[0] * nums2[1] = 2 * 4 = 8
The 2nd smallest product is 8.
```

```java
class Solution {
    static long INF = (long) 1e10;
    public long kthSmallestProduct(int[] nums1, int[] nums2, long k) {
        int m = nums1.length, n = nums2.length;
        long lo = -INF - 1, hi = INF + 1;
        while (lo < hi) {            
            long mid = lo + ((hi - lo) >> 1), cnt = 0;
            for (int i : nums1) {
                if (0 <= i) {
                    int l = 0, r = n - 1, p = 0;
                    while (l <= r) {
                        int c = l + ((r - l) >> 1);
                        long mul = i * (long) nums2[c];
                        if (mul <= mid) {
                            p = c + 1;
                            l = c + 1;
                        } else r = c - 1;
                    }
                    cnt += p;
                } else {
                    int l = 0, r = n - 1, p = 0;
                    while (l <= r) {
                        int c = l + ((r - l) >> 1);
                        long mul = i * (long) nums2[c];
                        if (mul <= mid) {
                            p = n - c;
                            r = c - 1;
                        } else l = c + 1;
                    }
                    cnt += p;
                }
            }
            if (cnt >= k) {
                hi = mid;
            } else lo = mid + 1L;
        }
        return lo;
    }
}
```

## 671 Second Minimum Node In a Binary Tree

Given a non-empty special binary tree consisting of nodes with the non-negative value, where each node in this tree has exactly `two` or `zero` sub-node. If the node has two sub-nodes, then this node's value is the smaller value among its two sub-nodes. More formally, the property `root.val = min(root.left.val, root.right.val)` always holds.

Given such a binary tree, you need to output the **second minimum** value in the set made of all the nodes' value in the whole tree.

If no such second minimum value exists, output -1 instead.

&#x20;

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/10/15/smbt1.jpg)

```java
class Solution {
    // 定义：输入一棵二叉树，返回这棵二叉树中第二小的节点值
    public int findSecondMinimumValue(TreeNode root) {
        if (root.left == null && root.right == null) {
            return -1;
        }
        // 左右子节点中不同于 root.val 的那个值可能是第二小的值
        int left = root.left.val, right = root.right.val;
        // 如果左右子节点都等于 root.val，则去左右子树递归寻找第二小的值
        if (root.val == root.left.val) {
            left = findSecondMinimumValue(root.left);
        }
        if (root.val == root.right.val) {
            right = findSecondMinimumValue(root.right);
        }
        if (left == -1) {
            return right;
        }
        if (right == -1) {
            return left;
        }
        // 如果左右子树都找到一个第二小的值，更小的那个是整棵树的第二小元素
        return Math.min(left, right);
    }
}
```

## 150 Evaluate Reverse Polish Notation



Evaluate the value of an arithmetic expression in [Reverse Polish Notation](http://en.wikipedia.org/wiki/Reverse\_Polish\_notation).

Valid operators are `+`, `-`, `*`, and `/`. Each operand may be an integer or another expression.

**Note** that division between two integers should truncate toward zero.

It is guaranteed that the given RPN expression is always valid. That means the expression would always evaluate to a result, and there will not be any division by zero operation.

```java
Input: tokens = ["2","1","+","3","*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9
ava
class Solution {
   public  int evalRPN(String[] tokens) {
        Stack<Integer> st = new Stack<>();
        for (int i = 0; i < tokens.length; i ++) {
          //  if("+-*/".contains(tokens[i])){
            if(tokens[i].equals("+")  || tokens[i].equals("-")  || tokens[i].equals("*" )  || tokens[i].equals("/")){
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

## 53 Maximum Subarray

Given an integer array `nums`, find the contiguous subarray (containing at least one number) which has the largest sum and return _its sum_.

A **subarray** is a **contiguous** part of an array.

```java
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.

class Solution {
int maxSubArray(int[] nums) {
    int n = nums.length;
    if (n == 0) return 0;
    int[] dp = new int[n];
    // base case
    // 第一个元素前面没有子数组
    dp[0] = nums[0];
    // 状态转移方程
    for (int i = 1; i < n; i++) {
        dp[i] = Math.max(nums[i], nums[i] + dp[i - 1]);
    }
    // 得到 nums 的最大子数组
    int res = Integer.MIN_VALUE;
    for (int i = 0; i < n; i++) {
        res = Math.max(res, dp[i]);
    }
    return res;
  }
}

```


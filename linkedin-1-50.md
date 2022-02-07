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

```

}

# Linkedin 26 \~ 50

## 432 All O\`one Data Structure



Design a data structure to store the strings' count with the ability to return the strings with minimum and maximum counts.

Implement the `AllOne` class:

* `AllOne()` Initializes the object of the data structure.
* `inc(String key)` Increments the count of the string `key` by `1`. If `key` does not exist in the data structure, insert it with count `1`.
* `dec(String key)` Decrements the count of the string `key` by `1`. If the count of `key` is `0` after the decrement, remove it from the data structure. It is guaranteed that `key` exists in the data structure before the decrement.
* `getMaxKey()` Returns one of the keys with the maximal count. If no element exists, return an empty string `""`.
* `getMinKey()` Returns one of the keys with the minimum count. If no element exists, return an empty string `""`.
*

```java
inc：
当节点不存在时，插入到链表末尾，O(1)；
当节点存在时，根据first得到链表中当前值的起始位置，将节点插入到起始位置之前，O(1)。

dec：
如果节点值为1，那么删除节点，O(1)；
如果节点值不为1，那么根据last得到链表中当前值的结束位置，将节点插入到结束位置之后，O(1)。

getMaxKey：
因为链表降序排列，所以第一个节点即是最大，O(1)。

getMinKey：
因为链表降序排列，所以最后一个节点即是最小，O(1)。

public class AllOne {
    class ValueNode {
        ValueNode prev, next;
        int val;
        Set<String> strs;
        ValueNode(int v) {
            val = v;
            strs = new LinkedHashSet<>();
        }
        void insertAt(ValueNode node) {
             this.prev = node.prev;
            this.next = node;
            node.prev.next = this;
            node.prev = this;
        }
        void remove(String str) {
            strs.remove(str);
            if (strs.isEmpty()) {
                prev.next = next;
                next.prev = prev;
            }
        }
    }
    
    ValueNode valueHead, valueTail; // dummy
    Map<String, ValueNode> keys;
    
    /** Initialize your data structure here. */
    public AllOne() {
        valueHead = new ValueNode(0);
        valueTail = new ValueNode(0);
        valueHead.next = valueTail;
        valueTail.prev = valueHead;
        keys = new HashMap<>();
    }
    
    /** Inserts a new key <Key> with value 1. Or increments an
     existing key by 1. */
    public void inc(String key) {
        ValueNode node = keys.getOrDefault(key, valueHead);
        ValueNode vn = node.next;
        if (vn.val != node.val + 1) {
            vn = new ValueNode(node.val + 1);
            vn.insertAt(node.next);
        }
        vn.strs.add(key);
        keys.put(key, vn);
        if (node != valueHead) node.remove(key);
    }
    
    /** Decrements an existing key by 1. If Key's value is 1,
     remove it from the data structure. */
    public void dec(String key) {
        ValueNode node = keys.get(key);
        if (node == null) return;
        if (node.val == 1) {
            keys.remove(key);
            node.remove(key);
            return;
        }
        ValueNode vn = node.prev;
        if (node.prev.val != node.val - 1) {
            vn = new ValueNode(node.val - 1);
            vn.insertAt(node);
        }
        vn.strs.add(key);
        keys.put(key, vn);
        node.remove(key);
    }
    
    /** Returns one of the keys with maximal value. */
    public String getMaxKey() {
        if (valueTail.prev == valueHead) return "";
        return valueTail.prev.strs.iterator().next();
    }
    
    /** Returns one of the keys with Minimal value. */
    public String getMinKey() {
        if (valueHead.next == valueTail) return "";
        return valueHead.next.strs.iterator().next();
    }
}

/**
 * Your AllOne object will be instantiated and called as such:
 * AllOne obj = new AllOne();
 * obj.inc(key);
 * obj.dec(key);
 * String param_3 = obj.getMaxKey();
 * String param_4 = obj.getMinKey();
 */

```

## 68 Text Justification

Given an array of strings `words` and a width `maxWidth`, format the text such that each line has exactly `maxWidth` characters and is fully (left and right) justified.

You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces `' '` when necessary so that each line has exactly `maxWidth` characters.

Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line does not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.

For the last line of text, it should be left-justified and no extra space is inserted between words.

**Note:**

* A word is defined as a character sequence consisting of non-space characters only.
* Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
* The input array `words` contains at least one word.

```
Input: words = ["This", "is", "an", "example", "of", "text", "justification."],
 maxWidth = 16
Output:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]
字符串大模拟，分情况讨论即可：

如果当前行只有一个单词，特殊处理为左对齐；
如果当前行为最后一行，特殊处理为左对齐；
其余为一般情况，分别计算「当前行单词总长度」、「当前行空格总长度」和
「往下取整后的单位空格长度」，然后依次进行拼接。当空格无法均分时，
每次往靠左的间隙多添加一个空格，直到剩余的空格能够被后面的间隙所均分。
class Solution {
    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> ans = new ArrayList<>();
        int n = words.length;
        List<String> list = new ArrayList<>();
        for (int i = 0; i < n; ) {
            // list 装载当前行的所有 word
            list.clear();
            list.add(words[i]);
            int cur = words[i++].length();
            while (i < n && cur + 1 + words[i].length() <= maxWidth) {
                cur += 1 + words[i].length();
                list.add(words[i++]);
            }

            // 当前行为最后一行，特殊处理为左对齐
            if (i == n) {
                StringBuilder sb = new StringBuilder(list.get(0));
                for (int k = 1; k < list.size(); k++) {
                    sb.append(" ").append(list.get(k));
                }
                while (sb.length() < maxWidth) sb.append(" ");
                ans.add(sb.toString());
                break;
            }

            // 如果当前行只有一个 word，特殊处理为左对齐
            int cnt = list.size();
            if (cnt == 1) {
                String str = list.get(0);
                while (str.length() != maxWidth) str += " ";
                ans.add(str);
                continue;
            }

            /**
            * 其余为一般情况
            * wordWidth : 当前行单词总长度;
            * spaceWidth : 当前行空格总长度;
            * spaceItem : 往下取整后的单位空格长度
            */
            int wordWidth = cur - (cnt - 1);
            int spaceWidth = maxWidth - wordWidth;
            int spaceItemWidth = spaceWidth / (cnt - 1);
            String spaceItem = "";
            for (int k = 0; k < spaceItemWidth; k++) spaceItem += " ";
            StringBuilder sb = new StringBuilder();
            for (int k = 0, sum = 0; k < cnt; k++) {
                String item = list.get(k);
                sb.append(item);
                if (k == cnt - 1) break;
                sb.append(spaceItem);
                sum += spaceItemWidth;
                // 剩余的间隙数量（可填入空格的次数）
                int remain = cnt - k - 1 - 1;
   // 剩余间隙数量 * 最小单位空格长度 + 当前空格长度 < 单词总长度，则在当前间隙多补充一个空格
                if (remain * spaceItemWidth + sum < spaceWidth) {
                    sb.append(" ");
                    sum++;
                }
            }
            ans.add(sb.toString());
        }
        return ans;
    }
}

```

## 152 Maximum Product Subarray

Given an integer array `nums`, find a contiguous non-empty subarray within the array that has the largest product, and return _the product_.

The test cases are generated so that the answer will fit in a **32-bit** integer.

A **subarray** is a contiguous subsequence of the array.

```java
Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
这题是求数组中子区间的最大乘积，对于乘法，我们需要注意，负数乘以负数，会变成正数，
所以解这题的时候我们需要维护两个变量，当前的最大值，以及最小值，最小值可能为负数，
但没准下一步乘以一个负数，当前的最大值就变成最小值，而最小值则变成最大值了。
我们的动态方程可能这样：
maxDP[i + 1] = max(maxDP[i] * A[i + 1], A[i + 1],minDP[i] * A[i + 1])
minDP[i + 1] = min(minDP[i] * A[i + 1], A[i + 1],maxDP[i] * A[i + 1])
dp[i + 1] = max(dp[i], maxDP[i + 1])

这里，我们还需要注意元素为0的情况，如果A[i]为0，那么maxDP和minDP都为0，
我们需要从A[i + 1]重新开始。

class Solution {
    public int maxProduct(int[] nums) {
        if(nums.length == 0)
            return 0;
        int ans = nums[0];
        //两个mDP分别定义为以i结尾的子数组的最大积与最小积；
        int[] maxDP = new int[nums.length];
        int[] minDP = new int[nums.length];
        //初始化DP；
        maxDP[0] = nums[0]; minDP[0] = nums[0];

        for(int i = 1; i < nums.length; i++){
            //最大积的可能情况有：元素i自己本身，上一个最大积与i元素累乘，
            //上一个最小积与i元素累乘；
            //与i元素自己进行比较是为了处理i元素之前全都是0的情况；
            maxDP[i] = Math.max(nums[i], Math.max(maxDP[i-1]*nums[i], 
                                  minDP[i-1]*nums[i]));
            minDP[i] = Math.min(nums[i], Math.min(maxDP[i-1]*nums[i], 
                                  minDP[i-1]*nums[i]));
            //记录ans；
            ans = Math.max(ans, maxDP[i]);
        }
        return ans;
    }
}

```

## 256 Factor Combinations

Numbers can be regarded as the product of their factors.

* For example, `8 = 2 x 2 x 2 = 2 x 4`.

Given an integer `n`, return _all possible combinations of its factors_. You may return the answer in **any order**.

**Note** that the factors should be in the range `[2, n - 1]`.

```java
Input: n = 12
Output: [[2,6],[3,4],[2,2,3]]
dfs(num)

遍历数字1~num，找到能被自己整除的因子mulNum，那么[mulNum, num/mulNum]就是一种结果，
 并往下继续dfs(num/mulNum)得到num/mulNum的可能情况并添加到返回结果。
剪枝点：
为了避免重复，没必要从1开始遍历，而是从上一次的mulNum开始遍历，
这样保证mulNum后续dfs的过程是递增的，所以不会出现重复。
遍历终点没必要为num， 而是num的开根号， 
因此最大情况2^32的开根号结果为2^16次方=65536，是可接受范围。

class Solution {
    public List<List<Integer>> getFactors(int n) {
        return dfs(2,n);
    }

    List<List<Integer>> dfs(int start, int num) {
        if (num == 1) {
            return new ArrayList<>();
        }

        int qNum = (int)Math.sqrt(num); // 算了 2，6 没必要算 6 2 了
        List<List<Integer>> result = new ArrayList<>();
        for (int mulNum = start; mulNum <= qNum;mulNum++) {
            if (num % mulNum == 0) { //这里还是用 num去除
                List<Integer> simpleList = new ArrayList<>();
                simpleList.add(mulNum);
                simpleList.add(num/mulNum);
                result.add(simpleList);
                // 检查mulNum能怎么拆
                List<List<Integer>> nextLists = dfs(mulNum, num/mulNum);
                for (List<Integer> list : nextLists) {
                    list.add(mulNum);
                    result.add(list);
                }          
            }
        }
        return result;
    }

}

class Solution {
    public List<List<Integer>> getFactors(int n) {
    List<List<Integer>> result = new ArrayList<List<Integer>>();
    helper(result, new ArrayList<Integer>(), n, 2);
    return result;
}

public void helper(List<List<Integer>> result, List<Integer> item, int n, 
                                     int start){
    if (n <= 1) {
        if (item.size() > 1) {
            result.add(new ArrayList<Integer>(item));
        }
        return;
    }
    
    for (int i = start; i <= n; ++i) {
        if (n % i == 0) {
            item.add(i);
            helper(result, item, n/i, i);
            item.remove(item.size()-1);
        }
    }
}
}
```

## 373 Find K Pairs with Smallest Sums

You are given two integer arrays `nums1` and `nums2` sorted in **ascending order** and an integer `k`.

Define a pair `(u, v)` which consists of one element from the first array and one element from the second array.

Return _the_ `k` _pairs_ `(u1, v1), (u2, v2), ..., (uk, vk)` _with the smallest sums_.



Basic idea: Use min\_heap to keep track on next minimum pair sum, and we only need to maintain K possible candidates in the data structure.

Some observations: For every numbers in nums1, its best partner(yields min sum) always strats from nums2\[0] since arrays are all sorted; And for a specific number in nums1, its next candidate sould be **\[this specific number]** + **nums2\[current\_associated\_index + 1]**, unless out of boundary;)

Here is a simple example demonstrate how this algorithm works.

![image](https://cloud.githubusercontent.com/assets/8743900/17332795/0bb46cfe-589e-11e6-90b5-5d3c9696c4f0.png)

```
Input: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
Output: [[1,2],[1,4],[1,6]]
Explanation: The first 3 pairs are returned from the sequence: 
[1,2],[1,4],[1,6],[7,2],[7,4],[11,2],[7,6],[11,4],[11,6]
多路归并

class Solution {
    boolean flag = true;
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        int n = nums1.length, m = nums2.length;
        if (n > m && !(flag = false)) return kSmallestPairs(nums2, nums1, k);
        List<List<Integer>> ans = new ArrayList<>();
        PriorityQueue<int[]> q = new PriorityQueue<>((a,b)->
                          (nums1[a[0]]+nums2[a[1]])-(nums1[b[0]]+nums2[b[1]]));
        for (int i = 0; i < Math.min(n, k); i++) 
               q.add(new int[]{i, 0});
        while (ans.size() < k && !q.isEmpty()) {
            int[] poll = q.poll();
            int a = poll[0], b = poll[1];
            ans.add(new ArrayList<>(){{
                add(flag ? nums1[a] : nums2[b]);
                add(flag ? nums2[b] : nums1[a]);
            }});
            if (b + 1 < m) q.add(new int[]{a, b + 1});
        }
        return ans;
    }
}

```

## 149 Max Points on a Line

Given an array of `points` where `points[i] = [xi, yi]` represents a point on the **X-Y** plane, return _the maximum number of points that lie on the same straight line_.

```java
Input: points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
Output: 4
//平行线有可能斜率一样
我们知道，两个点可以确定一条线。
直接计算每个点与其他所有点的斜率，转换成double防止精度损失，
将计算好的斜率加到map里面计数，需要注意的是每找完一个点就要更新全部map再清空map，
如果是所有点都统计完再去更新map将会导致重复计算，
将最后的结果+1表示自身的那个点（因为map里面都是相同斜率的数量）


class Solution {
    public int maxPoints(int[][] points) {
        if (points.length == 1) {
            return 1;
        }
        Map<Double, Integer> map = new HashMap<>();
        int max = Integer.MIN_VALUE;
        for (int[] point1 : points) {
            for (int[] point2 : points) {
double k = (double) (point2[1] - point1[1]) / (double) (point2[0] - point1[0]);
                map.put(k, map.getOrDefault(k, 0) + 1);
                max = Math.max(max, map.get(k));
            }
           
            map.clear();
        }
        return max + 1;
    }
}






```

## 678 Valid Parenthesis String

Given a string `s` containing only three types of characters: `'('`, `')'` and `'*'`, return `true` _if_ `s` _is **valid**_.

The following rules define a **valid** string:

* Any left parenthesis `'('` must have a corresponding right parenthesis `')'`.
* Any right parenthesis `')'` must have a corresponding left parenthesis `'('`.
* Left parenthesis `'('` must go before the corresponding right parenthesis `')'`.
* `'*'` could be treated as a single right parenthesis `')'` or a single left parenthesis `'('` or an empty string `""`.

```java
这道题和20.有效的括号其实很像，但里面多了一个星号，让这个题目不再这么直观。
我们很自然的会仍然用两个栈记录目前为止不能匹配的字符，也就是*和(，
每次出现右括号我们就应该去两个栈匹配可以匹配的左括号。

而*可以作为任何括号，也可以作为空字符串，所以我们应该优先用左括号匹配，所以我们出栈的策略如下：

遇到左括号，直接进栈，记录括号的位置。
遇到星号，直接进栈，记录星号的位置。
遇到右括号：
a: 左括号栈里有元素，直接出栈。
b: 左括号栈里无元素，*栈里有元素，直接出栈。无元素的话就已经匹配失败了。
如果遍历完数组的话，我们可能会发现左括号栈里还有结余元素。如果是20题的情况，已经失败了。
但现在我们可能还有一些星号可以作为右括号用，所以我们进行下面的匹配操作：
对左括号栈逐一出栈，然后去看此时星号栈的栈顶，如果栈顶元素的位置大于左括号栈顶元素的位置，
说明星号在括号的右侧，可以匹配。否则不可。

class Solution {
    public boolean checkValidString(String s) {
    //使用两个栈保存（和*
    Stack<Integer> stack1 = new Stack<>();
    Stack<Integer> stack2 = new Stack<>();
    for(int i=0;i<s.length();i++){
        char c = s.charAt(i);
        //入栈操作
        if(c=='(') stack1.push(i);
        else if(c=='*') stack2.push(i);
        // 出栈：优先出stack1
        else{
            if(!stack1.isEmpty()){
                stack1.pop();
            }
            else if(!stack2.isEmpty())
            {
                stack2.pop();
            }
            else{
                return false;
            }
        }
    }
    // 当右括号栈存在元素时，对左括号栈逐一出栈，然后去看此时星号栈的栈顶，
    // 如果栈顶元素的位置大于左括号栈顶元素的位置，说明星号在括号的右侧，可以匹配。否则不可。
    while(!stack1.isEmpty()){
        if(stack2.isEmpty()) return false;
        int posStack1 = stack1.pop();
        int posStack2 = stack2.pop();
        if(posStack1>posStack2){
            return false;
        }
    }
    return true;    
  }          
}

```

## 605 Can Place Flowers

You have a long flowerbed in which some of the plots are planted, and some are not. However, flowers cannot be planted in **adjacent** plots.

Given an integer array `flowerbed` containing `0`'s and `1`'s, where `0` means empty and `1` means not empty, and an integer `n`, return _if_ `n` new flowers can be planted in the `flowerbed` without violating the no-adjacent-flowers rule.

```
从左向右遍历花坛，在可以种花的地方就种一朵，能种就种（因为在任一种花时候，不种都不会得到更优解）
，就是一种贪心的思想
这里可以种花的条件是：
自己为空
左边为空 或者 自己是最左
右边为空 或者 自己是最右
最后判断n朵花是否有剩余，为了效率起见，可以在种花的过程中做判断，一旦花被种完就返回true

class Solution {
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        for(int i=0; i<flowerbed.length; i++) {
            if(flowerbed[i] == 0 && (i == 0 || flowerbed[i-1] == 0) 
                  && (i == flowerbed.length-1 || flowerbed[i+1] == 0)) {
                n--;
                if(n <= 0) return true;
                flowerbed[i] = 1;
            }
        }
        return n <= 0;
    }
}

```

## 205 Isomorphic Strings

Given two strings `s` and `t`, _determine if they are isomorphic_.

Two strings `s` and `t` are isomorphic if the characters in `s` can be replaced to get `t`.

All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character, but a character may map to itself.

```java
Input: s = "egg", t = "add"
Output: true
//s = ab, t = cc，如果单看 s -> t ，那么 a -> c, b -> c 是没有问题的。
//必须再验证 t -> s，此时，c -> a, c -> b，一个字母对应了多个字母，所以不是同构的。

class Solution {
    public boolean isIsomorphic(String s, String t) {
        HashMap<Character, Character> sMap = new HashMap<>();
        HashMap<Character, Character> tMap = new HashMap<>();
        for (int i=0; i<s.length(); i++){
            char sChar = s.charAt(i);
            char tChar = t.charAt(i);
            if ((sMap.containsKey(sChar) && sMap.get(sChar) != tChar) 
                || (tMap.containsKey(tChar) && tMap.get(tChar) != sChar)){
               return false;
            }
            sMap.put(sChar, tChar);
            tMap.put(tChar, sChar);
        }
        return true;
    }
}

```

## 261 Graph Valid Tree

You have a graph of `n` nodes labeled from `0` to `n - 1`. You are given an integer n and a list of `edges` where `edges[i] = [ai, bi]` indicates that there is an undirected edge between nodes `ai` and `bi` in the graph.

Return `true` _if the edges of the given graph make up a valid tree, and_ `false` _otherwise_.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/03/12/tree1-graph.jpg)

```java
Input: n = 5, edges = [[0,1],[0,2],[0,3],[1,4]]
Output: true

class Solution {
    public boolean validTree(int n, int[][] edges) {
          UnionFind uf = new UnionFind(n);
          for(int[] edge: edges){
              int p = edge[0];
              int f = edge[1];
              if(uf.isConnected(p,f)) return false;
              uf.union(p,f);
             
          }  
          return uf.count == 1;   
    }

        public class UnionFind{
        int[] parent;
        int count;
        public UnionFind(int n){
            parent = new int[n];
            count = n;
            for (int i = 0; i < n; i ++) {
                parent[i] = i;
            }
        }
        
        public int find(int p, int[] parent) {
            if (p == parent[p]) return p;
            parent[p] = find(parent[p], parent);
            return parent[p];
        }

        public boolean isConnected(int p, int f){
             int p1 = find(p, parent); int f1 = find(f, parent);
             return p1 == f1;
        }
        
        public void union(int p, int f) {
            int p1 = find(p, parent); int f1 = find(f, parent);
            if (p1 != f1) {
                parent[p1] = f1;
                count --;
            }
        }
    }
}


```

## 655 Print Binary Tree

Given the `root` of a binary tree, construct a **0-indexed** `m x n` string matrix `res` that represents a **formatted layout** of the tree. The formatted layout matrix should be constructed using the following rules:

* The **height** of the tree is `height` and the number of rows `m` should be equal to `height + 1`.
* The number of columns `n` should be equal to `2height+1 - 1`.
* Place the **root node** in the **middle** of the **top row** (more formally, at location `res[0][(n-1)/2]`).
* For each node that has been placed in the matrix at position `res[r][c]`, place its **left child** at `res[r+1][c-2height-r-1]` and its **right child** at `res[r+1][c+2height-r-1]`.
* Continue this process until all the nodes in the tree have been placed.
* Any empty cells should contain the empty string `""`.

Return _the constructed matrix_ `res`.

```java
Input: root = [1,2,3,null,4]
Output: 
[["","","","1","","",""],
 ["","2","","","","3",""],
 ["","","4","","","",""]]

class Solution {
    
    public List<List<String>> printTree(TreeNode root) {
        int height = height(root,1);
        int len = (int)Math.pow(2,height) - 1;
        List<List<String>> list = new ArrayList<>();
        for(int i = 0;i<height;i++){
            List<String> tempList = new ArrayList<>();
            for(int j = 0;j<len ;j++)
                tempList.add("");
            list.add(new ArrayList(tempList));
        }
        setTree(list,root,0,len - 1,height,0);
        return list;
    }
    
    private int height(TreeNode root,int level){
        if(root == null) return level - 1;
    return Math.max(height(root.left,level + 1),height(root.right, level + 1));
    }
    
    private void setTree(List<List<String>> list, TreeNode root,
    int left, int right, int height,int level){
        if(height == level || root == null) return;
        int mid = left + (right - left)/2; //    Here is the mid
        list.get(level).set(mid,String.valueOf(root.val));
        setTree(list,root.left,left,mid - 1,height,level+ 1);
        setTree(list,root.right,mid + 1,right,height,level+1);
    }
}
```

## 713 Subarray Product Less Than K

Given an array of integers `nums` and an integer `k`, return _the number of contiguous subarrays where the product of all the elements in the subarray is strictly less than_ `k`.

**Example 1:**

```java
Input: nums = [10,5,2,6], k = 100
Output: 8
Explanation: The 8 subarrays that have product less than 100 are:
[10], [5], [2], [6], [10, 5], [5, 2], [2, 6], [5, 2, 6]
Note that [10, 5, 2] is not included as the product of 100 is not 
strictly less than k.

The idea is always keep an max-product-window less than K;
Every time shift window by adding a new number on the right(j), 
if the product is greater than k, then try to reduce numbers on the left(i), 
until the subarray product fit less than k again, (subarray could be empty);
Each step introduces x new subarrays, where x is the size of the current window
 (j + 1 - i);
example:
for window (5, 2), when 6 is introduced, it add 3 new subarray: (5, (2, (6)))

class Solution {
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        if (k == 0) return 0;
        int cnt = 0;
        int pro = 1;
        for (int i = 0, j = 0; j < nums.length; j++) {
            pro *= nums[j];
            while (i <= j && pro >= k) {
                pro /= nums[i++];
            }
            cnt += j - i + 1;
        }
        return cnt;        
    }
}

```

## 170 Two Sum III - Data structure design

Design a data structure that accepts a stream of integers and checks if it has a pair of integers that sum up to a particular value.

Implement the `TwoSum` class:

* `TwoSum()` Initializes the `TwoSum` object, with an empty array initially.
* `void add(int number)` Adds `number` to the data structure.
* `boolean find(int value)` Returns `true` if there exists any pair of numbers whose sum is equal to `value`, otherwise, it returns `false`.

```
class TwoSum {
    Map<Integer, Integer> freq = new HashMap<>();

    public void add(int number) {
        // 记录 number 出现的次数
        freq.put(number, freq.getOrDefault(number, 0) + 1);
    }
    
    public boolean find(int value) { // find O(n)
        for (Integer key : freq.keySet()) {
            int other = value - key;
            // 情况一
            if (other == key && freq.get(key) > 1)
                return true;
            // 情况二
            if (other != key && freq.containsKey(other))
                return true;
        }
        return false;
    }
}

class TwoSum {
    Set<Integer> sum = new HashSet<>();
    List<Integer> nums = new ArrayList<>();

    public void add(int number) {
        // 记录所有可能组成的和
        for (int n : nums)
            sum.add(n + number);
        nums.add(number);
    }
    
    public boolean find(int value) {// find O(1);
        return sum.contains(value);
    }
}


```

## 256 Paint House

There is a row of `n` houses, where each house can be painted one of three colors: red, blue, or green. The cost of painting each house with a certain color is different. You have to paint all the houses such that no two adjacent houses have the same color.

The cost of painting each house with a certain color is represented by an `n x 3` cost matrix `costs`.

* For example, `costs[0][0]` is the cost of painting house `0` with the color red; `costs[1][2]` is the cost of painting house 1 with color green, and so on...

Return _the minimum cost to paint all houses_.

**Example 1:**

```java
Input: costs = [[17,2,17],[16,16,5],[14,3,19]]
Output: 10
Explanation: Paint house 0 into blue, paint house 1 into green, 
paint house 2 into blue.
Minimum cost: 2 + 5 + 3 = 10.

class Solution {
    public int minCost(int[][] costs) {
        int n = costs.length;
        int[][] dp = new int[n + 1][3];
        dp[0][0] = 0; dp[0][1] = 0; dp[0][2] = 0;
        for(int i = 0; i < n; i ++){
           dp[i + 1][0] = costs[i][0] + Math.min(dp[i][1], dp[i][2]);
           dp[i + 1][1] = costs[i][1] + Math.min(dp[i][0], dp[i][2]);
           dp[i + 1][2] = costs[i][2] + Math.min(dp[i][0], dp[i][1]);
        }
        return  Math.min(dp[n][2], Math.min(dp[n][0], dp[n][1]));
    }
}

上面的一维动态规划解法使用了一个 dp 数组，我们仔细观察可以发现，
计算 dp[i]的状态只取决于 dp[i-1]的状态，
所以我们可以用三个临时变量 red/blue/green 来代替dp[i-1][0]/dp[i-1][1]/dp[i-1][2]中的值。
class Solution {
    public int minCost(int[][] costs) {
        int[][] dp = new int[costs.length][3];
     int redCost = costs[0][0], blueCost = costs[0][1], greenCost = costs[0][2];
        for (int i = 1; i < costs.length; i++) {
            int newRedCost = Math.min(blueCost, greenCost) + costs[i][0];
            int newBlueCost = Math.min(redCost, greenCost) + costs[i][1];
            int newGreenCost = Math.min(redCost, blueCost) + costs[i][2];
            redCost = newRedCost;
            blueCost = newBlueCost;
            greenCost = newGreenCost;
        }
        return Math.min(redCost, Math.min(blueCost, greenCost));
    }
}

```

## 265 Paint House II

```
本题是256.粉刷房子的增强版，之前是给定3种颜色，现在是一般化为k种颜色。
依然是动态规划：
// f[i][j]表示将房子[0..i]刷完，并且i号房子是颜色j的最小花费
int[][] f = new int[n][k];
加速点在于：
因为只需满足相邻2个房子颜色不同，因此粉刷完[0..i]之后，
只需要记录下让f[i][j]取最小值和次最小值的两个j值（颜色编号）：
colorMin和colorMin2
在粉刷第i+1个房子时，只要所用的颜色j!=colorMin，那么就让前一个房子i取colorMin颜色，
这样能得到最小的f[i+1][j]。
如果粉刷第i+1个房子时所用的颜色j=colorMin，
那么只有让前一个房子取colorMin2颜色。

class Solution {
    public int minCostII(int[][] costs) {
        // n个房子
        int n = costs.length;
        if(n==0)return 0;
        // k种颜色的油漆
        int k = costs[0].length;
        if(k==0)return 0;
        // 至少1种颜色，1间房子
        if(k==1) {
            if(n==1) {
                return costs[0][0];
            } else {
                // 不可能完成任务
                return 0;
            }
        }

        // 至此，至少有2种颜色
        // 相邻两个房子的颜色不同，求最小花费
        // f[i][j]表示将房子[0..i]刷完，并且i号房子是颜色j的最小花费
        int[][] f = new int[n][k];

        int preColorMin = 0;
        int preColorMin2 = 0;
        for (int i = 0; i < n; i++) {
            int costMin = Integer.MAX_VALUE;//记录当前轮，粉刷完[0..i]的最小花费
            int colorMin = k-1;//记录最小花费下，i号房的颜色
            int costMin2 = Integer.MAX_VALUE;//记录当前轮，粉刷完[0..i-1]的次最小花费
            int colorMin2 = k-1;//记录次最小花费下，i号房的颜色

            for (int j = 0; j < k; j++) {
                if(i==0) {
                    f[i][j] = costs[i][j];
                } else {
                    // 当前颜色j不是前一轮的最小颜色，就让前一个房间取最小颜色
                    // 否则，让前一个房间粉刷为次最小颜色
f[i][j] = costs[i][j] + (j!=preColorMin?f[i-1][preColorMin]:f[i-1][preColorMin2]);
                }

                if(f[i][j] < costMin) {
                    // 最小变次最小
                    colorMin2 = colorMin;
                    costMin2 = costMin;
                    // 更新最小
                    colorMin = j;
                    costMin = f[i][j];
                } else if(f[i][j] < costMin2) {
                    // 更新次最小
                    colorMin2 = j;
                    costMin2 = f[i][j];
                }
            }
            preColorMin = colorMin;
            preColorMin2 = colorMin2;
        }
        return f[n-1][preColorMin];
    }
}

```

## 516 Longest Palindromic Subsequence

Given a string `s`, find _the longest palindromic **subsequence**'s length in_ `s`.

A **subsequence** is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.

**Example 1:**

```
Input: s = "bbbab"
Output: 4
Explanation: One possible longest palindromic subsequence is "bbbb".

int longestPalindromeSubseq(string s) {
    int n = s.size();
    // dp 数组全部初始化为 0
    vector<vector<int>> dp(n, vector<int>(n, 0));
    // base case
    for (int i = 0; i < n; i++)
        dp[i][i] = 1;
    // 反着遍历保证正确的状态转移
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            // 状态转移方程
            if (s[i] == s[j])
                dp[i][j] = dp[i + 1][j - 1] + 2;
            else
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
        }
    }
    // 整个 s 的最长回文子串长度
    return dp[0][n - 1];
}
```

## 716 Max Stack

Design a max stack data structure that supports the stack operations and supports finding the stack's maximum element.

Implement the `MaxStack` class:

* `MaxStack()` Initializes the stack object.
* `void push(int x)` Pushes element `x` onto the stack.
* `int pop()` Removes the element on top of the stack and returns it.
* `int top()` Gets the element on the top of the stack without removing it.
* `int peekMax()` Retrieves the maximum element in the stack without removing it.
* `int popMax()` Retrieves the maximum element in the stack and removes it. If there is more than one maximum element, only remove the **top-most** one.

```
对于 peekMax()，我们可以另一个栈来存储每个位置到栈底的所有元素的最大值。
例如，如果当前第一个栈中的元素为 [2, 1, 5, 3, 9]，那么第二个栈中的元素为 [2, 2, 5, 5, 9]。
在 push(x) 操作时，只需要将第二个栈的栈顶和 xx 的最大值入栈，而在 pop() 操作时，
只需要将第二个栈进行出栈。
对于 popMax()，由于我们知道当前栈中最大的元素值，因此可以直接将两个栈同时出栈，
并存储第一个栈出栈的所有值。当某个时刻，第一个栈的出栈元素等于当前栈中最大的元素值时，
就找到了最大的元素。此时我们将之前出第一个栈的所有元素重新入栈，并同步更新第二个栈，
就完成了 popMax() 操作。

class MaxStack {
    Stack<Integer> stack;
    Stack<Integer> maxStack;

    public MaxStack() {
        stack = new Stack();
        maxStack = new Stack();
    }

    public void push(int x) {
        int max = maxStack.isEmpty() ? x : maxStack.peek();
        maxStack.push(max > x ? max : x);
        stack.push(x);
    }

    public int pop() {
        maxStack.pop();
        return stack.pop();
    }

    public int top() {
        return stack.peek();
    }

    public int peekMax() {
        return maxStack.peek();
    }

    public int popMax() {
        int max = peekMax();
        Stack<Integer> buffer = new Stack();
        while (top() != max) buffer.push(pop());
        pop();
        while (!buffer.isEmpty()) push(buffer.pop());
        return max;
    }
}

我们使用双向链表存储栈，使用带键值对的平衡树（Java 中的 TreeMap）存储栈中出现的值以及这个值
在双向链表中出现的位置。
push(x) 操作：在双向链表的末尾添加一个节点，并且在平衡树上找到 xx，
给它的列表中添加一个位置，指向新的节点。
pop()操作：在双向链表的末尾删除一个节点，它的值为 \mathrm{val}val，
随后在平衡树上找到 \mathrm{val}val，删除它的列表中的最后一个位置。
top() 操作：返回双向链表中最后一个节点的值。
peekMax() 操作：返回平衡树上的最大值。
popMax() 操作：在平衡树上找到最大值和它对应的列表，得到列表中的最后一个位置，
并将它在双向链表中和平衡树上分别删除。

class MaxStack {
    TreeMap<Integer, List<Node>> map;
    DoubleLinkedList dll;

    public MaxStack() {
        map = new TreeMap();
        dll = new DoubleLinkedList();
    }

    public void push(int x) {
        Node node = dll.add(x);
        if(!map.containsKey(x))
            map.put(x, new ArrayList<Node>());
        map.get(x).add(node);
    }

    public int pop() {
        int val = dll.pop();
        List<Node> L = map.get(val);
        L.remove(L.size() - 1);
        if (L.isEmpty()) map.remove(val);
        return val;
    }

    public int top() {
        return dll.peek();
    }

    public int peekMax() {
        return map.lastKey();
    }

    public int popMax() {
        int max = peekMax();
        List<Node> L = map.get(max);
        Node node = L.remove(L.size() - 1);
        dll.unlink(node);
        if (L.isEmpty()) map.remove(max);
        return max;
    }
}

class DoubleLinkedList {
    Node head, tail;

    public DoubleLinkedList() {
        head = new Node(0);
        tail = new Node(0);
        head.next = tail;
        tail.prev = head;
    }

    public Node add(int val) {
        Node x = new Node(val);
        x.next = tail;
        x.prev = tail.prev;
        tail.prev = tail.prev.next = x;
        return x;
    }

    public int pop() {
        return unlink(tail.prev).val;
    }

    public int peek() {
        return tail.prev.val;
    }

    public Node unlink(Node node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
        return node;
    }
}

class Node {
    int val;
    Node prev, next;
    public Node(int v) {val = v;}
}

```

## 526 Beautiful Arrangement

Suppose you have `n` integers labeled `1` through `n`. A permutation of those `n` integers `perm` (**1-indexed**) is considered a **beautiful arrangement** if for every `i` (`1 <= i <= n`), **either** of the following is true:

* `perm[i]` is divisible by `i`.
* `i` is divisible by `perm[i]`.

```
假设有从 1 到 n 的 n 个整数。用这些整数构造一个数组 perm（下标从 1 开始），
只要满足下述条件 之一 ，该数组就是一个 优美的排列 ：
perm[i] 能够被 i 整除
i 能够被 perm[i] 整除
给你一个整数 n ，返回可以构造的 优美排列 的 数量 。

class Solution {
    int count = 0;
    public int countArrangement(int n) {
         dfs(n, 1, new boolean[n + 1]);
        return count;
    }

    private void dfs(int n, int i, boolean[] visited) {
        if (i > n) {
            count ++;
            return;
        }
        for (int num = 1; num <= n; num++) {
            if (!visited[num] && (num % i == 0 || i % num == 0)) { 
             // either condition is true;
                visited[num] = true;
                dfs(n, i + 1, visited);
                visited[num] = false;
            }
        }
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

I can put the index pair for the two arrays in a priority queue and 
compute the answer gradually. However, the K can be up to 10^9. 
This will lead to TLE.
The element can be negative. Maybe I need to know the number of negative 
elements and handle 4 different combinations: 
(negative array 1, negative array 2), 
(negative array 1, positive array 2), 
(positive array 1, negative array 2), 
(positive array 1, positive array 2). 
At least, I can know the number of products of each combination and 
locate k-th product among them.
Even though I know which combination the k-th product belongs to, 
it doesn't guarantee I can use the priority queue approach. 
I need another hint.
Continue with above, I think I need some way to eliminate some number of products
 step by step to reach the goal.
Since the array is sorted, if I turn my attention on nums1[i] x nums2[j], 
I can know there are j + 1 products which are less than or 
equal to nums1[i] x nums2[j] that are generated by nums1[i]. 
Then I realize that I should try the binary search.


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

## 698 partition to K Equal Sum Subsets

Given an integer array `nums` and an integer `k`, return `true` if it is possible to divide this array into `k` non-empty subsets whose sums are all equal.

**Example 1:**

```
Input: nums = [4,3,2,3,5,2,1], k = 4
Output: true
Explanation: It is possible to divide it into 4 subsets (5), (1, 4), 
(2,3), (2,3) with equal sums.

class Solution {
    public boolean canPartitionKSubsets(int[] nums, int k) {
        // 排除一些基本情况
        if (k > nums.length) return false;
        int sum = 0;
        for (int v : nums) sum += v;
        if (sum % k != 0) return false;
        
        int used = 0; // 使用位图技巧
        int target = sum / k;
        // k 号桶初始什么都没装，从 nums[0] 开始做选择
        return backtrack(k, 0, nums, 0, used, target);
    }

    HashMap<Integer, Boolean> memo = new HashMap<>();

    boolean backtrack(int k, int bucket,
                    int[] nums, int start, int used, int target) {        
        // base case
        if (k == 0) {
            // 所有桶都被装满了，而且 nums 一定全部用完了
            return true;
        }
        if (bucket == target) {
            // 装满了当前桶，递归穷举下一个桶的选择
            // 让下一个桶从 nums[0] 开始选数字
            boolean res = backtrack(k - 1, 0, nums, 0, used, target);
            // 缓存结果
            memo.put(used, res);
            return res;
        }
        
        if (memo.containsKey(used)) {
            // 避免冗余计算
            return memo.get(used);
        }

        for (int i = start; i < nums.length; i++) {
            // 剪枝
            if (((used >> i) & 1) == 1) { // 判断第 i 位是否是 1
                // nums[i] 已经被装入别的桶中
                continue;
            }
            if (nums[i] + bucket > target) {
                continue;
            }
            // 做选择
            used |= 1 << i; // 将第 i 位置为 1
            bucket += nums[i];



            // 递归穷举下一个数字是否装入当前桶
            if (backtrack(k, bucket, nums, i + 1, used, target)) {
                return true;
            }
            // 撤销选择
            used ^= 1 << i; // 将第 i 位置为 0
            bucket -= nums[i];
        }

        return false;
    }
}
```

## 1117 Building H2O

```
现在有两种线程，氧 oxygen 和氢 hydrogen，你的目标是组织这两种线程来产生水分子。
存在一个屏障（barrier）使得每个线程必须等候直到一个完整水分子能够被产生出来。
氢和氧线程会被分别给予 releaseHydrogen 和 releaseOxygen 方法来允许它们突破屏障。
这些线程应该三三成组突破屏障并能立即组合产生一个水分子。
你必须保证产生一个水分子所需线程的结合必须发生在下一个水分子产生之前。
换句话说:
如果一个氧线程到达屏障时没有氢线程到达，它必须等候直到两个氢线程到达。
如果一个氢线程到达屏障时没有其它线程到达，它必须等候直到一个氧线程和另一个氢线程到达。
书写满足这些限制条件的氢、氧线程同步代码。


import java.util.concurrent.*;
class H2O {
    Semaphore h, o;
      public H2O() {
        h = new Semaphore(2);
        o = new Semaphore(0);
    }

    public void hydrogen(Runnable releaseHydrogen) throws InterruptedException {
	 h.acquire();
        // releaseHydrogen.run() outputs "H". Do not change or remove this line.
        releaseHydrogen.run();
        o.release();
    }

    public void oxygen(Runnable releaseOxygen) throws InterruptedException {
        o.acquire(2);//要上面方法执行两次后O信号量才可以达到这个条件
        // releaseOxygen.run() outputs "O". Do not change or remove this line.
		releaseOxygen.run();
        h.release(2);//执行之前氢信号量是零 执行之后回到原来的2
    }
}

acquire()
Acquires a permit from this semaphore, blocking until one is available, 
or the thread is interrupted.
Acquires a permit, if one is available and returns immediately, 
reducing the number of available permits by one.

acquire(int permits)
Acquires the given number of permits from this semaphore, 
blocking until all are available, or the thread is interrupted.
Acquires the given number of permits, if they are available, 
and returns immediately, reducing the number of available permits 
by the given amount.

release()
Releases a permit, returning it to the semaphore.
Releases a permit, increasing the number of available permits by one

release(int permits)
Releases the given number of permits, returning them to the semaphore.
Releases the given number of permits, increasing the number of available 
permits by that amount. 
```

## 1188 Design Bounded Blocking Queue

Implement a thread-safe bounded blocking queue that has the following methods:

* `BoundedBlockingQueue(int capacity)` The constructor initializes the queue with a maximum `capacity`.
* `void enqueue(int element)` Adds an `element` to the front of the queue. If the queue is full, the calling thread is blocked until the queue is no longer full.
* `int dequeue()` Returns the element at the rear of the queue and removes it. If the queue is empty, the calling thread is blocked until the queue is no longer empty.
* `int size()` Returns the number of elements currently in the queue.

Your implementation will be tested using multiple threads at the same time. Each thread will either be a producer thread that only makes calls to the `enqueue` method or a consumer thread that only makes calls to the `dequeue` method. The `size` method will be called after every test case.

Please do not use built-in implementations of bounded blocking queue as this will not be accepted in an interview.

```
出队列为空的时候要等待，用一个信号量就可以了。
入队列满了要等待，用一个信号量就可以了。
然后还要实现的是入队的顺序要保证，在信号量acquire的时候可能会使后面的先于前面的。
使用一个公平锁即可解决。

      class BoundedBlockingQueue {
        Queue<Integer> queue = new LinkedList<>();
        Semaphore full;
        Semaphore empty;
        ReentrantLock lock = new ReentrantLock(true);

        public BoundedBlockingQueue(int capacity) {
            full = new Semaphore(capacity);
            empty = new Semaphore(0);
        }

        public void enqueue(int element) throws InterruptedException {
            try {
                lock.lock();
                full.acquire();
                synchronized (queue) {
                    queue.add(element);
                }
                empty.release();
            } finally {
                lock.unlock();
            }

        }

        public int dequeue() throws InterruptedException {
            empty.acquire();
            int x;
            synchronized (queue) {
                x = queue.poll();
            }
            full.release();
            return x;
        }

        public int size() {
            try {
                lock.lock();
                return queue.size();
            } finally {
                lock.unlock();
            }
        }
    }

```

## 187 Repeated DNA Sequences

The **DNA sequence** is composed of a series of nucleotides abbreviated as `'A'`, `'C'`, `'G'`, and `'T'`.

* For example, `"ACGAATTCCG"` is a **DNA sequence**.

When studying **DNA**, it is useful to identify repeated sequences within the DNA.

Given a string `s` that represents a **DNA sequence**, return all the **`10`-letter-long** sequences (substrings) that occur more than once in a DNA molecule. You may return the answer in **any order**.

```
滑动窗口 + 哈希表
数据范围只有 10^5
 ，一个朴素的想法是：从左到右处理字符串 s，使用滑动窗口得到每个以 s[i]
 为结尾且长度为 10 的子串，同时使用哈希表记录每个子串的出现次数，如果该子串出现次数超过一次，
 则加入答案。为了防止相同的子串被重复添加到答案，而又不使用常数较大的 Set 结构。
 我们可以规定：当且仅当该子串在之前出现过一次（加上本次，当前出现次数为两次）时，将子串加入答案。
 class Solution {
    public List<String> findRepeatedDnaSequences(String s) {
        List<String> ans = new ArrayList<>();
        int n = s.length();
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i + 10 <= n; i++) {
            String cur = s.substring(i, i + 10);
            int cnt = map.getOrDefault(cur, 0);
            if (cnt == 1) ans.add(cur);
            map.put(cur, cnt + 1);
        }
        return ans;
    }
}

```

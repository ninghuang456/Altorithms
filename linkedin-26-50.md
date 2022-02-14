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
    /**
     * k-v查找节点
     */
    private final Map<String, ListNode> map = new HashMap<>();
    /**
     * key - 节点的值；
     * value - 链表中第一个值为key的节点。
     */
    private final Map<Integer, ListNode> first = new HashMap<>();
    /**
     * key - 节点的值；
     * value - 链表中最后一个值为key的节点。
     */
    private final Map<Integer, ListNode> last = new HashMap<>();

    /**
     * 链表伪头节点
     */
    private final ListNode head = new ListNode(null, 0);
    /**
     * 链表伪尾节点
     */
    private final ListNode tail = new ListNode(null, 0);

    AllOne() {
        head.next = tail;
        tail.prev = head;
    }

    private class ListNode { // 链表节点
        ListNode prev, next;
        String key;
        int val;

        public ListNode(String key, int val) {
            this.key = key;
            this.val = val;
        }
    }

    /**
     * 将节点 [insert] 插入到 n1 与 n2 之间
     */
    private void insert(ListNode n1, ListNode n2, ListNode insert) {
        n1.next = insert;
        n2.prev = insert;
        insert.prev = n1;
        insert.next = n2;
    }

    /**
     * 删除链表节点[n]
     */
    private void remove(ListNode n) {
        ListNode prev = n.prev;
        ListNode next = n.next;
        prev.next = next;
        next.prev = prev;
        n.prev = null;
        n.next = null;
    }

    /**
     * 将节点node移动到prev与next之间
     */
    private void move(ListNode node, ListNode prev, ListNode next) {
        remove(node);
        insert(prev, next, node);
    }

    /**
     * 将[node]设置为新的val值起始点
     */
    private void newFirst(int val, ListNode node) {
        first.put(val, node);
        if (!last.containsKey(val)) last.put(val, node);
    }

    /**
     * 将[node]设置为新的val值终止点
     */
    private void newLast(int val, ListNode node) {
        last.put(val, node);
        if (!first.containsKey(val)) first.put(val, node);
    }

    /**
     * Inserts a new key <Key> with value 1. Or increments an existing key by 1.
     * <p>
     * 值加一后，当前节点会往左移动。
     * 如果当前key不存在，那就把这个节点插入到链表尾部.
     */
    public void inc(String key) {
        if (!map.containsKey(key)) { // 当前key不存在，插入到链表末尾
            ListNode node = new ListNode(key, 1);
            map.put(key, node);
            insert(tail.prev, tail, node); // 插入
            if (!first.containsKey(1)) newFirst(1, node); // 更新first
            newLast(1, node); // 更新last
        } else {
            ListNode node = map.get(key); // 当前节点
            int val = node.val; // 旧值
            int newVal = val + 1; // 新值
            ListNode firstNode = first.get(val); // 链表中第一个值为val的节点
            ListNode lastNode = last.get(val); // 链表中最后一个值为val的节点

            // 1. 找位置
            node.val = newVal;
            if (firstNode == lastNode) { // 当前节点是唯一一个值为val的节点
                first.remove(val); // 没有值为val的节点了
                last.remove(val); // 没有值为val的节点了
                newLast(newVal, node); // 更新last
            } else if (node == firstNode) { // 该节点是链表中第一个值为val的节点
                // 不动
                newLast(newVal, node);
                newFirst(val, node.next);
            } else {
                if (node == lastNode) newLast(val, node.prev); // 是最后一个值val的节点
                // 这个时候，节点应该移动到链表中第一个值为val的节点之前
                move(node, firstNode.prev, firstNode);
                newLast(newVal, node);
            }
        }
    }

    /**
     * Decrements an existing key by 1. If Key's value is 1, remove it from the data structure.
     * 
     * 值减一之后，节点在链表中的位置会往右移动
     */
    public void dec(String key) {
        // 与inc类似，不过多了一个值为1删除的判断
        ListNode node = map.get(key);
        if (node == null) return;

        int val = node.val;
        int newVal = val - 1;
        ListNode firstNode = first.get(val);
        ListNode lastNode = last.get(val);

        if (val == 1) { // 值为1，删除这个节点
            if (firstNode == lastNode) { // 没有值为1的节点了
                first.remove(1);
                last.remove(1);
            } else if (node == firstNode) { // 起始值右移
                first.put(1, node.next);
            } else if (node == lastNode) { // 终结值左移
                last.put(1, node.prev);
            }
            remove(node);
            map.remove(key);
        } else {
            node.val = newVal;
            if (firstNode == lastNode) { // 唯一值为val的节点
                // 位置不变，成为newVal的首位
                first.remove(val);
                last.remove(val);
                newFirst(newVal, node);
            } else if (node == lastNode) { // 是最后一项val值的节点
                // 位置不变，成为newVal的首位，并且prev成为val的最后一位
                newFirst(newVal, node);
                newLast(val, node.prev);
            } else {
                if (node == firstNode) newFirst(val, node.next); // 是第一项val值的节点
                move(node, lastNode, lastNode.next); // 移动到lastNode之后
                newFirst(newVal, node);
            }
        }
    }

    /**
     * Returns one of the keys with maximal value.
     * 返回链表头
     */
    public String getMaxKey() {
        return head.next == tail ? "" : head.next.key;
    }

    /**
     * Returns one of the keys with Minimal value.
     * 返回链表尾
     */
    public String getMinKey() {
        return tail.prev == head ? "" : tail.prev.key;
    }
}
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
Input: words = ["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16
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
            //最大积的可能情况有：元素i自己本身，上一个最大积与i元素累乘，上一个最小积与i元素累乘；
            //与i元素自己进行比较是为了处理i元素之前全都是0的情况；
            maxDP[i] = Math.max(nums[i], Math.max(maxDP[i-1]*nums[i], minDP[i-1]*nums[i]));
            minDP[i] = Math.min(nums[i], Math.min(maxDP[i-1]*nums[i], minDP[i-1]*nums[i]));
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
我们知道，两个点可以确定一条线。
因此一个朴素的做法是先枚举两条点（确定一条线），然后检查其余点是否落在该线中。
为了避免除法精度问题，当我们枚举两个点 i 和 j 时，不直接计算其对应直线的 斜率和 截距，
而是通过判断 i 和 j 与第三个点 kk 形成的两条直线斜率是否相等（斜率相等的两条直线要么平行，
要么重合，平行需要 4 个点来唯一确定，我们只有 3 个点，所以可以直接判定两直线重合）。


class Solution {
    public int maxPoints(int[][] ps) {
        int n = ps.length;
        int ans = 1;
        for (int i = 0; i < n; i++) {
            int[] x = ps[i];
            for (int j = i + 1; j < n; j++) {
                int[] y = ps[j];
                int cnt = 2;
                for (int k = j + 1; k < n; k++) {
                    int[] p = ps[k];
                    int s1 = (y[1] - x[1]) * (p[0] - y[0]);
                    int s2 = (p[1] - y[1]) * (y[0] - x[0]);
                    if (s1 == s2) cnt++;
                }
                ans = Math.max(ans, cnt);
            }
        }
        return ans;
    }
}

根据「朴素解法」的思路，枚举所有直线的过程不可避免，但统计点数的过程可以优化。
具体的，我们可以先枚举所有可能出现的 直线斜率（根据两点确定一条直线，即枚举所有的「点对」）
使用「哈希表」统计所有 斜率 对应的点的数量，在所有值中取个 maxmax 即是答案。
一些细节：在使用「哈希表」进行保存时，为了避免精度问题，我们直接使用字符串进行保存，
同时需要将 斜率 约干净。

class Solution {
    public int maxPoints(int[][] ps) {
        int n = ps.length;
        int ans = 1;
        for (int i = 0; i < n; i++) {
            Map<String, Integer> map = new HashMap<>();
            // 由当前点 i 发出的直线所经过的最多点数量
            int max = 0;
            for (int j = i + 1; j < n; j++) {
                int x1 = ps[i][0], y1 = ps[i][1], x2 = ps[j][0], y2 = ps[j][1];
                int a = x1 - x2, b = y1 - y2;
                int k = gcd(a, b);
                String key = (a / k) + "_" + (b / k);
                map.put(key, map.getOrDefault(key, 0) + 1);
                max = Math.max(max, map.get(key));
            }
            ans = Math.max(ans, max + 1);
        }
        return ans;
    }
    int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
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
        return Math.max(height(root.left,level + 1),
                        height(root.right, level + 1));
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

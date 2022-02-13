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


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

## 273 - Integer to English Words

```java
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
     
        String[] words1 = new String[]{"","","Twenty ", "Thirty ", "Forty ", 
        "Fifty ", "Sixty ","Seventy ", "Eighty ", "Ninety "};
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

## 200- Number of island

```java
class Solution {
    int[][] dis = new int[][]{{-1,0},{0,-1},{0,1},{1,0}};
    int R = 0;
    int C =  0;
    boolean[][] visited ;
    char[][] grid;
    public int numIslands(char[][] grid) {
      this.grid = grid;
      R = grid.length;  
      if(R == 0) return 0;
      C = grid[0].length;
      visited = new boolean[R][C];  
      int result = 0;
      for(int i = 0; i < R; i ++)
         for(int j = 0; j < C; j ++) {
            if(grid[i][j] == '1' && !visited[i][j]){
               dfs(grid,i,j);
               result ++;
            }
         }
      return result;
        
    }
     public void dfs(char[][] grid, int x, int y){
      visited[x][y] = true;
      for (int i = 0; i < 4; i ++){
         int nextx = x + dis[i][0];
         int nexty = y + dis[i][1]; // x , y 别弄混了
         if(inArea(nextx,nexty) && 
         grid[nextx][nexty] == '1' && !visited[nextx][nexty] ){
            dfs(grid,nextx,nexty);
         }
      }
   }
   private boolean inArea(int x, int y){
     return x >= 0 && x < R && y >=0 && y < C;
   }
}
```

## 1239 - Maximum Length of a Concatenated String with Unique Characters

```java
Given an array of strings arr. String s is a concatenation of a sub-sequence 
of arr which have unique characters.
Return the maximum possible length of s.

Example 1:
Input: arr = ["un","iq","ue"]
Output: 4
Explanation: All possible concatenations are "","un","iq","ue","uniq" and "ique".
Maximum length is 4.
Example 2:

Input: arr = ["cha","r","act","ers"]
Output: 6
Explanation: Possible solutions are "chaers" and "acters".

class Solution {
    private boolean isUnique(String str) {
        if (str.length() > 26) return false;
        boolean[] used = new  boolean [26];
        char[] arr = str.toCharArray();
        for (char ch : arr) {
            if (used[ch - 'a']){
            return false; 
            } else {
            used[ch -'a'] = true;
        }
    }
        return true;
    }
    public int maxLength(List<String> arr) {
        List<String> res = new ArrayList<>();
        res.add("");
        for (String str : arr) {
            if (!isUnique(str)) continue;
            List<String> resList = new ArrayList<>();
            for (String candidate : res) {
                String temp = candidate + str;
                if (isUnique(temp)) resList.add(temp);
            }
            res.addAll(resList);
        }
        int ans = 0;
        for (String str : res) ans = Math.max(ans, str.length());
        return ans;
    }
}
```

## 428 Serialize and Deserialize N-ary Tree

```java
class Codec {

    // Encodes a tree to a single string.
    public String serialize(Node root) {
        List<String> list=new LinkedList<>();
        serializeHelper(root,list);
        return String.join(",",list);
    }
    
    private void serializeHelper(Node root, List<String> list){
        if(root==null){
            return;
        }else{
            list.add(String.valueOf(root.val));
            list.add(String.valueOf(root.children.size()));
            for(Node child:root.children){
                serializeHelper(child,list);
            }
        }
    }

    // Decodes your encoded data to tree.
    public Node deserialize(String data) {
        if(data.isEmpty())
            return null;
        
        String[] ss=data.split(",");
        Queue<String> q=new LinkedList<>(Arrays.asList(ss));
        return deserializeHelper(q);
    }
    
    private Node deserializeHelper(Queue<String> q){
        Node root=new Node();
        root.val=Integer.parseInt(q.poll());
        int size=Integer.parseInt(q.poll());
        root.children=new ArrayList<Node>(size);
        for(int i=0;i<size;i++){
            root.children.add(deserializeHelper(q));
        }
        return root;
    }
}
```

## 642 Design Search Autocomplete System

```java
Input
["AutocompleteSystem", "input", "input", "input", "input"]
[[["i love you", "island", "iroman", "i love leetcode"], [5, 3, 2, 2]], ["i"], 
[" "], ["a"], ["#"]]
Output
[null, ["i love you", "island", "i love leetcode"], 
["i love you", "i love leetcode"], [], []]

Explanation
AutocompleteSystem obj = new AutocompleteSystem(["i love you", "island", "iroman",
 "i love leetcode"], [5, 3, 2, 2]);
obj.input("i"); // return ["i love you", "island", "i love leetcode"]. 
//There are four sentences that have prefix "i". Among them, "ironman" 
//and "i love leetcode" have same hot degree. Since ' ' has ASCII code 32 
//and 'r' has ASCII code 114, "i love leetcode" should be in front of "ironman". 
//Also we only need to output top 3 hot sentences, so "ironman" will be ignored.
obj.input(" "); // return ["i love you", "i love leetcode"]. There are only two 
//sentences that have prefix "i ".
obj.input("a"); // return []. There are no sentences that have prefix "i a".
obj.input("#"); // return []. The user finished the input, the sentence "i a"
// should be saved as a historical sentence in system. And the following 
//input will be counted as a new search.
/**
 将整个句子都视作单词加入字典树。前缀匹配回溯较多，修改字典树结点结构，
 使得每个结点维护其后计数最大的三个字符串，加速匹配。
 */
class AutocompleteSystem {
    // 构建Trie
    class Trie {
        // 确定路径上的某个字符是否是某个单词（句子）的结尾字符
        boolean isEnding;

        int count;
        String s;

        Trie[] children = new Trie[27]; // 26个字符加‘ ’

        // 小顶堆保存最大的k个
        int k;
        PriorityQueue<Trie> queue;

        Trie(int k) {
            this.k = k;
            // min Heap
            this.queue = new PriorityQueue<>((a, b) -> 
            a.count == b.count ? b.s.compareTo(a.s) : a.count - b.count);
            // 字典树的一条路径的最后一位存空格
            this.children = new Trie[27];

        }

        private void insert(String word, int val) {
            if (word.isEmpty()) {
                return;
            }

            Trie temp = this;
            // 记录路径上的节点
            List<Trie> path = new LinkedList<>();
            for (char c : word.toCharArray()) {
                int index = (c == ' ') ? 26 : (c - 'a');
                if (temp.children[index] == null) {
                    temp.children[index] = new Trie(k);
                }

                temp = temp.children[index];
                path.add(temp);
            }

            // 结尾的字符特殊标记，并进行整个路径的计数更新
            temp.isEnding = true;
            temp.count += val;
            temp.s = word;

            // 关联终止节点到路径上的每个节点
            for (Trie cur : path) {
                // remove old value
                if (cur.queue.contains(temp)) {
                    cur.queue.remove(temp);
                }
                // update new value
                cur.queue.add(temp);
                // 大于k个，将最小的弹出
                while (cur.queue.size() > k) {
                    cur.queue.poll();
                }
            }
        }

        private void search(List<String> results) {
            List<Trie> temp = new ArrayList<>();
            // 加入堆中元素
            while (!queue.isEmpty()) {
                Trie trie = queue.poll();
                temp.add(trie);
                results.add(trie.s);
            }
            queue.addAll(temp);
            Collections.reverse(results);
        }
    }

    // 字典树
    private final Trie root;
    // 记录前一个节点
    private Trie pre;
    //记录历史字符串
    private StringBuilder sb;

    public AutocompleteSystem(String[] sentences, int[] times) {
        this.root = new Trie(3); // 每条路径最多三个单词
        this.pre = root;
        sb = new StringBuilder();

        for (int i = 0; i < sentences.length; i++) {
            root.insert(sentences[i], times[i]);
        }
    }
    
    public List<String> input(char c) {
        List<String> results = new LinkedList<>();
        // 遇到‘#’终止
        if (c == '#') {
            // 更新历史记录
            saveHistory(sb.toString());
            // 清空状态
            pre = root;
            sb = new StringBuilder();
            return results;
        }

        // 尚未终止
        // 记录历史
        sb.append(c);

        // 更新节点
        if (pre != null) {
            pre = (c == ' ') ? pre.children[26] : pre.children[c - 'a'];
        }
        // 搜索其后的所有值
        if (pre != null) {
            pre.search(results);
        }
        return results;
    }

    private void saveHistory(String s) {
        root.insert(s, 1);
    }

    
}
```

## 706 Design Hashmap

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

## 157 Replace All ?'s to Avoid Consecutive Repeating Characters

```java
Input: s = "?zs"
Output: "azs"
Explanation: There are 25 solutions for this problem. From "azs" to "yzs", 
all are valid. Only "z" is an invalid modification as the string will consist
of consecutive repeating characters in "zzs".
//for each char, just try ‘a’, ‘b’, ‘c’, and select the one not the same as 
//neighbors.

    public String modifyString(String s) {
        char[] arr = s.toCharArray();
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == '?') {
                for (int j = 0; j < 3; j++) {
                    if (i > 0 && arr[i - 1] - 'a' == j) continue;
                    if (i + 1 < arr.length && arr[i + 1] - 'a' == j) continue;
                    arr[i] = (char) ('a' + j);
                    break;
                }
            }
        }
        return String.valueOf(arr);
    }
```

## 54 Spiral Matrix

```java
class Solution {
      public List<Integer> spiralOrder(int[][] matrix) {

        LinkedList<Integer> result = new LinkedList<>();
        if(matrix==null||matrix.length==0) return result;
        int left = 0;
        int right = matrix[0].length - 1;
        int top = 0;
        int bottom = matrix.length - 1;
        int numEle = matrix.length * matrix[0].length;
        while (numEle >= 1) {
            for (int i = left; i <= right && numEle >= 1; i++) {
                result.add(matrix[top][i]);
                numEle--;
            }
            top++;
            for (int i = top; i <= bottom && numEle >= 1; i++) {
                result.add(matrix[i][right]);
                numEle--;
            }
            right--;
            for (int i = right; i >= left && numEle >= 1; i--) {
                result.add(matrix[bottom][i]);
                numEle--;
            }
            bottom--;
            for (int i = bottom; i >= top && numEle >= 1; i--) {
                result.add(matrix[i][left]);
                numEle--;
            }
            left++;
        }
        return result;
    }
}
```

## 117 Populating Next Right Pointers in Each Node II

![](.gitbook/assets/117_sample.png)

```java
Input: root = [1,2,3,4,5,null,7]
Output: [1,#,2,3,#,4,5,7,#]
Explanation: Given the above binary tree (Figure A), your function should populate
 each next pointer to point to its next right node, just like in Figure B.
  The serialized output is in level order as connected by the next pointers, 
  with '#' signifying the end of each level.
  
  public void connect(TreeLinkNode root) {
    TreeLinkNode dummyHead = new TreeLinkNode(0);
    TreeLinkNode pre = dummyHead;
    while (root != null) {
	    if (root.left != null) {
		    pre.next = root.left;
		    pre = pre.next;
	    }
	    if (root.right != null) {
		    pre.next = root.right;
		    pre = pre.next;
	    }
	    root = root.next;
	    if (root == null) {
		    pre = dummyHead;
		    root = dummyHead.next;
		    dummyHead.next = null;
	    }
    }
}
```

## 212 Word Search II

```java
Input:board=[["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],
            ["i","f","l","v"]], 
words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]

class Solution {
    public List<String> findWords(char[][] board, String[] words) {
        List<String> res = new ArrayList<>();
        for (String word : words) {
            if (exist(board, word)) {
                res.add(word);
            }
        }
        return res;
    }

    public boolean exist(char[][] board, String word) {
        int row = board.length;
        int col = board[0].length;
        boolean[][] visited = new boolean[row][col];
        for (int i = 0; i < row; i ++) {
            for (int j = 0; j < col; j ++){
                if (board[i][j] == word.charAt(0) && findWord(board, word, i, j, 0, visited)) { // start from 0;
                    return true;
                }
            }
        }
        return false;
    }
    
    public boolean findWord(char[][] board, String word, int i, int j, int size,  boolean[][] visited){
        if(size == word.length()){// it means last run's index reach last letter; similer like node == null;
            return true;
        }
        
        if (!inArea(board, i , j) || visited[i][j] || board[i][j] != word.charAt(size)){ // need check in area before visited[i][j]
            return false;
        }
        visited[i][j] = true;
        boolean res = findWord(board, word, i - 1, j, size + 1, visited) ||
            findWord(board, word, i, j - 1, size + 1, visited) ||
            findWord(board, word, i + 1, j, size + 1, visited) ||
            findWord(board, word, i , j + 1, size + 1, visited);
         visited[i][j] = false;   
        return res;
    }
    
    boolean inArea(char[][] board,int i, int j){
        return i >= 0 && i < board.length && j >= 0 && j < board[0].length;
    }
}

// 

class Solution {
    public List<String> findWords(char[][] board, String[] words) {
        //构建字典树
        wordTrie myTrie=new wordTrie();
        trieNode root= myTrie.root;
        for(String s:words)
            myTrie.insert(s);
        //使用set防止重复
        Set<String> result =new HashSet<>();
        int m=board.length;
        int n=board[0].length;
        boolean [][]visited=new boolean[m][n];
        //遍历整个二维数组
        for(int i=0;i<board.length; i++){
            for (int j = 0; j < board [0].length; j++){
                find(board,visited,i,j,m,n,result,root);
            }
        }
        System.out.print(result);
        return new LinkedList<String>(result);
    }
    private void find(char [] [] board, boolean [][]visited,int i,int j,int m,int n,Set<String> result,trieNode cur){
        //边界以及是否已经访问判断
        if(i<0||i>=m||j<0||j>=n||visited[i][j])
            return;
        cur=cur.child[board[i][j]-'a'];
        visited[i][j]=true;
        if(cur==null)
        {
            //如果单词不匹配，回退
            visited[i][j]=false;
            return;
        }
        //找到单词加入
        if(cur.isLeaf)
        {
            result.add(cur.val);
            //找到单词后不能回退，因为可能是“ad” “addd”这样的单词得继续回溯
//            visited[i][j]=false;
//            return;
        }
        find(board,visited,i+1,j,m,n,result,cur);
        find(board,visited,i,j+1,m,n,result,cur);
        find(board,visited,i,j-1,m,n,result,cur);
        find(board,visited,i-1,j,m,n,result,cur);
        //最后要回退，因为下一个起点可能会用到上一个起点的字符
        visited[i][j]=false;
    }


//字典树
class wordTrie{
    public trieNode root=new trieNode();
    public void insert(String s){
        trieNode cur=root;
        for(char c:s.toCharArray()){
            if(cur.child[c-'a']==null){ // like containsKey
                cur.child [c-'a'] = new trieNode();
            }
            cur=cur.child [c-'a'];
        }
        cur.isLeaf=true;
        cur.val=s;
    }
}
//字典树结点
class trieNode {
    public String val;
    public trieNode[] child=new trieNode[26];
    public boolean isLeaf=false;
    trieNode(){
    }
}


}
```

## 297 Serialize and Deserialize Binary Tree

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

## 23 Merge k Sorted Lists

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

## 151 Reverse Words in a String

```java
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

## 295 Find Median from Data Stream

```java
For example, for arr = [2,3,4], the median is 3.
For example, for arr = [2,3], the median is (2 + 3) / 2 = 2.5.

class MedianFinder {
   private Queue<Integer> small = new PriorityQueue<>((o1,o2) -> (o2 - o1));
   private Queue<Integer> large = new PriorityQueue();
    // Adds a number into the data structure.
    public void addNum(int num) {
        large.add(num);
        small.add(large.poll());
        if (large.size() < small.size())
            large.add(small.poll());
    }

    // Returns the median of current data stream
    public double findMedian() {
        return large.size() > small.size()
               ? large.peek()
               : (large.peek() + small.peek()) / 2.0;
    }
}
```

## 236 Lowest Common Ancestor of a Binary Tree

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || p == root || q == root) return root;
        Map<TreeNode, TreeNode> parent = new HashMap<>();
        Queue<TreeNode> queue = new LinkedList<>();
        parent.put(root, null);//根节点没有父节点，所以为空
        queue.add(root);
        //直到两个节点都找到为止。
        while (!parent.containsKey(p) || !parent.containsKey(q)) {
           TreeNode cur = queue.poll();
           if(cur.right != null){
               parent.put(cur.right, cur);
               queue.offer(cur.right);
           }
           if(cur.left != null){
               parent.put(cur.left, cur);
               queue.offer(cur.left);
           } 
        }
        
        HashSet<TreeNode> pParents = new HashSet<>();
        while(p != null){
            pParents.add(p);
            p = parent.get(p);
        }
        while(q != null){
            if(pParents.contains(q)){
                return q;
            }
            q = parent.get(q);
        }
        return root;
    }
}

// 
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null) return root;
        if(p == root || q == root) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if(left != null && right != null) return root;
        if(right == null) return left;
        return right;
    }
}
```

## 224 Basic Calculator

```java
class Solution {
    public int calculate(String s) {
        Stack<Integer> stack = new Stack<Integer>();
    int result = 0;
    int number = 0;
    int sign = 1;
    for(int i = 0; i < s.length(); i++){
        char c = s.charAt(i);
        if(Character.isDigit(c)){
            number = 10 * number + (int)(c - '0');
        }else if(c == '+'){
            result += sign * number;
            number = 0;
            sign = 1;
        }else if(c == '-'){
            result += sign * number;
            number = 0;
            sign = -1;
        }else if(c == '('){
            //we push the result first, then sign;
            stack.push(result);
            stack.push(sign);
            //reset the sign and result for the value in the parenthesis
            sign = 1;   
            result = 0;
        }else if(c == ')'){
            result += sign * number;  
            number = 0;
            result *= stack.pop();    //stack.pop() is the sign before the parenthesis
            result += stack.pop();   //stack.pop() now is the result calculated before the parenthesis
            
        }
    }
    if(number != 0) result += sign * number;
    return result;
        
    }
}
```

## 138 Copy List with Random Pointer

```java
class Solution {
    public Node copyRandomList(Node head) {
        HashMap<Node, Node> map = new HashMap<>();
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

## 41 First Missing Positive

```java
Given an unsorted integer array nums, find the smallest missing positive integer.
You must implement an algorithm that runs in O(n) time and uses constant extra 
space.
Example 1:
Input: nums = [1,2,0]
Output: 3
Example 2:

Input: nums = [3,4,-1,1]
Output: 2

class Solution {
    public int firstMissingPositive(int[] nums) {
        if (nums == null || nums.length == 0) return 1;
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            while (nums[i] > 0 && nums[i] <= n && nums[i] != nums[nums[i] - 1]) {
                int temp = nums[nums[i] - 1];
                nums[nums[i] - 1] = nums[i];
                nums[i] = temp;
            }
        }
        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) return i + 1;
        }

        //从1开始连续出现
        return n + 1;
    }
}
```

## 17 Letter Combinations of a Phone Number

```java
class Solution {
    public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();
        String[] phone = new String[]{" ","","abc", "def", "ghi","jkl",
         "mno","pqrs","tuv","wxyz"};
        if(digits.length() != 0) {
        lettterCombinHelper(digits, "", res, phone, 0);
        }
        return res;
    }
    
    public void lettterCombinHelper(String digits, String temp, List<String> res, 
                                   String[] phone, int start) {
        if (temp.length() == digits.length()) {
            res.add(temp);
            return;
        }
        for (int i = start; i < digits.length(); i ++ ) {
            String c = digits.substring(i, i + 1);
            int index = Integer.valueOf(c);
            String nums = phone[index];
            for (int j = 0; j < nums.length(); j ++) {
                char cur = nums.charAt(j);
                lettterCombinHelper(digits, temp + cur, res, phone, i + 1);
            }
        }
    }
}
```

## 49 Group Anagrams

```java
Example 1:

Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> res = new ArrayList<>();
        if(strs == null || strs.length == 0) return res;
        HashMap<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            int[] temp = new int[26];
            for (char c : chars){
                temp[c - 'a']++;
            }
            StringBuilder sb = new StringBuilder();
           for(int i = 0; i < 26; i++){
               if(temp[i] != 0){
                   sb.append(temp[i]).append(i); 
                   // it will like 197296: 97 is assii code position
               }
           }
           String key = sb.toString(); 
           if(map.containsKey(key)){
               map.get(key).add(str);
           } else {
               ArrayList<String> ls = new ArrayList<>();
               ls.add(str);
               map.put(key,ls);
           }
            
        }
        for (String key : map.keySet()){
            res.add(map.get(key));
        }
        return res;
    }
}


```

## 62 Unique Paths

```java
Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the bottom-right
 corner:
1. Right -> Down -> Down
2. Down -> Down -> Right
3. Down -> Right -> Down

class Solution {
    public int uniquePaths(int m, int n) {
        int[][] grid = new int[m][n];
        grid[0][0] = 1;
        for (int i = 1; i < m; i ++) {
            grid[i][0] = 1;
        }
        
        for (int j = 1; j < n; j ++) {
            grid[0][j] = 1;
        }
        
        for (int i = 1; i < m; i ++){
            for (int j  = 1; j < n; j ++){
                grid[i][j] = grid[i - 1][j] + grid[i][j - 1];
            }
        }
            
        return grid[m - 1][n - 1];
    }
}
```

## 

## 

## 

## 

## 


# Amazon Frequency 1\~ 30

## 1: Two Sum

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int[] res = new int[2];
        if (nums == null || nums.length == 0 ) return res;
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i ++) {
            if (map.containsKey(target - nums[i])){
               int index1 = map.get(target - nums[i]);
               res[0] = index1;
               res[1] = i; 
            } else {
                map.put(nums[i], i);
            }
        }
        
        return res;
    }
}
```

## 42-Trapping Rain Water

```java
class Solution {
    public int trap(int[] height) {
        if (height == null || height.length == 0) return 0;
        int left = 0; int right = height.length - 1;
        int leftMax = height[left];
        int rightMax = height[right];
        int sum = 0;
        while (left <= right){
           leftMax = Math.max(leftMax, height[left]);
           rightMax = Math.max(rightMax, height[right]);
           if (leftMax < rightMax){
               sum += leftMax - height[left];
               left ++;
           } else {
               sum += rightMax - height[right];
               right --;
           }
            
        }
        return sum;
    }
}
```

## 200 Number of Islands

```java
//BFS
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
// Union find
class Solution {
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) return 0;
        UnionFind uf = new UnionFind(grid);
        int r = grid.length;
        int c = grid[0].length;
        
        for (int i = 0; i < r; i ++){
            for (int j = 0; j < c; j ++){
                if (i + 1 < r && grid[i][j] == '1' && grid[i][j] == grid[i+1][j]){
                    uf.union(i * c + j, (i + 1)* c + j);
                }
                if (j + 1 < c && grid[i][j] == '1' && grid[i][j] == grid[i][j + 1]){
                    uf.union(i * c + j, i * c + j + 1);
                }
            }
        }
        
        return uf.count;
        
    }
    
  class UnionFind{
      int[] parent;
      int count; 
      
      public UnionFind(char[][] grid){ 
          int r = grid.length;
          int c = grid[0].length;
          parent = new int[r*c];
          count = 0;
          for (int i = 0; i < r; i ++){
              for (int j = 0; j < c; j ++){
                  if(grid[i][j] == '1'){
                      parent[i * c + j] = i * c + j;
                      count ++;
                  }
              }
          }
          
      }
      
      public int find(int p, int[] parent){
          if (p == parent[p]) {
              return p;
          }
          parent[p] = find(parent[p], parent);
          return parent[p];
      }
      
      public void union(int p , int f){
          int p1 = find(p, parent); int f1 = find(f, parent);
          if (p1 != f1){
              parent[p1] = f1;
              count --;
          }
      }
  }  
    
    
}

// DFS
class Solution {
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) return 0;
        int r = grid.length;
        int c = grid[0].length;
        int res = 0;
    //    boolean[][] visted = new boolean[r][c];
        for (int i = 0; i < r; i ++){
            for (int j = 0; j < c; j ++){
                if (grid[i][j] == '1'){
                    searchIsland(grid,i,j);
                    res ++;
                }               
            }
        }
        return res;
    }
        
    public void searchIsland(char[][] grid, int row, int col) {
        if(!inArea(grid,row, col)) return;
        if(grid[row][col] != '1') return;
        grid[row][col] = '2';
        searchIsland(grid, row - 1, col);
        searchIsland(grid, row + 1, col);
        searchIsland(grid, row, col - 1);
        searchIsland(grid, row, col + 1);
    }
    
    public boolean inArea(char[][] grid, int row, int col){
        return row >= 0 && row < grid.length && col >= 0 && col < grid[0].length;
    }
    
}
```

## 146-LRU Cache

```java
class Node {
    int key, value;
    Node pre, next;
    public Node (int key, int value) {
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
        node.pre = null;
        node.next = null;
        size --;
    }
    
    public void addFirst(Node node) {
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
    
    public int getSize() {
        return this.size;
    }
    
}

class LRUCache {
    DoubleList cache;
    HashMap<Integer, Node> map;
    int capacity;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        cache = new DoubleList(0);
        map = new HashMap<>();
    }
    
    public int get(int key) {
        if (!map.containsKey(key)){
            return -1;
        }
        Node node = map.get(key);
        put(key, node.value);
        return node.value;
        
    }
    
    public void put(int key, int value) {
        if(map.containsKey(key)){
           Node node = map.get(key);
            cache.remove(node);
            node.value = value;
            map.put(key,node);
            cache.addFirst(node);
            return;
        }
        if (cache.getSize() == capacity){
           Node lastNode = cache.removeLast();
           map.remove(lastNode.key); 
        }
        Node newNode = new Node(key,value);
        map.put(key,newNode);
        cache.addFirst(newNode);
    }
}
```

## 937-Reorder Data in Log Files

```java
You have an array of logs.  Each log is a space delimited string of words.
For each log, the first word in each log is an alphanumeric identifier.  
Then, either:
Each word after the identifier will consist only of lowercase letters, or;
Each word after the identifier will consist only of digits.
We will call these two varieties of logs letter-logs and digit-logs.  
It is guaranteed that each log has at least one word after its identifier.
Reorder the logs so that all of the letter-logs come before any digit-log.  
The letter-logs are ordered lexicographically ignoring identifier, with the 
identifier used in case of ties.  The digit-logs should be put in their original
order. Return the final order of the logs.

Example 1:
Input: logs = ["dig1 8 1 5 1","let1 art can","dig2 3 6",
"let2 own kit dig","let3 art zero"]
Output: ["let1 art can","let3 art zero","let2 own kit dig",
"dig1 8 1 5 1","dig2 3 6"]

class Solution {
    public String[] reorderLogFiles(String[] logs) {
        List<String> letterLogs = new ArrayList<>();
        List<String> numLogs = new ArrayList<>();
        // 将字母日志和数字日志分开，分别放入两个list
        for (String log : logs) {
            int i = log.indexOf(" ") + 1;
            if (log.charAt(i) >= '0' && log.charAt(i) <= '9')
                numLogs.add(log);
            else
                letterLogs.add(log);
        }
        Collections.sort(letterLogs, (a,b) -> {
                // 取字母a日志的标识符及内容
                int ai = a.indexOf(" ");
                String ida = a.substring(0, ai);
                String loga = a.substring(ai + 1);

                // 取字母b日志的标识符及内容
                int bi = b.indexOf(" ");
                String idb = b.substring(0, bi);
                String logb = b.substring(bi + 1);
                
                // 对比二者内容，如果相同则对比标识符
                int cmp = loga.compareTo(logb);
                if (cmp == 0) 
                    return ida.compareTo(idb);
                return cmp;
            
        });
        letterLogs.addAll(numLogs);
        return letterLogs.toArray(new String[letterLogs.size()]);
    }
}


```

## 5-Longest Palindromic Substring

```java
class Solution {
    int lo = 0; int maxLen = 0;
    public String longestPalindrome(String s) {
        if (s.length() < 2) return s;
        for (int i = 0; i < s.length(); i ++) {
            extendedPalindrome(s, i , i);
            extendedPalindrome(s, i, i + 1);
        }
        return s.substring(lo, lo + maxLen);
    }
    public void extendedPalindrome(String s, int l, int r){
      //  int l = left; int r = right; //作用域
        while (l >= 0 && l < s.length() && r >= 0 
        && r < s.length() && s.charAt(l) == s.charAt(r)){
            l --;
            r ++;
        }
        if (maxLen < r - l - 1){ // 
            lo = l + 1;
            maxLen = r - l - 1;
        }
    }
}
```

## 21-Merge Two Sorted Lists

```java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        if (l2 == null) return l1;
        if (l1.val < l2.val){
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        }
        l2.next = mergeTwoLists(l1, l2.next);
        return l2;
        }
        
    }

```

## 763-Partition Labels

```java
A string S of lowercase English letters is given.
 We want to partition this string into as many parts as possible so that each 
 letter appears in at most one part, and return a list of integers 
 representing the size of these parts.
Example 1:
Input: S = "ababcbacadefegdehijhklij"
Output: [9,7,8]
Explanation:
The partition is "ababcbaca", "defegde", "hijhklij".
This is a partition so that each letter appears in at most one part.
A partition like "ababcbacadefegde", "hijhklij" 
is incorrect, because it splits S into less parts.

class Solution {
    public List<Integer> partitionLabels(String S) {
        if(S == null || S.length() == 0){
            return null;
        }
        List<Integer> list = new ArrayList<>();
        int[] map = new int[26];  
       // record the last index of the each char
        for(int i = 0; i < S.length(); i++){
            map[S.charAt(i)-'a'] = i;
        }
        // record the end index of the current sub string
        int last = 0;
        int start = 0;
        for(int i = 0; i < S.length(); i++){
            last = Math.max(last, map[S.charAt(i)-'a']);
            if(last == i){
                list.add(last - start + 1);
                start = last + 1;
            }
        }
        return list;
    }
}
```

## 56.Merge Intervals

```java
class Solution {
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, (a,b) -> a[0] - b[0]); // -> not =>
        int k = 0;
        int i = 0;
        while (i < intervals.length) {
            int start = intervals[i][0];
            int end = intervals[i][1];
            while (i < intervals.length - 1 && end >= intervals[i + 1][0]){ 
            // test overlap need to first check end with next start
                i ++;
                end = Math.max(intervals[i][1], end);
            }
            intervals[k][0] = start;
            intervals[k][1] = end;
            k ++;
            i ++; 
        }
        return Arrays.copyOf(intervals, k);
    }
}
```

## 238-Product of Array Except Self

```java
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] front = new int[n]; int[] back = new int[n]; int[] res = new int[n];
        front[0] = 1;
        back[n - 1] = 1;
        for (int i = 1; i < n; i ++) {
            front[i] = nums[i - 1] * front[i - 1];
        }
        
        for (int j = n - 2; j >= 0; j --) {
            back[j] = back[j + 1] * nums[j + 1];
        }
        
        for (int i = 0; i < n; i ++) {
            res[i] = front[i] * back[i];
        }
        return res;
    }
}
```

## 23-Merge k Sorted Lists

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

## 273-Integer to English Words

```java
class Solution {
    public String numberToWords(int num) {
        if(num == 0) return "Zero";
        return helper(num);  
    }
    
    public String helper(int num ) {
        String[] words = new String[] {"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
                                      "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen",
                                       "Eighteen", "Nineteen"}; // Fifteen, Twelve Forty Nineteen Ninety Hundred
        String[] words1 = new String[]{"","","Twenty ", "Thirty ", "Forty ", "Fifty ", "Sixty ",  "Seventy ", "Eighty ", "Ninety "};
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

## 139-Word Break

![](../.gitbook/assets/wordbreak.png)

```java
方法一：动态规划

初始化 dp=[False,\cdots,False]dp=[False,⋯,False]，长度为 n+1n+1。
nn 为字符串长度。dp[i]dp[i] 表示 ss 的前 ii 位是否可以用 wordDictwordDict 中的单词表示。
初始化 dp[0]=Truedp[0]=True，空字符可以被表示。
遍历字符串的所有子串，遍历开始索引 ii，遍历区间 [0,n)[0,n)：
遍历结束索引 jj，遍历区间 [i+1,n+1)[i+1,n+1)：
若 dp[i]=Truedp[i]=True 且 s[i,\cdots,j)s[i,⋯,j) 
在 wordlistwordlist 中：dp[j]=Truedp[j]=True。解释：dp[i]=Truedp[i]=True 
说明 ss 的前 ii 位可以用 wordDictwordDict 表示，则 s[i,\cdots,j)s[i,⋯,j) 
出现在 wordDictwordDict 中，说明 ss 的前 jj 位可以表示。
返回 dp[n]dp[n]
复杂度分析
时间复杂度：O(n^{2})O(n 
2
 )
空间复杂度：O(n)O(n)

class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        boolean[] dp = new boolean[s.length() + 1];
        Set<String> set = new HashSet<>(wordDict);
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && set.contains(s.substring(j, i))){
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }
}
```

## 973-K Closest Points to Origin

```java
class Solution {
    public int[][] kClosest(int[][] points, int K) {
    PriorityQueue<int[]> pq = new PriorityQueue<int[]>((p1,p2) -> p2[0] * p2[0] + p2[1] * p2[1] - p1[0] * p1[0] - p1[1] * p1[1]);
    for (int i = 0; i < points.length; i ++){
        pq.offer(points[i]);
        if (pq.size() > K){
            pq.poll();
        }
    }
    int[][] res = new int[K][2];
                                                    
    for (int i = K - 1; i >= 0; i --){
        res[i] = pq.poll();
    }
     return res;                                               
    }
}
```

## 221-Maximal Square

![](../.gitbook/assets/image\_1573111823.png)

```java
Given a 2D binary matrix filled with 0's and 1's, find
the largest square containing only 1's and return its area.
Example:
Input: 
1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0
Output: 4

dp[i][j] represents the length of the square which lower right corner 
is located at (i, j).
If the value of this cell is also 1, then the length of the square is the 
minimum of: the one above, its left, and diagonal up-left value +1. 
Because if one side is short or missing, it will not form a square.

class Solution {
    public int maximalSquare(char[][] matrix) {
      if(matrix.length == 0) return 0;
      int m = matrix.length, n = matrix[0].length;
      int[][] dp = new int[m + 1][n + 1];
   
      int maxEdge = 0;      
      for(int i = 1; i <= m; i++){
        for(int j = 1; j <= n; j++){
          if(matrix[i - 1][j - 1] == '1'){
            dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]),
            dp[i - 1][j - 1]) + 1;
            maxEdge = Math.max(maxEdge, dp[i][j]);
          }
        }
      }
      
      return maxEdge * maxEdge;  
    }
}
```

## 819- Most Common Word

```java
paragraph = "Bob hit a ball, the hit BALL flew far after it was hit."
banned = ["hit"]
Output: "ball"
Explanation: 
"hit" occurs 3 times, but it is a banned word.
"ball" occurs twice (and no other word does), 
so it is the most frequent non-banned word in the paragraph. 

class Solution {
    public String mostCommonWord(String paragraph, String[] banned) {
        if (paragraph.length()==0) {
            return "";
        }
        String result = "";
        HashMap<String, Integer> map = new HashMap<>();
        String[] words = paragraph.replaceAll("\\W+" , " ")
                         .toLowerCase().split("\\s+");
        for (String word : words) {
            if (map.containsKey(word)){
                map.put(word,map.get(word) + 1);
            } else {
                map.put(word,1);
            }
        }
        
        for (String ban : banned) {
            if (map.containsKey(ban)) {
                map.remove(ban); // use remove method to ban
            }
        }
        for (Map.Entry<String, Integer> wordEntry : map.entrySet()) {
         if (result.length() == 0 || wordEntry.getValue() > map.get(result)) {
            result = wordEntry.getKey();
         }
      }
       return result;
    }
}
```

## 682- Baseball Game

```java
An integer x - Record a new score of x.
"+" - Record a new score that is the sum of the previous two scores. 
It is guaranteed there will always be two previous scores.
"D" - Record a new score that is double the previous score. 
It is guaranteed there will always be a previous score.
"C" - Invalidate the previous score, removing it from the record. 
It is guaranteed there will always be a previous score.
Return the sum of all the scores on the record.
Example 1:

Input: ops = ["5","2","C","D","+"]
Output: 30
Explanation:
"5" - Add 5 to the record, record is now [5].
"2" - Add 2 to the record, record is now [5, 2].
"C" - Invalidate and remove the previous score, record is now [5].
"D" - Add 2 * 5 = 10 to the record, record is now [5, 10].
"+" - Add 5 + 10 = 15 to the record, record is now [5, 10, 15].
The total sum is 5 + 10 + 15 = 30.

class Solution {
    public int calPoints(String[] ops) {
        Stack<Integer> stack = new Stack();

        for(String op : ops) {
            if (op.equals("+")) {
                int top = stack.pop();
                int newtop = top + stack.peek();
                stack.push(top);
                stack.push(newtop);
            } else if (op.equals("C")) {
                stack.pop();
            } else if (op.equals("D")) {
                stack.push(2 * stack.peek());
            } else {
                stack.push(Integer.valueOf(op));
            }
        }

        int ans = 0;
        for(int score : stack) ans += score;
        return ans;
    }
}

```

## 1010- Pairs of Songs With Total Durations Divisible by 60

```java
Input: [30,20,150,100,40]
Output: 3
Explanation: Three pairs have a total duration divisible by 60:
(time[0] = 30, time[2] = 150): total duration 180
(time[1] = 20, time[3] = 100): total duration 120
(time[1] = 20, time[4] = 40): total duration 60

整数对60取模，可能有60种余数。故初始化一个长度为60的数组，统计各余数出现的次数。
遍历time数组，每个值对60取模，并统计每个余数值（0-59）出现的个数。
因为余数部分需要找到合适的cp组合起来能被60整除。
余数为0的情况，只能同余数为0的情况组合（如60s、120s等等）。
0的情况出现k次，则只能在k中任选两次进行两两组合。用k * (k - 1) / 2表示。
余数为30的情况同上。
其余1与59组合，2与58组合，故使用双指针分别从1和59两头向中间遍历。
1的情况出现m次，59的情况出现n次，则总共有m*n种组合。

class Solution {
    public int numPairsDivisibleBy60(int[] time) {
        int count = 0;
		int[] seconds = new int[60];
		for(int t : time) {
			seconds[t % 60] += 1; 
		}
		count += seconds[30] * (seconds[30] - 1) / 2;
		count += seconds[0] * (seconds[0] - 1) / 2;
		int i = 1, j = 59;
		while(i < j) {
			count += seconds[i++] * seconds[j--];
		}
		return count;
	}
}
```

## 994-Rotting Oranges

```java
一开始，我们找出所有腐烂的橘子，将它们放入队列，作为第 0 层的结点。
然后进行 BFS 遍历，每个结点的相邻结点可能是上、下、左、右四个方向的结点，
注意判断结点位于网格边界的特殊情况。
由于可能存在无法被污染的橘子，我们需要记录新鲜橘子的数量。
在 BFS 中，每遍历到一个橘子（污染了一个橘子），就将新鲜橘子的数量减一。
如果 BFS 结束后这个数量仍未减为零，说明存在无法被污染的橘子。

class Solution {
    public int orangesRotting(int[][] grid) {
    int M = grid.length;
    int N = grid[0].length;
    Queue<Integer> queue = new LinkedList<>();

    int count = 0; // count 表示新鲜橘子的数量
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            if (grid[r][c] == 1) {
                count++;
            } else if (grid[r][c] == 2) {
                queue.add(r*N + c);
            }
        }
    }

    int round = 0; // round 表示腐烂的轮数，或者分钟数
    while (count > 0 && !queue.isEmpty()) {
        round++;
        int n = queue.size();
        for (int i = 0; i < n; i++) {
            int orange = queue.poll();
            int r = orange/N;
            int c = orange%N;
            if (r-1 >= 0 && grid[r-1][c] == 1) {
                grid[r-1][c] = 2;
                count--;
                queue.add((r-1)*N + c);
            }
            if (r+1 < M && grid[r+1][c] == 1) {
                grid[r+1][c] = 2;
                count--;
                queue.add((r+1)*N + c);
            }
            if (c-1 >= 0 && grid[r][c-1] == 1) {
                grid[r][c-1] = 2;
                count--;
                queue.add(r*N + c - 1);
            }
            if (c+1 < N && grid[r][c+1] == 1) {
                grid[r][c+1] = 2;
                count--;
                queue.add(r*N + c + 1);
            }
        }
    }

    if (count > 0) {
        return -1;
    } else {
        return round;
    }
}
    
}
```

## 127-Word Ladder

```java
Input:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
         HashSet<String> wordSet = new HashSet<>(wordList);
         if(!wordSet.contains(endWord)) return 0;
         Queue<String> queue = new LinkedList<>();
         HashSet<String> visited = new HashSet<>();
         queue.offer(beginWord);
         visited.add(beginWord);
         int steps = 0;
         while (!queue.isEmpty()){
             steps ++;
             int size = queue.size();
             for (int i = 0; i < size; i ++){
                 String word = queue.poll();
                if(word.equals(endWord)) return steps;
                 for (int j = 0; j < word.length(); j ++){
                     char[] letters = word.toCharArray();
                     for (char l = 'a'; l <= 'z'; l ++){
                         letters[j] = l;
                         String nextWord = new String(letters);
                         if(!visited.contains(nextWord) 
                         && wordSet.contains(nextWord)){
                             queue.offer(nextWord);
                             visited.add(nextWord);
                         }
                     }
                 }
             }
         }
         return 0;    
         
        
    }
}
```

## 212-Word Search II

```java
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

## 210-Course Schedule II

```java
class Solution {
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] indegrees = new int[numCourses];
        List<ArrayList<Integer>> adjTable = new ArrayList<>();
        ArrayList<Integer> res = new ArrayList<>();
        for (int i = 0; i < numCourses; i ++) {
            adjTable.add(new ArrayList<Integer>());
        }
        for (int[] pre : prerequisites){
            indegrees[pre[0]] ++;
            adjTable.get(pre[1]).add(pre[0]);
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < indegrees.length; i ++) {
            if(indegrees[i] == 0){
                queue.offer(i);
            }
        }
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            numCourses --;
            res.add(cur);
            List<Integer> adj = adjTable.get(cur);
            for (int next : adj) {
                indegrees[next] --;
                if (indegrees[next] == 0) {
                    queue.offer(next);
                }
            }
        }
        if (numCourses != 0) return new int[0];
        int[] finalRes = new int[res.size()];
        for (int i = 0; i < res.size(); i ++){
            finalRes[i] = res.get(i);
        }
        return finalRes;
    }
}
```

## 348-Design Tic-Tac-Toe

```java
Input
["TicTacToe", "move", "move", "move", "move", "move", "move", "move"]
[[3], [0, 0, 1], [0, 2, 2], [2, 2, 1], [1, 1, 2], [2, 0, 1], [1, 0, 2], [2, 1, 1]]
Output
[null, 0, 0, 0, 0, 0, 0, 1]

public class TicTacToe {
private int[] rows;
private int[] cols;
private int diagonal;
private int antiDiagonal;

/** Initialize your data structure here. */
public TicTacToe(int n) {
    rows = new int[n];
    cols = new int[n];
}

/** Player {player} makes a move at ({row}, {col}).
    @param row The row of the board.
    @param col The column of the board.
    @param player The player, can be either 1 or 2.
    @return The current winning condition, can be either:
            0: No one wins.
            1: Player 1 wins.
            2: Player 2 wins. */
public int move(int row, int col, int player) {
    int toAdd = player == 1 ? 1 : -1;
    
    rows[row] += toAdd;
    cols[col] += toAdd;
    if (row == col)
    {
        diagonal += toAdd;
    }
    
    if (col == (cols.length - row - 1))
    {
        antiDiagonal += toAdd;
    }
    
    int size = rows.length;
    if (Math.abs(rows[row]) == size ||
        Math.abs(cols[col]) == size ||
        Math.abs(diagonal) == size  ||
        Math.abs(antiDiagonal) == size)
    {
        return player;
    }
    
    return 0;
}
}
```

## 957-Prison Cells After N Days

```java


class Solution {
    public int[] prisonAfterNDays(int[] cells, int N) {
		if(cells==null || cells.length==0 || N<=0) return cells;
        boolean hasCycle = false;
        int cycle = 0;
        HashSet<String> set = new HashSet<>(); 
        for(int i=0;i<N;i++){
            int[] next = nextDay(cells);
            String key = Arrays.toString(next);
            if(!set.contains(key)){ //store cell state
                set.add(key);
                cycle++;
            }
            else{ //hit a cycle
                hasCycle = true;
                break;
            }
            cells = next;
        }
        if(hasCycle){
            N%=cycle;
            for(int i=0;i<N;i++){
                cells = nextDay(cells);
            }   
        }
        return cells;
    }
    
    private int[] nextDay(int[] cells){
        int[] tmp = new int[cells.length];
        for(int i=1;i<cells.length-1;i++){
            tmp[i]=cells[i-1]==cells[i+1]?1:0;
        }
        return tmp;
    }
}
```

## 138 -Copy List with Random Pointer

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

##

##

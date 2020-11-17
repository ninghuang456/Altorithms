---
description: Facebook
---

# Farebook Frequency 1-50

## 953 - Verifying an Alien Dictionary

```java
//Input: words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz" Output: true
//Explanation: As 'h' comes before 'l' in this language, then the sequence is sorted.
class Solution {
      int[] dic = new int[26]; // 26 not order.length()
    public boolean isAlienSorted(String[] words, String order) {
        for (int i = 0; i < order.length(); i ++) {
            char c = order.charAt(i);
            dic[c - 'a'] = i;
        }
        
        for (int i = 1; i < words.length; i ++){
            if(bigger(words[i - 1], words[i])){
                return false;
            }
        }
        return true;  
    }
    
    public boolean bigger(String w1, String w2){
        int m = w1.length(); int n = w2.length();
        for (int i = 0; i < m && i < n; i ++) {
            if (w1.charAt(i)!= w2.charAt(i)){
                return dic[w1.charAt(i) - 'a'] > dic[w2.charAt(i) - 'a'];
                //need return in this time.
            }
        }
        return m > n;// 记住这个要后面判断
    }
}
```

## 1249- Minimum Remove to Make Valid Parentheses

```java
//Input: s = "lee(t(c)o)de)" Output: "lee(t(c)o)de"
//Explanation: "lee(t(co)de)" , "lee(t(c)ode)" would also be accepted.

class Solution {
  public String minRemoveToMakeValid(String s) { // 子问题不独立 所以不用DP
      StringBuilder sb = new StringBuilder(s); 
      Stack<Integer> st = new Stack<>();
      // 用stack 判断是否当前的括号需要换掉
      for (int i = 0; i < sb.length(); ++i) {
         if (sb.charAt(i) == '(') st.add(i);
         // stack 存的是位置 不是CHAR
          if (sb.charAt(i) == ')') {
           if (!st.empty()) st.pop();
              else sb.setCharAt(i, '*');
              // 因为SB是根据S变得 所以是要setCharAt() 而不是APPEND
         }
  }
  while (!st.empty())
    sb.setCharAt(st.pop(), '*');
  return sb.toString().replaceAll("\\*", "");
  }
}

```

## 1428- Leftmost Column with at Least a One

```java
class Solution {
// 	至少有一个 1 的最左端列  
    public int leftMostColumnWithOne(BinaryMatrix binaryMatrix) {
        List<Integer> dim = binaryMatrix.dimensions();
        int r = dim.get(0); int c = dim.get(1);
        int left = 0; int right = c - 1; int res = -1;
        while (left <= right){
            int mid = left + (right - left) / 2;
            if(containsOne(r, mid, binaryMatrix)){
                right = mid - 1;
                res = mid; // record mid since it may have no such condition.
                // To find left most we reduce right side once it satisfied condition.
            } else {
                left = mid + 1;
            }
        
        }
     return res;
        
    }
    public  boolean containsOne(int r, int c, BinaryMatrix binaryMatrix){
        for (int i = 0; i < r; i ++) {
            if (binaryMatrix.get(i, c) == 1) return true;
        }
        return false;
    }
}
```

## 560 - Subarray Sum Equals K

```java
// Input:nums = [1,1,1], k = 2 Output: 2
//Input: nums = [1,2,3], k = 3 Output: 2

class Solution {
    public int subarraySum(int[] nums, int k) {
        int sum = 0; int res = 0; // sum[i, j] = sum[0, j] - sum[0, i - 1];
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(0,1); // don't forget this, we have one presum equals 0;
        // key is sum value, is nums of same key.
        for (int i = 0; i < nums.length; i ++) {
            sum += nums[i];
            if (map.containsKey(sum - k)){
                res += map.get(sum - k); 
         // may be has more than one presum equals same number.
            }
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return res;
    }
}
```

## 680 Valid Palindrome II

```java
Given a non-empty string s, you may delete at most 
one character. Judge whether you can make it a palindrome.
// Input: "aba" Output: True
class Solution {
    public boolean validPalindrome(String s) {
        int left = 0; int right = s.length() - 1;
        while (left < right) {
            if (s.charAt(left) != s.charAt(right)){
                return extendPalindrome(s,left + 1, right) || 
                 extendPalindrome(s,left, right - 1);
                 // 把没走完的路继续下去 因为只让挪动一个数字
            }
            left ++;
            right--;
        }
        return true;
    }
    
    public boolean extendPalindrome(String s, int left, int right){
        int l = left; int r = right;
        while (l < r){
            if (s.charAt(l) != s.charAt(r)) return false;
            l++;
            r--;
        }
        return true;
    }
}
```

## 973-K Closest Points to Origin

```java
class Solution {
    public int[][] kClosest(int[][] points, int K) {
    PriorityQueue<int[]> pq = new PriorityQueue<int[]>((p1,p2) -> 
p2[0] * p2[0] + p2[1] * p2[1] - p1[0] * p1[0] - p1[1] * p1[1]);
   // P2 - P1, int[]
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

## 238-Product of Array Except Self

```java
// Input:  [1,2,3,4] Output: [24,12,8,6]
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] front = new int[n]; int[] back = new int[n]; 
        int[] res = new int[n];
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

## 273- Integer to English Words

```java
Convert a non-negative integer num to its English words representation.
Example 1: Input: num = 123 Output: "One Hundred Twenty Three"

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

## 415 - Add Strings

```java
Given two non-negative integers num1 and num2 
represented as string, return the sum of num1 and num2.

class Solution {
    public String addStrings(String num1, String num2) {
        StringBuilder res = new StringBuilder("");
        int i = num1.length() - 1, j = num2.length() - 1, carry = 0;
        //从后往前加 用最后一个数字开始
        while(i >= 0 || j >= 0){
            int n1 = i >= 0 ? num1.charAt(i) - '0' : 0; 
            //因为有可能一长一短的情况
            int n2 = j >= 0 ? num2.charAt(j) - '0' : 0;
            int tmp = n1 + n2 + carry;
            carry = tmp / 10;
            res.append(tmp % 10);
            i--; j--;
        }
        if(carry == 1) res.append(1);
        return res.reverse().toString();
    }
}
```

## 67- Add Binary

```java
Given two binary strings, return their sum (also a binary string).
The input strings are both non-empty and contains only characters 1 or 0.
Example 1: Input: a = "11", b = "1" Output: "100"

class Solution {
    public String addBinary(String a, String b) {
        StringBuilder sb = new StringBuilder();
        int i = a.length() - 1; int j = b.length() - 1; int carry = 0;
        while (i >= 0  || j >= 0) {
            int m = i >= 0 ? a.charAt(i) - '0' : 0;
            // also - '0';
            int n = j >= 0 ? b.charAt(j) - '0' : 0;
            int sum = m + n + carry;
            carry = sum / 2;
            sb.append(sum % 2);
            i --; j --;
        }
        if (carry > 0) sb.append(carry);
        return sb.reverse().toString();
    }
}
```

## 269-Alien Dictionary

```java
//[
//  "wrt",
//  "wrf",
//  "er",
//  "ett",
//  "rftt"
//]
//Output: "wertf"
//该题是要在前后单词之间寻找排序关系 不是在本单词中找

class Solution {
public String alienOrder(String[] words) {
    
    // Step 0: Create data structures and find all unique letters.
    Map<Character, List<Character>> adjList = new HashMap<>();
    Map<Character, Integer> counts = new HashMap<>();
    for (String word : words) {
        for (char c : word.toCharArray()) {
            counts.put(c, 0);
            adjList.put(c, new ArrayList<>());
        }
    }
    
    // Step 1: Find all edges.
    for (int i = 0; i < words.length - 1; i++) {
        String word1 = words[i];
        String word2 = words[i + 1];
        // Check that word2 is not a prefix of word1.
        if (word1.length() > word2.length() && word1.startsWith(word2)) {
            return "";
        }
        // Find the first non match and insert the corresponding relation.
        for (int j = 0; j < Math.min(word1.length(), word2.length()); j++) {
            if (word1.charAt(j) != word2.charAt(j)) {
                adjList.get(word1.charAt(j)).add(word2.charAt(j));
                counts.put(word2.charAt(j), counts.get(word2.charAt(j)) + 1);
                break;
            }
        }
    }
    
    // Step 2: Breadth-first search.
    StringBuilder sb = new StringBuilder();
    Queue<Character> queue = new LinkedList<>();
    for (Character c : counts.keySet()) {
        if (counts.get(c).equals(0)) {
            queue.add(c);
        }
    }
    while (!queue.isEmpty()) {
        Character c = queue.remove();
        sb.append(c);
        for (Character next : adjList.get(c)) {
            counts.put(next, counts.get(next) - 1);
            if (counts.get(next).equals(0)) {
                queue.add(next);
            }
        }
    }
    
    if (sb.length() < counts.size()) {
        return "";
    }
    return sb.toString();
  }
}
```

## 301 - Remove Invalid Parentheses

```java
Remove the minimum number of invalid parentheses in order 
to make the input string valid. Return all possible results.
//Input: "()())()"  Output: ["()()()", "(())()"]
Input: "(a)())()"
Output: ["(a)()()", "(a())()"]

class Solution {
    public List<String> removeInvalidParentheses(String s) {
        List<String> result = new ArrayList<>();
        if (s == null) {
            return result;
        }
        Set<String> visited = new HashSet<>();
        Queue<String> queue = new LinkedList<>();//存储状态
        //初始化
        visited.add(s);
        queue.add(s);
        boolean found = false;//在某个level是否发现valid的状态
        while (!queue.isEmpty()) {
            s = queue.poll();
            if (isValid(s)) {
                //找到一个答案，加入到结果集
                result.add(s);
                found = true;
            }
            if (found) {// 不用再把下面一层再加入了 只要循环完这一轮就可以了
                continue;
            }
            //产生所有的状态
            for (int i = 0; i < s.length(); i++) {
                //只移除左括号或右括号，字母什么的不移除
                if (s.charAt(i) != '(' && s.charAt(i) != ')') {
                    continue;
                }
                //产生一个临时状态
                String temp = s.substring(0, i) + s.substring(i + 1);
               //为什么这里i + 1不会越界？
                if (!visited.contains(temp)) {
                    queue.add(temp);
                    visited.add(temp);
                }
            }
        }
        return result;
    }
    
    public boolean isValid(String s){
        int count = 0;
        for (int i = 0; i < s.length(); i++){
            if(s.charAt(i)=='(') count++;
            if(s.charAt(i)==')') count--;
            if(count < 0) return false;
        }
        return count == 0;
    }
}
```

## 158- Read N Characters Given Read4 II - Call multiple times

```java
public class Solution extends Reader4 {
    int size = 0;
    int i = 0;
    char[] temp = new char[4];
    public int read(char[] buf, int n) {
        int index = 0;
        while(index < n){
            if(size == 0){
                size = read4(temp);
                if(size == 0)
                    break;
            }      
            while(index < n && i < size){
                buf[index++] = temp[i++];
            }
            if(i == size){
                // 说明临时字符数组中的内容已经读完，size置零以便执行下一次read4操作
                i = 0;
                size = 0;
            }     
        }
        
        return index;
    }
}
```

## 211 Design Add and Search Words Data Structure

```java
class WordDictionary {
     boolean isEnd;
     WordDictionary[] next;
    /** Initialize your data structure here. */
    public WordDictionary() {
        isEnd = false;
        next = new WordDictionary[26];
    }
    /** Adds a word into the data structure. */
    public void addWord(String word) {
        WordDictionary root = this;
        WordDictionary cur = root;
        for (char c : word.toCharArray()){
            if (cur.next[c - 'a'] == null){
               WordDictionary nodeNext = new WordDictionary();
               cur.next[c-'a'] = nodeNext;
            }
            cur = cur.next[c - 'a']; 
        }
      cur.isEnd = true;  
    }
    
    /** Returns if the word is in the data structure. 
     A word could contain the dot character '.' to represent any one letter. */
    public boolean search(String word) {
        return wordSearchHelper(word,0, this);
        
    }
    
    public boolean wordSearchHelper(String word, int index, WordDictionary cur){
        if(index == word.length()) return cur.isEnd;
        char c = word.charAt(index);
        if(c != '.'){
            return (cur.next[c - 'a'] != null) && 
            wordSearchHelper(word, index + 1, cur.next[c - 'a']);
        } else {
            for (char k = 'a'; k <= 'z'; k++){
                if(cur.next[k - 'a'] != null && 
                  wordSearchHelper(word, index + 1, cur.next[k - 'a'])){
                    return true;
                }
            }
        }
        return false;
    }
}
```

## 297- Serialize and Deserialize Binary Tree

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

## 215- Kth Largest Element in an Array

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
    PriorityQueue<Integer> pq = new PriorityQueue<>();
    // default size will resize automatically
    for(int val : nums) {
        pq.offer(val);
        if(pq.size() > k) {
            pq.poll();
        }
    }
    return pq.peek();
  }
}
```

## 523- Continuous Subarray Sum

```java
Given a list of non-negative numbers and a target integer k,
 write a function to check if the array has a continuous 
 subarray of size at least 2 that sums up to a multiple of k, 
that is, sums up to n*k where n is also an integer.

Input: [23, 2, 4, 6, 7],  k=6 Output: True
Explanation: Because [2, 4] is a continuous subarray of size 2 and sums up to 6.
Input: [23, 2, 6, 4, 7],  k=6 Output: True
Explanation: Because [23, 2, 6, 4, 7] is an continuous subarray of 
size 5 and sums up to 42.

class Solution {
  public boolean checkSubarraySum(int[] nums, int k) {
    Map<Integer, Integer> map = new HashMap<Integer, Integer>(){{put(0,-1);}};;
    int runningSum = 0;
    for (int i=0; i < nums.length;i++) {
        runningSum += nums[i];
        if (k != 0) runningSum %= k; 
        Integer prev = map.get(runningSum);
        if (prev != null) {
            if (i - prev > 1) return true;
        }
        else map.put(runningSum, i);
    }
    return false;
  }
}

```

## 438-Find All Anagrams in a String

```java
Input: s: "cbaebabacd" p: "abc" Output: [0, 6]
class Solution {
    public List<Integer> findAnagrams(String s, String t) {
        HashMap<Character, Integer> window = new HashMap<>(); 
        HashMap<Character, Integer> need = new HashMap<>();
        List<Integer> res = new ArrayList<>();
        for (char t1 : t.toCharArray()) {
            need.put(t1, need.getOrDefault(t1,0) + 1);
        }
        int left = 0; int right = 0; int valid = 0;
        while (right < s.length()) { 
   // 区间[left, right)是左闭右开的，所以初始情况下窗口没有包含任何元素：
            char s1 = s.charAt(right);
            right ++;
            if (need.containsKey(s1)) {
                window.put(s1,window.getOrDefault(s1,0) + 1);
                if (window.get(s1).equals(need.get(s1))){
                    valid ++;
                }
            }          
    // 右指针移动当于在寻找一个「可行解」，然后移动左指针在优化这个「可行解」，最终找到最优解
            while (right - left >= t.length()) {
                 if (valid == need.size()) res.add(left);
                  char s2 = s.charAt(left);
                  left ++;
                  if (need.containsKey(s2)) { 
                 // every time need containsKey not contains
                   if (window.get(s2).equals(need.get(s2))){
                      valid --;
                   }
                    window.put(s2,window.get(s2) - 1);
                }
            }
        }
        return res;  
    }
}
```

## 938-Range Sum of BST

```java
Given the root node of a binary search tree, return the
 sum of values of all nodes with value between L and R (inclusive).
Input: root = [10,5,15,3,7,null,18], L = 7, R = 15
Output: 32
对于当前节点 node，如果 node.val 小于等于 L，那么只需要继续搜索它的右子树；
如果 node.val 大于等于 R，那么只需要继续搜索它的左子树；
如果 node.val 在区间 (L, R) 中，则需要搜索它的所有子树。
class Solution { // 递归
    int ans;
    public int rangeSumBST(TreeNode root, int L, int R) {
        ans = 0;
        dfs(root, L, R);
        return ans;
    }

    public void dfs(TreeNode node, int L, int R) {
        if (node != null) {
            if (L <= node.val && node.val <= R)
                ans += node.val;
            if (L < node.val)
                dfs(node.left, L, R);
            if (node.val < R)
                dfs(node.right, L, R);
        }
    }
}
//迭代
class Solution {
    public int rangeSumBST(TreeNode root, int L, int R) {
        int ans = 0;
        Stack<TreeNode> stack = new Stack();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            if (node != null) {
                if (L <= node.val && node.val <= R)
                    ans += node.val;
                if (L < node.val)
                    stack.push(node.left);
                if (node.val < R)
                    stack.push(node.right);
            }
        }
        return ans;
    }
}


```

## 282- Expression Add Operators

```java
Given a string that contains only digits 0-9 and a target value, 
return all possibilities to add binary operators (not unary) +, -, or * 
between the digits so they evaluate to the target value.
Input: num = "123", target = 6
Output: ["1+2+3", "1*2*3"] 
public class Solution {
    public List<String> addOperators(String num, int target) {
        List<String> rst = new ArrayList<String>();
        if(num == null || num.length() == 0) return rst;
        helper(rst, "", num, target, 0, 0, 0);
        return rst;
    }
    public void helper(List<String> rst, String path, String num, 
                     int target, int pos, long eval, long multed){
        if(pos == num.length()){
            if(target == eval)
                rst.add(path);
            return;
        }
        for(int i = pos; i < num.length(); i++){
            if(i != pos && num.charAt(pos) == '0') break;
            long cur = Long.parseLong(num.substring(pos, i + 1));
            if(pos == 0){
                helper(rst, path + cur, num, target, i + 1, cur, cur);
            }
            else{
            helper(rst, path + "+" + cur, num, target, i + 1, eval + cur , cur);                           
            helper(rst, path + "-" + cur, num, target, i + 1, eval -cur, -cur);
            helper(rst, path + "*" + cur, num, target, i + 1, 
             eval - multed + multed * cur, multed * cur );
            }
        }
    }
}
```

## 65- Valid Number

```java
//"0" => true " 0.1 " => true "abc" => false "1 a" => false "2e10" => true 
// " -90e3   " => true " 1e" => false "e3" => false " 6e-1" => true 
// " 99e2.5 " => false "53.5e93" => true " --6 " => false "-+3" => false 
// "95a54e53" => false

class Solution {
    public boolean isNumber(String s) {
        if(s==null||s.length()==0) return false;
        boolean numSeen=false;
        boolean dotSeen=false;
        boolean eSeen=false;
        char arr[]=s.trim().toCharArray();
        for(int i=0; i<arr.length; i++){
            if(arr[i]>='0'&&arr[i]<='9'){
                numSeen=true;
            }else if(arr[i]=='.'){
                if(dotSeen||eSeen){
                    return false;
                }
                dotSeen=true;
            }else if(arr[i]=='E'||arr[i]=='e'){
                if(eSeen||!numSeen){
                    return false;
                }
                eSeen=true;
                numSeen=false;
            }else if(arr[i]=='+'||arr[i]=='-'){
                if(i!=0&&arr[i-1]!='e'&&arr[i-1]!='E'){
                    return false;
                }
            }else{
                return false;
            }
        }
        return numSeen;
    }
}
```

## 636- Exclusive Time of Functions

```java
//Input: n = 2, 
// logs = ["0:start:0","1:start:2","1:end:5","0:end:6"] Output: [3,4]
class Solution {
  public int[] exclusiveTime(int n, List<String> logs) {
    int[] res = new int[n];
    Stack<Integer> stack = new Stack<>();
    int prevTime = 0;
    for (String log : logs) {
        String[] parts = log.split(":");
   if (!stack.isEmpty()) res[stack.peek()] += Integer.valueOf(parts[2]) - prevTime; 
        prevTime = Integer.valueOf(parts[2]);
        if (parts[1].equals("start")) stack.push(Integer.valueOf(parts[0]));
        else {
            res[stack.pop()]++;
            prevTime++;
        }
    }
    return res;
  }
}
```

## 173-Binary Search Tree Iterator

```java
class BSTIterator {
    Stack<TreeNode> stack = new Stack<>();
    TreeNode cur = null;

    public BSTIterator(TreeNode root) {
        cur = root;
    }

    /** @return the next smallest number */
    public int next() {
        int res = -1;
        while (cur != null || !stack.isEmpty()) {
            // 节点不为空一直压栈
            while (cur != null) {
                stack.push(cur);
                cur = cur.left; // 考虑左子树
            }
            // 节点为空，就出栈
            cur = stack.pop();
            res = cur.val;
            // 考虑右子树
            cur = cur.right;
            break; 
// even sometime next() is o(h) but amortized time complex is o(1);
        }

        return res;
    }

    /** @return whether we have a next smallest number */
    public boolean hasNext() {
        return cur != null || !stack.isEmpty();
    }
}
/*只需要把 stack 和 cur 作为成员变量，然后每次调用 next 
就执行一次 while 循环，并且要记录当前值，结束掉本次循环。*/

```

## 124- Binary Tree Maximum Path Sum

```java
class Solution { //分两步策略 ：1 经过节点的路径最大， 
//2：全局变量对比出经过每一个节点的路径最大
   int max = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        if (root == null) {
            return 0;
        }
        dfs(root);
        return max;
    }
    public int dfs(TreeNode root) { // 后序遍历 左右根
        if (root == null) {
            return 0;
        }
        //计算左边分支最大值，左边分支如果为负数还不如不选择
        int leftMax = Math.max(0, dfs(root.left));
        //计算右边分支最大值，右边分支如果为负数还不如不选择
        int rightMax = Math.max(0, dfs(root.right));
        //left->root->right 作为路径与历史最大值做比较  
        // 更新遍历在当前节点时的最大路径和全局变量
        max = Math.max(max, root.val + leftMax + rightMax);
        //如果不考虑这个步骤 那就是经过节点的最大路径和
        // 选择以当前节点为根的含有最大值的路劲，左或右；返回给上一层递归的父节点作为路径
        return root.val + Math.max(leftMax, rightMax); // 不能左右同时返回
    }
}
```

## 528-Random Pick with Weight

```java
w[i] describes the weight of ith index (0-indexed).
We need to call the function pickIndex() which randomly
 returns an integer in the range [0, w.length - 1]. 
pickIndex() should return the integer proportional to its weight in the w array. 
For example, for w = [1, 3], 
the probability of picking the index 0 is 1 / (1 + 3) = 0.25 (i.e 25%) 
while the probability of picking the index 1 is 3 / (1 + 3) = 0.75 (i.e 75%).
More formally, the probability of picking index i is w[i] / sum(w).
Example 1:
Input
["Solution","pickIndex"]
[[[1]],[]]
Output
[null,0]

Explanation
Solution solution = new Solution([1]);
solution.pickIndex(); 
// return 0. Since there is only one single element on the array
// the only option is to return the first element.

class Solution {
    //权重累加数组
    int[] arr;
    public Solution(int[] w) {
        arr = new int[w.length];
        int sum = 0;
        for (int i = 0; i < w.length; i++) {
            sum += w[i];
            arr[i] = sum;
        }
    }
    public int pickIndex() {
        //产生随机数
        Random random = new Random();
        int randomNum = random.nextInt(arr[arr.length - 1]) + 1;
        //找到累加后的最后一个数
        //二分查找随机数所在的区间
        int left = 0, right = arr.length - 1; int ans = -1;
        while (left <= right) {
            int mid = left + ((right - left) / 2);
            if (arr[mid] == randomNum) {
                return mid;
            } else if (arr[mid] > randomNum) {
                right = mid - 1;
                ans = mid;
            } else {
                left = mid + 1;
            }
        }
        return ans;
    }
}
```

## 125- Valid Palindrome

```java
Given a string, determine if it is a palindrome,
considering only alphanumeric characters and ignoring cases.
Input: "A man, a plan, a canal: Panama"
Output: true
class Solution {
    public boolean isPalindrome(String s) {
        int left = 0;
        int right = s.length() - 1;
        while (left < right) {
       while(left < right && !Character.isLetterOrDigit(s.charAt(left))) 
          left ++;
       while(left < right && ! Character.isLetterOrDigit(s.charAt(right))) 
          right --;
       if(Character.toLowerCase(s.charAt(left)) 
                              != Character.toLowerCase(s.charAt(right)))
        return false;
            left ++;
            right--;
        }
       return true;
    }
}
```

## 278-First Bad Version

```java
Input: n = 5, bad = 4
Output: 4
Explanation:
call isBadVersion(3) -> false
call isBadVersion(5) -> true
call isBadVersion(4) -> true
Then 4 is the first bad version.

public class Solution extends VersionControl {
    public int firstBadVersion(int n) {
        int left = 1; int right = n; int ans = 0;
        while (left <= right){
            int mid = left + (right - left)/2;
            if (isBadVersion(mid)){
                right = mid - 1;
                ans = mid;
            } else {
                left = mid + 1;
            }
        }
        return ans;
    }
}
```

## 88-Merge Sorted Array

```java
addNum(1)
addNum(2)
findMedian() -> 1.5
addNum(3) 
findMedian() -> 2

class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int last = m + n - 1;
        int index1 = m - 1;
        int index2 = n - 1;
        while (index1 >= 0 && index2 >= 0){
            if (nums1[index1] > nums2[index2]){
                nums1[last--] = nums1[index1--];
            } else {
                nums1[last--] = nums2[index2--];
            }
        }
        while (index2 >= 0){
            nums1[last--] = nums2[index2--];
        }
    }
}
```

## 295- Find Median from Data Stream

```java
class MedianFinder {
   private Queue<Integer> small = new PriorityQueue<>((o1,o2) -> (o2 - o1)); // need <>
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

## 766-Toeplitz Matrix

```java
A matrix is Toeplitz if every diagonal from top-left to 
bottom-right has the same element.
Now given an M x N matrix, return True if and only if the matrix is Toeplitz.
Example 1:
Input:
matrix = [
  [1,2,3,4],
  [5,1,2,3],
  [9,5,1,2]
]
Output: True
Explanation:
In the above grid, the diagonals are:
"[9]", "[5, 5]", "[1, 1, 1]", "[2, 2, 2]", "[3, 3]", "[4]".
In each diagonal all elements are the same, so the answer is True.

class Solution {
    public boolean isToeplitzMatrix(int[][] matrix) {
        int r = matrix.length;
        int c = matrix[0].length;
        for (int i = 1; i < r; i ++) {
            for (int j = 1; j < c; j ++) {
                if (matrix[i][j] != matrix[i - 1][j - 1]){
                    return false;
                }
            }
        }
        
        return true;
    }
}


```

## 317-Shortest Distance from All Buildings

```java
Input: [[1,0,2,0,1],[0,0,0,0,0],[0,0,1,0,0]]

1 - 0 - 2 - 0 - 1
|   |   |   |   |
0 - 0 - 0 - 0 - 0
|   |   |   |   |
0 - 0 - 1 - 0 - 0

Output: 7 

Explanation: Given three buildings at (0,0), (0,4), (2,2), 
 and an obstacle at (0,2),
the point (1,2) is an ideal empty land to build a house, as the total 
travel distance of 3+3+1=7 is minimal. So return 7.
             
public class Solution {
public int shortestDistance(int[][] grid) {
    int row = grid.length;
    if (row == 0) {
        return -1;
    }
    int col = grid[0].length;
    int[][] record1 = new int[row][col]; // visited num
    int[][] record2 = new int[row][col]; // distance
    int num1 = 0;
    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            if (grid[r][c] == 1) {
                num1 ++;
                boolean[][] visited = new boolean[row][col];
                Queue<int[]> queue = new LinkedList<int[]>();
                queue.offer(new int[]{r, c});
                int dist = 0;
                while (!queue.isEmpty()) {
                    int size = queue.size();
                    for (int i = 0; i < size; i++) {
                        int[] node = queue.poll();
                        int x = node[0];
                        int y = node[1];
                        record2[x][y] += dist;
                        record1[x][y] ++;
                if (x > 0 && grid[x - 1][y] == 0 && !visited[x - 1][y]) {
                            queue.offer(new int[]{x - 1, y});
                            visited[x - 1][y] = true;
              }
                if (x + 1 < row && grid[x + 1][y] == 0 && !visited[x + 1][y]) {
                            queue.offer(new int[]{x + 1, y});
                            visited[x + 1][y] = true;
                        }
               if (y > 0 && grid[x][y - 1] == 0 && !visited[x][y - 1]) {
                            queue.offer(new int[]{x, y - 1});
                            visited[x][y - 1] = true;
                        }
           if (y + 1 < col && grid[x][y + 1] == 0 && !visited[x][y + 1]) {
                            queue.offer(new int[]{x, y + 1});
                            visited[x][y + 1] = true;
                        }
                    }
                    dist ++;
                }
            }
        }
    }
    int result = Integer.MAX_VALUE;
    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            if (grid[r][c] == 0 && record1[r][c] == num1
                 && record2[r][c] < result) {
                result = record2[r][c];
            }
        }
    }
    return result == Integer.MAX_VALUE ? -1 : result;
}
}
```

## 76-Minimum Window Substring

```java
Given a string S and a string T, find the
 minimum window in S which will contain all the characters in T 
 in complexity O(n).
Input: S = "ADOBECODEBANC", T = "ABC"
Output: "BANC"

class Solution {
    public String minWindow(String s, String t) {
        if (s == null || t == null || s.length() < t.length()) {
            return "";
        }
        HashMap<Character, Integer> window = new HashMap<>(); 
        HashMap<Character, Integer> need = new HashMap<>();
        
        for (char t1 : t.toCharArray()) {
            need.put(t1, need.getOrDefault(t1,0) + 1);
        }
        int left = 0; int right = 0; int valid = 0;
        int start = 0; int len = Integer.MAX_VALUE;
        // use one minRight to record
        while (right < s.length()) { 
        // 区间[left, right)是左闭右开的，所以初始情况下窗口没有包含任何元素：
            char s1 = s.charAt(right);
            right ++;
            if (need.containsKey(s1)) {
                window.put(s1,window.getOrDefault(s1,0) + 1);
                if (window.get(s1).equals(need.get(s1))) valid ++;
            }
 // 右指针移动当于在寻找一个「可行解」，然后移动左指针在优化这个「可行解」，最终找到最优解
            while (valid == need.size()) {
              if (right - left < len) {
                    start = left;
                    len = right - left;
                }
                char s2 = s.charAt(left);
                left ++;
                if (need.containsKey(s2)) { 
                // every time need containsKey not contains
                   if (window.get(s2).equals(need.get(s2))) valid --;
                   window.put(s2,window.get(s2) - 1);
                }
              
            }
        }
        if (len == Integer.MAX_VALUE) return ""; // don't miss
        return s.substring(start, start + len);  
    }
}
```

## 896- Monotonic Array

```java
An array is monotonic if it is either monotone increasing or monotone decreasing.
An array A is monotone increasing if for all i <= j, A[i] <= A[j].
 An array A is monotone decreasing if for all i <= j, A[i] >= A[j].
Return true if and only if the given array A is monotonic.
Input: [1,2,2,3]
Output: true

class Solution {
      public boolean isMonotonic(int[] A) {
        boolean inc = true, dec = true;
        for (int i = 1; i < A.length; ++i) {
            inc &= A[i - 1] <= A[i];
            dec &= A[i - 1] >= A[i];
        }
        return inc || dec;
    }
}
```

## 311-Sparse Matrix Multiplication

```java
Input:

A = [
  [ 1, 0, 0],
  [-1, 0, 3]
]
B = [
  [ 7, 0, 0 ],
  [ 0, 0, 0 ],
  [ 0, 0, 1 ]
]
Output:
     |  1 0 0 |   | 7 0 0 |   |  7 0 0 |
AB = | -1 0 3 | x | 0 0 0 | = | -7 0 3 |
                  | 0 0 1 |          
public class Solution {
    public int[][] multiply(int[][] A, int[][] B) {
        int m = A.length, n = A[0].length, nB = B[0].length;
        int[][] C = new int[m][nB];

        for(int i = 0; i < m; i++) {
            for(int k = 0; k < n; k++) {
                if (A[i][k] != 0){
                    for (int j = 0; j < nB; j++) {
                        if (B[k][j] != 0) C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }
        return C;   
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

## 56-Merge Intervals

```java
class Solution {
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, (a,b) -> a[0] - b[0]); // -> not =>
        int k = 0;
        int i = 0;
        while (i < intervals.length) {
            int start = intervals[i][0];
            int end = intervals[i][1];
            while (i < intervals.length - 1 
            && end >= intervals[i + 1][0]){ 
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

## 1060-Missing Element in Sorted Array

```java
Input: A = [4,7,9,10], K = 1
Output: 5
Explanation: 
The first missing number is 5.

class Solution {
   public int missingElement(int[] nums, int k) {
        if(nums == null || nums.length == 0 ) return 0;
        int l = nums.length;
        int left = 0, right = nums.length - 1;
        int missing = nums[right] - nums[left] - (right - left);
        if(missing < k) return nums[right] + k - missing;
        while(left + 1 < right){
            int mid = left + (right - left) / 2;
            int missingLeft = nums[mid] - nums[left] - (mid - left);
            if(missingLeft >= k) right = mid;
            else{
                k -= missingLeft;
                left = mid;
            } 
        }
        return nums[left] + k;
    }
}
```

## 140-Word Break II

```java
Input:
s = "catsanddog"
wordDict = ["cat", "cats", "and", "sand", "dog"]
Output:
[
  "cats and dog",
  "cat sand dog"
]

class Solution {
   public List<String> wordBreak(String s, List<String> wordDict) {
    HashSet<String>  wordset = new HashSet<>(wordDict);
    return DFS(s, wordset, new HashMap<String, LinkedList<String>>());
}       

// DFS function returns an array including all substrings derived from s.
List<String> DFS(String s, Set<String> wordDict, 
HashMap<String, LinkedList<String>>map) {
    if (map.containsKey(s)) 
        return map.get(s);
        
    LinkedList<String>res = new LinkedList<String>();     
    if (s.length() == 0) {
        res.add("");
        return res;
    }               
    for (String word : wordDict) {
        if (s.startsWith(word)) {
            List<String>sublist = DFS(s.substring(word.length()), wordDict, map);
            for (String sub : sublist) 
                res.add(word + (sub.isEmpty() ? "" : " ") + sub);               
        }
    }       
    map.put(s, res);
    return res;
}
}
```

## 23-Merge k Sorted Lists

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        
       // if(lists == null || lists.length == 0) return null;
        PriorityQueue<ListNode> pq = 
        new PriorityQueue<>((l1,l2)-> l1.val - l2.val);
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

## 670-Maximum Swap

```java
Input: 2736
Output: 7236
Explanation: Swap the number 2 and the number 7.
1)先转化为字符串
2)对字符串从大->小排序
3)排序前和排序后的字符串 从0->len 比较字符， 找到第一个不一样的字符A/B就是 待交换的字符
4)在排序前的字符串中，逆序查找该字符 交换A/B即可
5)字符串再转成数字

class Solution {
    public int maximumSwap(int num) {
            char[] s1 = Integer.toString(num).toCharArray();
            PriorityQueue<Character> pq = new PriorityQueue<>((a, b) -> b - a);
            for (char c : s1){
                pq.offer(c);
            }
            char swap = 'a';
            int index = -1;
            for (int i = 0; i < s1.length; i ++){
                char val = pq.poll();
                if(s1[i] != val){
                    swap = val;
                    index = i;
                    break; // don't forget this.
                }
            }
            if(swap!= 'a' && index != -1){
                for (int i = s1.length - 1; i >= 0; i --){
                    if(s1[i] == swap){
                        char temp = s1[index];
                        s1[index] = s1[i];
                        s1[i] = temp;

                    }
                }
                return Integer.valueOf(new String(s1));
            }

            return Integer.valueOf(new String(s1));
        }
}
```

## 314-Binary Tree Vertical Order Traversal

```java
Examples 1:

Input: [3,9,20,null,null,15,7]

   3
  /\
 /  \
 9  20
    /\
   /  \
  15   7 

Output:

[
  [9],
  [3,15],
  [20],
  [7]
]

class Solution {
  public List<List<Integer>> verticalOrder(TreeNode root) {
    List<List<Integer>> result = new ArrayList<>();
    if (root == null) return result;
    TreeMap<Integer, List<Integer>> map = new TreeMap<>();
    Queue<TreeNode> queue = new LinkedList<>();
    Queue<Integer> helper = new LinkedList<>();
    queue.offer(root);
    helper.offer(0);
    while (!queue.isEmpty()) {
        int size = queue.size();
        for (int i = 0; i < size; i++) {
            TreeNode cur = queue.poll();
            int pos = helper.poll();
            if (!map.containsKey(pos)) map.put(pos, new ArrayList<>());
            map.get(pos).add(cur.val);// add current value based on post
            if (cur.left != null) {
                queue.offer(cur.left);
                helper.offer(pos - 1);
            }
            if (cur.right != null) {
                queue.offer(cur.right);
                helper.offer(pos + 1);   
            }
        }
    }
    for (Map.Entry<Integer, List<Integer>> entry : map.entrySet()){
        result.add(entry.getValue());
    }
    return result;
}

}
```

## 1026-Maximum Difference Between Node and Ancestor

```java
 Given the root of a binary tree, find the maximum value V for which 
 there exists different nodes A and B where V = |A.val - B.val| and 
 A is an ancestor of B.
 // 最大差值一定是ancestors里面的最大值或最小值跟当前值的差值的绝对值。
class Solution {
    int res = 0;
    public int maxAncestorDiff(TreeNode root) {
        maxAncestorDiffHelper(root, root.val, root.val); 
        // 开始是ROOT.VAL不是0
        return res;
    }
    
    public void maxAncestorDiffHelper(TreeNode root, int min, int max) {
        if (root == null) return;
         min = Math.min(root.val, min);
         max = Math.max(root.val, max);
        res = getMax(res, Math.abs(root.val - min), Math.abs(root.val - max));
        maxAncestorDiffHelper(root.left, min, max);
        maxAncestorDiffHelper(root.right, min, max);
    }
    
    public int getMax(int va1, int va2, int va3) {
       int max1 = Math.max(va1, va2);
       return Math.max(max1, va3); 
    }
}
```

## 31-Next Permutation

```java
class Solution {
    public void nextPermutation(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int len = nums.length, p = len - 2, q = len - 1;
        
        // 1. 从后向前找到一个非增长的元素，等于的话也继续向前
        while (p >= 0 && nums[p] >= nums[p + 1]) {
            p--;
        }
        //全逆序的数组不会进入这个判断，全逆序p的位置为-1
        // 2. 从后向前找到第一个比p位置元素大的元素，注意这个数字肯定有，等于的话继续向前
        if (p >= 0) {
            while (nums[q] <= nums[p]) {
                q--;
            }
            swap(nums, p, q);
        }
        // 3. p位置后面的数组元素进行翻转
        reverse(nums, p + 1, len - 1);
    }
    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    private void reverse(int[] nums, int left, int right) {
        while (left < right) {
            swap(nums, left, right);
            left++;
            right--;
        }
    }
}
```

## 173- Binary Search Tree Iterator

```java

class BSTIterator {
    Stack<TreeNode> stack;
    TreeNode cur;
    
    public BSTIterator(TreeNode root) {
        stack = new Stack<>();
        cur = root;
    }
    
    public int next() {
        int val = 0;
        while (!stack.isEmpty() || cur != null) {
            while (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            TreeNode node = stack.pop();
            val = node.val;
            cur = node.right;
            break;
        }
        return val;
    }
    
    public boolean hasNext() {
        return !stack.isEmpty() || cur != null;
    }
}
```

## 34-Find First and Last Position of Element in Sorted Array

```java
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]

class Solution {
   public int[] searchRange(int[] nums, int target) {
    int[] result = new int[2];
    result[0] = findFirst(nums, target);
    result[1] = findLast(nums, target);
    return result;
}

private int findFirst(int[] nums, int target){
    int idx = -1;
    int start = 0;
    int end = nums.length - 1;
    while(start <= end){
        int mid = (start + end) / 2;
        if(nums[mid] >= target){
            end = mid - 1;
        }else{
            start = mid + 1;
        }
        if(nums[mid] == target) idx = mid;
    }
    return idx;
}

private int findLast(int[] nums, int target){
    int idx = -1;
    int start = 0;
    int end = nums.length - 1;
    while(start <= end){
        int mid = (start + end) / 2;
        if(nums[mid] <= target){
            start = mid + 1;
        }else{
            end = mid - 1;
        }
        if(nums[mid] == target) idx = mid;
    }
    return idx;
}
}
```

## 426-Convert Binary Search Tree to Sorted Doubly Linked List

```java

class Solution {
    public Node treeToDoublyList(Node root) {
        if (root == null) return null;
        Stack<Node> stack = new Stack<>();
        Node cur = root;
        Node first = null; Node last = null;
        while (!stack.isEmpty() || cur != null) {
            while (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            Node node = stack.pop();
            if (first == null){
                first = node;
            }
            if (last != null) {
                last.right = node;
                node.left = last;
            }
            last = node;
            cur = node.right;
            
        }
        first.left = last;
        last.right = first;
        return first;
    }
}
```

## 708 Insert into a Sorted Circular Linked List

```java
Given a node from a Circular Linked List which is sorted in ascending order, 
write a function to insert a value insertVal into the list such that it 
remains a sorted circular list. The given node can be a reference to 
any single node in the list, and may not be necessarily the smallest 
value in the circular list.

We could also use 2 pass traverse where in the 1st pass try to find max the
 tipping point node, then break the cycle into ordinary list, 
 then insert x, then reconnect newTail and head to be a cycle.
I think the following solution is correct. but the judging system go 
against with this one caseinsert(1->3->5->{loopback} , 1).
This solution returns 1->3->5->1->{loopback} vs what 's 
expected by the judging system 1->1->3->5->{loopback}, 
since it's already mentioned in the problems
If there are multiple suitable places for insertion, 
you may choose any place to insert the new value. After the insertion,
 the cyclic list should remain sorted. I think 1->3->5->1->{loopback} 
 could also be considered correct. Let me know if you have any insight into this.

class Solution {
    public Node insert(Node start, int x) {
        // if start is null, create a node pointing to itself and return
        if (start == null) {
            Node node = new Node(x, null);
            node.next = node;
            return node;
        }
        // if start is not null, try to insert it into correct position
        // 1st pass to find max node
        Node cur = start;
        while (cur.val <= cur.next.val && cur.next != start) 
            cur = cur.next;
        // 2nd pass to insert the node in to correct position
        Node max = cur;
        Node dummy = new Node(0, max.next); 
        // use a dummy head to make insertion process simpler
        max.next = null; // break the cycle
        cur = dummy;
        while (cur.next != null && cur.next.val < x) {
            cur = cur.next;
        }
        cur.next = new Node(x, cur.next); // insert
        Node newMax = max.next == null ? max : max.next; // reconnect to cycle
        newMax.next = dummy.next;
        return start;
    }
}
```

## 

## 

## 

## 




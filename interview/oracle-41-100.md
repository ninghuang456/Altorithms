# Oracle 41-100

## 208- Implement Trie \(Prefix Tree\)

```java
class Trie {
   boolean isEnd;
   Trie[] next; 
   
    public Trie() {
        isEnd = false;
        next = new Trie[26];
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
        Trie cur = this;
        for (int i = 0; i < word.length(); i ++) {
            char c = word.charAt(i);
            if (cur.next[c - 'a'] == null){
                cur.next[c - 'a'] = new Trie();
            }
            cur = cur.next[c-'a'];
        }
        
        cur.isEnd = true;
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        Trie node = startWithPrefix(word);
        if (node == null) return false;
        return node.isEnd;
        
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        return startWithPrefix(prefix) != null;
    }
    
    public Trie startWithPrefix(String prefix){
        Trie cur = this;
        for (int i = 0; i < prefix.length(); i ++) {
            char c = prefix.charAt(i);
            if (cur.next[c - 'a'] == null) return null;
            cur = cur.next[c - 'a'];
        }
        return cur;
    }
}

```

## 65-Valid Number

```java
Some examples:
"0" => true
" 0.1 " => true
"abc" => false
"1 a" => false
"2e10" => true
" -90e3   " => true
" 1e" => false
"e3" => false
" 6e-1" => true
" 99e2.5 " => false
"53.5e93" => true
" --6 " => false
"-+3" => false
"95a54e53" => false

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

## 402 - Remove K Digits

```java
Given a non-negative integer num represented as a string, 
remove k digits from the number so that the new number is the smallest possible.
Example 1:
Input: num = "1432219", k = 3
Output: "1219"
Explanation: Remove the three digits 4, 3, and 2 to form the
 new number 1219 which is the smallest.
 
class Solution {
    public String removeKdigits(String num, int k) {
        //特殊情况全部删除
        if(num.length() == k){
            return "0";
        }
        char[] s = num.toCharArray();
        Stack<Character> stack = new Stack<>();
        //遍历数组
        for(Character i : s){
          //移除元素的情况，k--
            while(!stack.isEmpty() && i < stack.peek() && k > 0){
                   stack.pop();
                   k--;
            }
            //栈为空，且当前位为0时，我们不需要将其入栈
            if(stack.isEmpty() && i == '0'){
                continue;
            }
            stack.push(i);
        }
        while(k > 0){
            stack.pop();
            k--;
        }
        //这个是最后栈为空时，删除一位，比如我们的10，删除一位为0，
        //按上面逻辑我们会返回""，所以我们让其返回"0"
         if(stack.isEmpty()){
             return "0";
         }
         //反转并返回字符串
         StringBuilder str = new StringBuilder();
         while(!stack.isEmpty()){
             str.append(stack.pop());
         }
         return str.reverse().toString();
    }
}
//链接：https://leetcode-cn.com/problems/remove-k-digits/solution/dong-tu-shuo-suan-fa-zhi-yi-diao-kwei-shu-zi-by-yu/

```

## 75-Sort Colors

```java
Input: nums = [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]

public class Solution {

    public void sortColors(int[] nums) {
        int len = nums.length;
        if (len < 2) {
            return;
        }

        // all in [0, zero) = 0
        // all in [zero, i) = 1
        // all in [two, len - 1] = 2
        
        // 循环终止条件是 i == two，那么循环可以继续的条件是 i < two
        // 为了保证初始化的时候 [0, zero) 为空，设置 zero = 0，
        // 所以下面遍历到 0 的时候，先交换，再加
        int zero = 0;

        // 为了保证初始化的时候 [two, len - 1] 为空，设置 two = len
        // 所以下面遍历到 2 的时候，先减，再交换
        int two = len;// 
        int i = 0;
        // 当 i == two 上面的三个子区间正好覆盖了全部数组
        // 因此，循环可以继续的条件是 i < two
        while (i < two) {
            if (nums[i] == 0) {
                swap(nums, i, zero);
                zero++;
                i++;
            } else if (nums[i] == 1) {
                i++;
            } else {
                two--;// 要明白为什么这个时候i不动
                swap(nums, i, two);
            }
        }
    }

    private void swap(int[] nums, int index1, int index2) {
        int temp = nums[index1];
        nums[index1] = nums[index2];
        nums[index2] = temp;
    }
}
```

## 79- Word Search

```java
Given an m x n board and a word, find if the word exists in the grid.

```

## 




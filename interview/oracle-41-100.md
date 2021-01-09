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
//链接：https://leetcode-cn.com/problems/remove-k-digits/
//solution/dong-tu-shuo-suan-fa-zhi-yi-diao-kwei-shu-zi-by-yu/

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
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], 
word = "ABCCED"
Output: true

class Solution {
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0 || board[0].length == 0) 
        return false;
        int r = board.length;
        int c = board[0].length;
        boolean[][] visited = new boolean[r][c];
        for (int i = 0; i < r; i ++) {
            for (int j = 0; j < c; j ++) {
                if (board[i][j] == word.charAt(0) && 
                findWord(board, i, j, word, 0, visited)){
                    return true;
                }
                
            }
        }
        return false;
    }
    
    public boolean findWord(char[][] board, int r, int c, String word, 
                          int index, boolean[][] visited) {
        if (index == word.length()) return true;
        if(!inArea(r,c,board) || visited[r][c] || 
        board[r][c] != word.charAt(index)) return false;
        
        visited[r][c] = true;
        boolean res = findWord(board, r + 1, c, word, index + 1, visited) ||
                      findWord(board, r - 1, c, word, index + 1, visited) ||
                      findWord(board, r, c + 1, word, index + 1, visited) ||
                       findWord(board, r, c - 1, word, index + 1, visited);
        
        visited[r][c] = false;
        return res;
        
    }
    
    public boolean inArea(int r, int c, char[][] board) {
        return r >= 0 && r < board.length && c >= 0 && c < board[0].length;
    }
}
```

## 295 Find Median from Data Stream

```java
class MedianFinder {
   private Queue<Integer> small = new PriorityQueue<>((o1,o2) -> (o2 - o1)); 
                                              // need <>
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
/**
```

## 15-3Sum

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null || nums.length < 3) return res;
        Arrays.sort(nums);
        Set<List<Integer>> set = new HashSet<>();
        for (int left = 0; left < nums.length - 2; left ++){
            int mid = left + 1;
            int right = nums.length - 1;
            while (mid < right){
                int sum = nums[left] + nums[mid] + nums[right];
                if (sum == 0){
                    ArrayList<Integer> ls = new ArrayList<>();
                    ls.add(nums[left]);
                    ls.add(nums[mid]);
                    ls.add(nums[right]);
                    if(!set.contains(ls)){
                        res.add(ls);
                        set.add(ls);
                    }
                    mid ++;
                    right --; //容易忘记
                } else if (sum > 0){
                    right --;
                } else {
                    mid ++;
                }
                
            }
        }
        return res;
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
            while (i < intervals.length - 1 && end >= 
            intervals[i + 1][0]){ 
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

## 167-Two Sum II - Input array is sorted

```java
class Solution {
    public int[] twoSum(int[] numbers, int target) {
        int[] result = new int[2];
        
        int left = 0, right = numbers.length - 1;
        while (left < right) {
            if (numbers[left] + numbers[right] < target) {
                left++;
            } else if (numbers[left] + numbers[right] > target) {
                right--;
            } else {
                break;//遇到合适的pair直接break，省点时间，有唯一解的情况
            }
        }
        
        //因为两个indices不是zero-based，所以+1，从1开始数
        result[0] = left + 1;
        result[1] = right + 1;
        
        return result;
    }
}
```

## 341-Flatten Nested List Iterator

```java
Given a nested list of integers, implement an iterator to flatten it.
Each element is either an integer, or a list -- 
whose elements may also be integers or other lists.
Example 1:
Input: [[1,1],2,[1,1]]
Output: [1,1,2,1,1]
Explanation: By calling next repeatedly until hasNext returns false, 
             the order of elements returned by next should be: [1,1,2,1,1].
             
 public class NestedIterator implements Iterator<Integer> {
    private List<Integer> listIterator;
    private int index;

    public NestedIterator(List<NestedInteger> nestedList) {
        listIterator = getListIterator(nestedList);
        index = 0;
    }

    @Override
    public Integer next() {
        if (hasNext()){
            Integer val = listIterator.get(index ++);
            return val;
        }
        return null;
    }

    @Override
    public boolean hasNext() {
        return index  < listIterator.size();
        
        
    }
    
    public static List<Integer> getListIterator(List<NestedInteger> nestedList) {
        ArrayList<Integer> res = new ArrayList<>();
        for (NestedInteger val : nestedList) {
            if (val.isInteger()) {
                res.add(val.getInteger());
            } else {
                res.addAll(getListIterator(val.getList()));
            }
        }
        return res;
    }
    
}            
```

## 154- Find Minimum in Rotated Sorted Array II

```java
Suppose an array sorted in ascending order is rotated at some pivot 
unknown to you beforehand.
(i.e.,  [0,1,2,4,5,6,7] might become  [4,5,6,7,0,1,2]).
Find the minimum element.
The array may contain duplicates.
Example 1:
Input: [1,3,5]
Output: 1
class Solution {
    public int findMin(int[] nums) {
         int l = 0, r = nums.length-1;
	 while (l < r) {
		 int mid = (l + r) / 2;
		 if (nums[mid] < nums[r]) {
			 r = mid;
		 } else if (nums[mid] > nums[r]){
			 l = mid + 1;
		 } else {  
			 r--;  
		 }
	 }
	 return nums[l];
        
    }
}
```

## 450： Delete Node in a BST

```java
class Solution {
    public TreeNode deleteNode(TreeNode root, int key) {
        if(root == null){
            return null;
        }
        //当前节点值比key小，则需要删除当前节点的左子树中key对应的值，并保证二叉搜索树的性质不变
        if(key < root.val){
            root.left = deleteNode(root.left,key);
        }
        //当前节点值比key大，则需要删除当前节点的右子树中key对应的值，并保证二叉搜索树的性质不变
        else if(key > root.val){
            root.right = deleteNode(root.right,key);
        }
        //当前节点等于key，则需要删除当前节点，并保证二叉搜索树的性质不变
        else{
            //当前节点没有左子树
            if(root.left == null){
                return root.right;
            }
            //当前节点没有右子树
            else if(root.right == null){
                return root.left;
            }
            //当前节点既有左子树又有右子树
            else{
                TreeNode node = root.right;
                //找到当前节点右子树最左边的叶子结点
                while(node.left != null){
                    node = node.left;
                }
                //将root的左子树放到root的右子树的最下面的左叶子节点的左子树上
                node.left = root.left;
                return root.right;
            }
        }
        return root;
    }
}

```

## 384- Shuffle an Array

```java
Given an integer array nums, design an algorithm to randomly shuffle the array.
Implement the Solution class:
Solution(int[] nums) Initializes the object with the integer array nums.
int[] reset() Resets the array to its original configuration and returns it.
int[] shuffle() Returns a random shuffling of the array.
Example 1:
Input
["Solution", "shuffle", "reset", "shuffle"]
[[[1, 2, 3]], [], [], []]
Output
[null, [3, 1, 2], [1, 2, 3], [1, 3, 2]]
Explanation
Solution solution = new Solution([1, 2, 3]);
solution.shuffle(); // Shuffle the array [1,2,3] and return its result. 
//Any permutation of [1,2,3] must be equally likely to be returned. 
//Example: return [3, 1, 2]
solution.reset(); // Resets the array back to its original configuration [1,2,3].
// Return [1, 2, 3]
solution.shuffle();// Returns the random shuffling of array [1,2,3]. 
//Example: return [1, 3, 2]

public class Solution {
    private int[] nums;
    private Random random;

    public Solution(int[] nums) {
        this.nums = nums;
        random = new Random();
    }
    
    /** Resets the array to its original configuration and return it. */
    public int[] reset() {
        return nums;
    }
    
    /** Returns a random shuffling of the array. */
    public int[] shuffle() {
        if(nums == null) return null;
        int[] a = nums.clone();
        for(int j = 1; j < a.length; j++) {
            int i = random.nextInt(j + 1);
            swap(a, i, j);
        }
        return a;
    }
    
    private void swap(int[] a, int i, int j) {
        int t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
}

```

## 20-Valid Parentheses

```java
Input: s = "()[]{}"
Output: true
class Solution {
    public boolean isValid(String s) {
        if (s == null || s.length() == 0) return true;
        Stack<Character> stack = new Stack<>();
        for (char c : s.toCharArray()){
            if (c == '(' || c == '{' || c == '['){
                stack.push(c);
            }
            if (stack.isEmpty()) return false;
            if ((c == ')' && stack.pop() != '(') ||
                (c == '}' && stack.pop() != '{') ||
                (c == ']' && stack.pop() != '[') ) {
                return false;
            }
        }
        if (!stack.isEmpty()) return false;
        return true;
        
    }
}
```

## 109- Convert Sorted List to Binary Search Tree

```java
Given the head of a singly linked list where elements are sorted in ascending order,
convert it to a height balanced BST.

public class Solution {
public TreeNode sortedListToBST(ListNode head) {
    if(head==null) return null;
    return toBST(head,null);
}
public TreeNode toBST(ListNode head, ListNode tail){
    ListNode slow = head;
    ListNode fast = head;
    if(head==tail) return null;
    
    while(fast!=tail&&fast.next!=tail){
        fast = fast.next.next;
        slow = slow.next;
    }
    TreeNode thead = new TreeNode(slow.val);
    thead.left = toBST(head,slow);
    thead.right = toBST(slow.next,tail);
    return thead;
}
}
```

## 540-Single Element in a Sorted Array

```java
You are given a sorted array consisting of only integers where every element 
appears exactly twice, except for one element which appears exactly once.
Find this single element that appears only once.

  public static int singleNonDuplicate(int[] nums) {
        int start = 0, end = nums.length - 1;

        while (start < end) {
            // We want the first element of the middle pair,
            // which should be at an even index if the left part is sorted.
            // Example:
            // Index: 0 1 2 3 4 5 6
            // Array: 1 1 3 3 4 8 8
            //            ^
            int mid = (start + end) / 2;
            if (mid % 2 == 1) mid--;

            // We didn't find a pair. The single element must be on the left.
            // (pipes mean start & end)
            // Example: |0 1 1 3 3 6 6|
            //               ^ ^
            // Next:    |0 1 1|3 3 6 6
            if (nums[mid] != nums[mid + 1]) end = mid;

            // We found a pair. The single element must be on the right.
            // Example: |1 1 3 3 5 6 6|
            //               ^ ^
            // Next:     1 1 3 3|5 6 6|
            else start = mid + 2;
        }

        // 'start' should always be at the beginning of a pair.
        // When 'start > end', start must be the single element.
        return nums[start];
    }
    
```

## 297-Serialize and Deserialize Binary Tree

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

## 83 - Remove Duplicates from Sorted List

```java
Given the head of a sorted linked list, delete all duplicates such that 
each element 
appears only once. Return the linked list sorted as well.

class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode first = head;
        ListNode second = head.next;
        while (second != null) {
            if (second.val == first.val){
                first.next = second.next;
            } else {
                first = second;
            } 
            second = second.next;
        }
        return head;  
    }
}
```

## 105 - Construct Binary Tree from Preorder and Inorder Traversal

```java
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return buildTreeHelper(preorder, 0, preorder.length - 1, inorder, 0, 
                             inorder.length - 1);
    }   
        
     public TreeNode buildTreeHelper(int[] preorder, int pleft, int pright, 
                                  int[] inorder, int ileft, int iright) {
         if (pleft > pright || ileft > iright ) {
             return null;
         }
         int rootIndex = 0;
         for (int i = ileft; i <= iright; i ++) {
             if (preorder[pleft] == inorder[i]){
                 rootIndex = i;
                 break;
             }
         }
         TreeNode root = new TreeNode(preorder[pleft]);
     root.left = buildTreeHelper(preorder, pleft + 1, 
                    pleft + rootIndex - ileft, inorder, ileft, rootIndex - 1); 
     root.right =  buildTreeHelper(preorder, pleft + rootIndex - ileft + 1, 
                   pright,inorder, rootIndex + 1,iright ); 
         return root;    
     }
    
    }
```

## 206 - Reverse Linked List

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        if(head == null || head.next == null) return head;
        ListNode pre = null;
        ListNode cur = head;
        while (cur != null) {
            ListNode temp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = temp;
        }
        return pre;
    }
}

class Solution {
    public ListNode reverseList(ListNode head) {
        if(head == null || head.next == null) return head;
        ListNode res = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return res;
            
        
    }
}
```

## 33 - Search in Rotated Sorted Array

```java
class Solution { 
        public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0){return -1;}
        int start = 0;
        int end = nums.length - 1;
        int mid;
        while (start <= end) {
             mid = start + (end - start) /2;
            if (nums[mid] == target) {return mid;}
            if (nums[start] <= nums[mid]){
                if(target >= nums[start] && target < nums[mid]){
                    end = mid - 1;
                } else {
                    start = mid + 1;
                }
            } else {
                if(target <= nums[end] && nums[mid] < target ){
                    start = mid + 1;
                } else {
                    end = mid - 1;
                }
            }
        }
        return -1;
    }
}
```

## 21- Merge Two Sorted Lists

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
    
    class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null || l2 == null) {
            return l1 == null ? l2 : l1;
        }
        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                cur.next = l1;
                cur = l1;
                l1 = l1.next;
                
            } else {
                cur.next = l2;
                cur = l2;
                l2 = l2.next;
            }
        }
        cur.next = (l1 == null) ? l2 : l1;
        
        return dummy.next;
        
    }
}
```

## 437 - Path Sum III

```java
root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8
      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1
Return 3. The paths that sum to 8 are:
1.  5 -> 3
2.  5 -> 2 -> 1
3. -3 -> 11

class Solution {
     public int pathSum(TreeNode root, int sum) {
        HashMap<Integer, Integer> preSum = new HashMap();
        preSum.put(0,1);
        return helper(root, 0, sum, preSum);
    }
    
    public int helper(TreeNode root, int currSum, int target, 
                 HashMap<Integer, Integer> preSum) {
        if (root == null) {
            return 0;
        }
        
        currSum += root.val;
        int res = preSum.getOrDefault(currSum - target, 0);
        preSum.put(currSum, preSum.getOrDefault(currSum, 0) + 1);
        
        res += helper(root.left, currSum, target, preSum) + 
               helper(root.right, currSum, target, preSum);
        preSum.put(currSum, preSum.get(currSum) - 1);
        return res;
    }
}
```

## 350- Intersection of Two Arrays II

```java
Given two arrays, write a function to compute their intersection.
Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
Output: [4,9]

public class Solution {
    public int[] intersect(int[] nums1, int[] nums2) {
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        ArrayList<Integer> result = new ArrayList<Integer>();
        for(int i = 0; i < nums1.length; i++)
        {
            if(map.containsKey(nums1[i])) map.put(nums1[i], map.get(nums1[i])+1);
            else map.put(nums1[i], 1);
        }
    
        for(int i = 0; i < nums2.length; i++)
        {
            if(map.containsKey(nums2[i]) && map.get(nums2[i]) > 0)
            {
                result.add(nums2[i]);
                map.put(nums2[i], map.get(nums2[i])-1);
            }
        }
    
       int[] r = new int[result.size()];
       for(int i = 0; i < result.size(); i++)
       {
           r[i] = result.get(i);
       }
    
       return r;
    }
}
```

## 212- Word Search II

```java
Given an m x n board of characters and a list of strings words, 
return all words on the board.
Input: board = [["o","a","a","n"],["e","t","a","e"],
["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]

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
    private void find(char [] [] board, boolean [][]visited,int i,
    int j,int m,int n,Set<String> result,trieNode cur){
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

//慢的DFS
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
                if (board[i][j] == word.charAt(0) && 
                findWord(board, word, i, j, 0, visited)) { // start from 0;
                    return true;
                }
            }
        }
        return false;
    }
    
    public boolean findWord(char[][] board, String word, int i, int j, 
    int size,  boolean[][] visited){
        if(size == word.length()){
        // it means last run's index reach last letter; similer like node == null;
            return true;
        }
        
        if (!inArea(board, i , j) || visited[i][j] || 
        board[i][j] != word.charAt(size)){ 
        // need check in area before visited[i][j]
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
```

## 994 - Rotting Oranges

```java
In a given grid, each cell can have one of three values:
the value 0 representing an empty cell;
the value 1 representing a fresh orange;
the value 2 representing a rotten orange.
Every minute, any fresh orange that is adjacent (4-directionally) to a rotten 
orange becomes rotten.
Return the minimum number of minutes that must elapse until no cell has a 
fresh orange.  If this is impossible, return -1 instead.

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

## 211- Design Add and Search Words Data Structure

```java
class WordDictionary {
    boolean isEnd;
    WordDictionary[] next;
    
    public WordDictionary() {
        isEnd = false;
        next = new WordDictionary[26];
        
    }
    
    /** Adds a word into the data structure. */
    public void addWord(String word) {
        WordDictionary cur = this;
        for (char c : word.toCharArray()){
            if (cur.next[c - 'a'] == null){
                WordDictionary wd = new WordDictionary();
                cur.next[c - 'a'] = wd;
            }
            cur = cur.next[c - 'a'];
        }
        cur.isEnd = true;
    }
    
    /** Returns if the word is in the data structure. 
    A word could contain the dot character '.' to represent any one letter. */
    public boolean search(String word) {
        return searchHelper(word, 0, this);
    
    }
    
    public boolean searchHelper(String word, int level, WordDictionary node){
        if (level == word.length()) return node.isEnd;
        char c = word.charAt(level);
        if (c != '.') {
            return (node.next[c - 'a'] != null) && 
            searchHelper(word, level + 1, node.next[c - 'a']);
        } 
            for (int i = 0; i < node.next.length; i ++){
                if((node.next[i] != null) &&
                 searchHelper(word, level + 1, node.next[i])) { 
                 // 如果不满足没关系 因为有可能后面有满足的 所以不要返回false
                    return true;
                }
            }
        
        return false;
    }
   
}

```

## 91- Decode Ways

```java
Input: s = "226"
Output: 3
Explanation: "226" could be decoded as "BZ" (2 26), 
"VF" (22 6), or "BBF" (2 2 6).

class Solution {
    public int numDecodings(String s) {
        int len = s.length();
        int[] dp = new int[len + 1];
        dp[0] = 1;
        dp[1] = s.charAt(0) == '0' ? 0: 1;
        for (int i = 2; i <= len; i ++) {
            int step1 = Integer.valueOf(s.substring(i-1, i));
            int step2 = Integer.valueOf(s.substring(i-2,i));
            if (step1 >= 1){
                dp[i] += dp[i - 1];
            }
            if(step2 >= 10 && step2 <= 26){
                dp[i] += dp[i - 2];
            }
            
        }
        return dp[len];
    }
}
```

## 139- Word Break

```java
Input: s = "leetcode", wordDict = ["leet", "code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".

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

## 695- Max Area of Island

```java
class Solution {
    public int maxAreaOfIsland(int[][] grid) {
        int row = grid.length;
        int col = grid[0].length;
        int max = 0;
        for (int i = 0; i < row; i ++){
            for (int j = 0; j < col; j ++){
                if (grid[i][j] == 1){
                    int area = getArea(grid, i , j);
                    max = Math.max(area, max);
                }
            }
        }
        return max;
    }
    
    public int getArea(int[][] grid, int i, int j){
        if (!inArea(grid, i, j)){
            return 0;
        }
        if (grid[i][j] != 1){
            return 0;
        }
        grid[i][j] = 2;
        int sum = 1 + getArea(grid, i - 1, j) + getArea(grid, i, j - 1) 
        + getArea(grid, i + 1, j) + getArea(grid, i, j + 1);   
        return sum; 
    }
    
    public boolean inArea(int[][] grid, int i, int j){
        return i >= 0 && i < grid.length && j >= 0 && j < grid[0].length;
    }
}
```

## 739-Daily Temperatures

```java
Given a list of daily temperatures T, return a list such that, for each day in the
 input, tells you how many days you would have to wait until a warmer temperature.
 If there is no future day for which this is possible, put 0 instead.
For example, given the list of temperatures T = [73, 74, 75, 71, 69, 72, 76, 73], 
your output should be [1, 1, 4, 2, 1, 1, 0, 0].
Note: The length of temperatures will be in the range [1, 30000]. 
Each temperature will be an integer in the range [30, 100].

//单调栈
class Solution {
    public int[] dailyTemperatures(int[] T) {
    int[] res = new int[T.length];
    Stack<Integer> stack = new Stack<>();
    for(int i = T.length - 1; i >= 0; i--){
        while(!stack.isEmpty() && T[i] >= T[stack.peek()]){
            stack.pop();
        }
        res[i] = stack.isEmpty() ? 0 : (stack.peek() - i); 
        stack.push(i);
    }
    return res;
 }
}
```

## 92-Reverse Linked List II

```java
Reverse a linked list from position m to n. Do it in one-pass.
Note: 1 ≤ m ≤ n ≤ length of list.

class Solution {
    public ListNode reverseBetween(ListNode head, int m, int n) {
        if (head == null) {
            return null;
        }
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode pre = dummy;
        
        for (int i = 1; i < m; i++) {
            pre = pre.next;
        }
        ListNode first = pre.next;
        ListNode second = pre.next.next;
        for (int i = 0; i < n - m; i++) {
            first.next = second.next;
            second.next = pre.next;
            pre.next = second;
            second = first.next;
        }
        return dummy.next;
    }
}

```

## 242- Valid Anagram

```java
Given two strings s and t , write a function to determine if t is an anagram of s.
Example 1:
Input: s = "anagram", t = "nagaram"
Output: true

class Solution {
    public boolean isAnagram(String s, String t) {
        int[] table = new int[26];
        for(char c : s.toCharArray()){
            table[c - 'a'] ++;
        }
        for(char c : t.toCharArray()){
            table[c - 'a']--;
        }
        for(int i = 0; i < 26; i ++){
            if (table[i] != 0) return false;
        }
        return true;
    }
}
```

## 

## 

## 

## 

## 

## 

## 




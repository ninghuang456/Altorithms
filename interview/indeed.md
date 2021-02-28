# Indeed

## 171 - Excel Sheet Column Number

```java
public class ExcelAndNumber {
  /*  A -> 1
    B -> 2
    C -> 3         ...
    Z -> 26
    AA -> 27
    AB -> 28*/
    public int titleToNumber(String s) {
            int r = 0;
            for (char c : s.toCharArray())
                r = r * 26 + (int) (c - 'A' + 1);
            return r;
        }
/*  1 -> A
    2 -> B
    3 -> C
    ...
            26 -> Z
    27 -> AA
    28 -> AB*/
    public String convertToTitle(int n) {
        String ans = "";
        while (n > 0) {
            n --;
            ans = (char)(n % 26 + 'A') + ans;
            n /= 26;
        }
        return ans;
    }

}
```

## 151 Reverse Words in a String

```java
like reverse word：
public String reverseWordsInString(String s){
String[] parts = s.trim().split("\\s+");
String out = "";
for (int i = parts.length - 1; i > 0; i--) {
    out += parts[i] + " ";
}
return out + parts[0];
}
// reverse string but HTML
public class ReverseStringButHtml {
	 public String reverseHtml2(String html) {
		 if(html == null || html.length() < 2){
			 return html;
		 }
		 int len = html.length();
		 char[] htmlArr = html.toCharArray();
		 reverseChar(htmlArr,0,len -1);
		 int left = 0;
		 while(left < len){
			 if(htmlArr[left] != ';'){
				 left++;
			 }else {
				 int right = left+1;
				 if(right >= len -1){
					 break;
				 }
				 while(htmlArr[right] != '&'){
					 if(htmlArr[right] == ';'){
						 left = right;
					 }
					 right++;
				 }
				 reverseChar(htmlArr,left,right);
				 left = right+1;
			 }
		 }
		 return new String(htmlArr);
	 }
	 private  void reverseChar(char[] chars,int start,int end){
		 while(start < end) {
			 char temp = chars[start];
			 chars[start++] = chars[end];
			 chars[end--] = temp;
		 }
	 }
	 public static void main(String[] args) {
	   	ReverseStringButHtml sol = new ReverseStringButHtml();
	        
	          String s = "1234&eur;o;5677&&eu;567&";
	       // String s = "&euro4321";
	        System.out.println(sol.reverseHtml2(s));
	        System.out.println(sol.reverseHtml(s));
	    }
	 
	 public  String  reverseHtml(String html) {
		 if(html == null || html.length()==0) {
			 return null;
		 }
		 int  len = html.length();
		 char[] chArr = html.toCharArray();
		 reverseChar(chArr,0,len-1);
		 int start = 0;
		 int  end = 0;
		 while(end<len){
			 if(chArr[end] == ';'){
				 start = end;
			 }else if(chArr[end] == '&'){
				 if(chArr[start] == ';')
					 reverseChar(chArr,start,end);
				  start = end+1;
			 } 
			 end++;
			 
		 }
		return new String(chArr); 
	 }
}
```

## 228 Summary Ranges

```java
Input: nums = [0,1,2,4,5,7]
Output: ["0->2","4->5","7"]
Explanation: The ranges are:
[0,2] --> "0->2"
[4,5] --> "4->5"
[7,7] --> "7"

class Solution {
    public List<String> summaryRanges(int[] nums) {
        List<String> res = new ArrayList<>();
        if(nums.length == 0 || nums == null) return res;
        int start = nums[0];
        int end = nums[0];
        for (int i = 1; i < nums.length; i ++){
            if(nums[i] - end == 1 || nums[i] == end){
                end = nums[i];
            } else {
res.add((start==end) ? String.valueOf(start) : 
String.valueOf(start) + "->" + String.valueOf(end));
               start = nums[i];
               end = nums[i];
            }
          
        }
 res.add( (start==end) ? 
 String.valueOf(start) : String.valueOf(start) + "->" + String.valueOf(end));
        return res;
    }
}

public class SumaryRange {

	 public List<String> summaryRanges(int[] nums) {
	       List<String> result = new ArrayList<>();
	       if(nums ==null || nums.length == 0) {
	           return result;
	       }
	       int left =0;
	       int right = 0 ;
	       while(right< nums.length){
	          if( right < nums.length-1 && (nums[right+1] -nums[right] <= 1)){
	              right++;
	          } else {
	              StringBuilder sb = new StringBuilder();
	              if(left == right){
	                  sb.append(nums[left]);
	              } else {
	                 sb.append(nums[left]).append("->").append(nums[right]);
	              } 
	              result.add(sb.toString());
	              right++;
	              left = right;
	          }
	       }
	       return result;
	    }
	 public List<String> summaryRanges2(int[] nums) {
	       List<String> result = new ArrayList<>();
	       if(nums ==null || nums.length == 0) {
	           return result;
	       }
	       int left =0;
	       int right = 0 ;
	       int gap = 0;
	       while(right< nums.length){
	           if(right< nums.length -1 && gap == 0) {
	        		  gap = nums[right+1] -nums[right];
	        		  right++;
	        	  } 
	else if(right< nums.length -1 && gap == nums[right+1] -nums[right]){
		              right++;
		          }
	        	  else {
		              StringBuilder sb = new StringBuilder();
		              if(left == right){
		                  sb.append(nums[left]);
		              } else {
		sb.append(nums[left]).append("-").append(nums[right]).append("/").append(gap);
		              } 
		              result.add(sb.toString());
		              right++;
		              gap =0;
		              left = right;
		          }
	       }
	       return result;
	    }
	public static void main(String[] args) {
		int[] nums = {1,3,3,3,5,7,8,13,20,21,29};
		SumaryRange sr = new SumaryRange();
		System.out.println(sr.summaryRanges(nums));
		System.out.println(sr.summaryRanges3(nums));
		System.out.println(sr.summaryRanges2(nums));
		System.out.println(sr.summaryWithGap(nums));
	}
	
	public List<String> summaryRanges3(int[] nums){
		List<String> result = new ArrayList<>();
		if(nums == null || nums.length == 0) {
			return result;
		}
		int start = 0;
		int end = 0;
		int count = 0;
		while(end < nums.length){
			if(end < nums.length -1 && nums[end+1] == nums[end]+1){
				end++;
			} else {
				StringBuilder sb = new StringBuilder();
				if(start < end){
					sb.append(nums[start]).append("-").append(nums[end]);
				}
				else {
					sb.append(nums[start]);
				}
				start = end+1;
				result.add(sb.toString());
				end++;
			}
		}
		return result;
	}
	
	public List<String> summaryWithGap(int[] nums) {
		List<String> result = new ArrayList<>();
		if(nums == null || nums.length ==0) {
			return result;
		}
		int start = 0;
		int end = 0;
		int gap = 0;
		while(end < nums.length){
			if(end < nums.length -1 && gap == 0){
				gap = nums[end+1] - nums[end];
				end++;
			} else if(end < nums.length -1 && gap == nums[end+1] - nums[end]){ 
				end++;
			} else {
				StringBuilder sb = new StringBuilder();
				if(start == end){
	                  sb.append(nums[start]);
	              } else {
sb.append(nums[start]).append("-").append(nums[end]).append("/").append(gap);
	              }
				result.add(sb.toString());
				gap = 0;
				end++;
				start = end;
			}
		}
		return result;
	}
}

```

## indeed1: Auto complted

```java
package Pratice;

import java.util.ArrayList;
import java.util.List;

 class Trie {
    public boolean isEnd;
    public Trie[] next;
    public Trie() {
        isEnd = false;
        next = new Trie[26];

    }

    public void insert(String word) {
        if (word == null || word.length() == 0) return;
        Trie curr = this;
        char[] words = word.toCharArray();
        for (int i = 0; i < words.length; i ++) {
            int n = words[i] - 'a';
            if(curr.next[n] == null) {
                curr.next[n] = new Trie(); // not new Trie[26];
            }
            curr = curr.next[n];
        }
        curr.isEnd = true;
    }

    public boolean search(String word) {
        Trie node = searchPrefix(word);
        return node!=null && node.isEnd;
    }

    public boolean startsWith(String prefix) {
        Trie node = searchPrefix(prefix);
        return node != null;
    }

    public Trie searchPrefix(String word) {
        Trie node = this;
        char[] words = word.toCharArray();
        for (int i = 0; i < words.length; i ++) {
            node = node.next[words[i] - 'a'];
            if(node == null) return null;
        }
        return node;
    }
}

public class Auto_CompleteOne {
    Trie root;
    public Auto_CompleteOne(List<String> words){
        this.root = new Trie();
        for (String word: words) {
            root.insert(word);
        }
    }
    public List<String> find(String prefix){
        List<String> res = new ArrayList<>();
        Trie cur = root;
        Trie pRoot = cur.searchPrefix(prefix);
        helper(res, pRoot,prefix);
        return res;
    }
    public void helper(List<String> res, Trie pRoot, String cur){
        if (pRoot == null){
            return;
        }
        if (pRoot.isEnd){
            res.add(cur);
        }
        for (int i = 0; i < 26; i++){
            char c = (char)('a' + i);
            helper(res, pRoot.next[i], cur + c);
        }
    }
    public static void main(String[] args) {
        List<String> words = new ArrayList<>();
        words.add("ab");
        words.add("a");
        words.add("de");
        words.add("abde");
        words.add("cdf");
        words.add("cde");
        words.add("cdk");
        words.add("cdbb");

        Auto_CompleteOne test = new Auto_CompleteOne(words);
        String prefix = "c";
        List<String> res = test.find(prefix);
        System.out.println(res);


    }

}

```

## indeed2: BST TO ARRAY

```java
package Pratice;

import java.util.*;

class TreeNode{
    int val;
    TreeNode left, right;
    public TreeNode(int val){
        this.val = val;
        this.left = null;
        this.right = null;
    }
}
public class BT_to_Array {
    public int[] compressDenseTree(TreeNode root){
        int height = getHeight(root);
        if (height == 0){
            return new int[0];
        }
        //dense tree的情况下,默认null node位置放0。(假设原来的tree里面没有0)
        int len = (int) Math.pow(2, height);
        int[] heap = new int[len];
        //BFS一下就压缩好了
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        Queue<Integer> idxQueue = new LinkedList<>();
        //这里如果是1开头,那么就是(2i, 2i+1),如果是0开头,就是(2i+1,2i+2),
        //其实1,2,3一下就看出来了。
        // parent = i / 2;
        idxQueue.offer(1);

        while (!queue.isEmpty()){
            TreeNode cur = queue.poll();
            Integer curI = idxQueue.poll();
            heap[curI] = cur.val;
            if (cur.left != null){
                queue.offer(cur.left);
                idxQueue.offer(2*curI);
            }
            if (cur.right != null){
                queue.offer(cur.right);
                idxQueue.offer(2*curI+1);
            }
        }

        return heap;
    }
    public Map<Integer, Integer> compressSparseTree(TreeNode root){
        //前提假设是sparse tree,用map来记录,key是index,value是root的value
        Map<Integer, Integer> record = new HashMap<>();
        if (root == null) {
            return record;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        Queue<Integer> idxQueue = new LinkedList<>();
        queue.offer(root);
        idxQueue.offer(1);

        while (!queue.isEmpty()){
            TreeNode cur = queue.poll();
            int idx = idxQueue.poll();
            record.put(idx, cur.val);
            if (cur.left != null) {
                queue.offer(cur.left);
                idxQueue.offer(2*idx);
            }
            if (cur.right != null) {
                queue.offer(cur.right);
                idxQueue.offer(2*idx+1);
            }
        }
        return record;
    }
    public int getHeight(TreeNode root){
        if (root == null) {
            return 0;
        }
        int left = getHeight(root.left);
        int right = getHeight(root.right);
        return Math.max(left, right)+1;
    }

    public static void main(String[] args) {
        BT_to_Array test = new BT_to_Array();
        TreeNode t1 = new TreeNode(1);
        TreeNode t2 = new TreeNode(2);
        TreeNode t3 = new TreeNode(3);
        TreeNode t4 = new TreeNode(4);
        t1.left = t2;
        t1.right = t3;
        t2.left = t4;
        int[] res = test.compressDenseTree(t1);
        System.out.println(Arrays.toString(res));
        Map<Integer, Integer> resMap = test.compressSparseTree(t1);
        Iterator<Integer> ite = resMap.keySet().iterator();
        while (ite.hasNext()){
            int num = ite.next();
 System.out.println("root index is "+num + " root value is "+resMap.get(num));
        }
    }
}
```

## indeed3 Dice sum

```java

package Pratice;

public class Dice_Sum {
    int count = 0;
    public double getPossibility(int dice, int target){
        if (dice <= 0 || target < dice || target > 6 * dice) {
            return 0.0;
        }
        int total = (int) Math.pow(6, dice);
        helper(dice, target);

        System.out.println(count);
        System.out.println(total);
        return (double)count/total;
    }

    public void helper(int dice, int cur){
        if (dice == 0 && cur == 0){
            count++;
            return;
        }
        if (dice <= 0 || cur <= 0){
            return;
        }
        for (int i = 1; i <= 6; i++){
            helper(dice-1, cur-i);
        }
        return;
    }
    public double getMemPossibility(int dice, int target){
        if (dice <= 0 || target < dice || target > 6*dice) {
            return 0.0;
        }
        int total = (int) Math.pow(6, dice); //这里注意一下，pow的返回类型是double
        int[][] memo = new int[dice+1][target+1];
        int count = dfsMemo(dice, target, memo);
        return (double)count/total;
    }
    public int dfsMemo(int dice, int target, int[][] memo) {
        int res = 0;
        //三个终止条件，能加速就加速吧。
        if (dice == 0 && target == 0) return 1;
        if (target > dice * 6 || target < dice) {
            return 0;
        }
        if (memo[dice][target] != 0) {
            return memo[dice][target];
        }

        for (int i = 1; i <= 6; i++) {
            res += dfsMemo(dice - 1, target - i, memo);
        }
        //这一步是更新记忆矩阵
        memo[dice][target] = res;
        return res;
    }

    public static void main(String[] args) {
        Dice_Sum test = new Dice_Sum();
        int dice = 10;
        int target = 20;
        double res1 = test.getPossibility(dice, target);
        double res2 = test.getMemPossibility(dice,target);
        System.out.print(res1);
        System.out.print(res2);
    }
}

```

## Indeed4 excel to number

```java
package Pratice;

public class ExcelAndNumber {
  /*  A -> 1
    B -> 2
    C -> 3         ...
    Z -> 26
    AA -> 27
    AB -> 28*/
    public int titleToNumber(String s) {
            int r = 0;
            for (char c : s.toCharArray())
                r = r * 26 + (int) (c - 'A' + 1);
            return r;
        }
/*  1 -> A
    2 -> B
    3 -> C
    ...
            26 -> Z
    27 -> AA
    28 -> AB*/
    public String convertToTitle(int n) {
        String ans = "";
        while (n > 0) {
            n --;
            ans = (char)(n % 26 + 'A') + ans;
            n /= 26;
        }
        return ans;
    }

}

```

## indeed5 expired map

```java
package Pratice;

import java.util.*;

public class ExpiringMap<K, V> extends TimerTask {

    class Node<K, V> {

        private K key;
        private V value;
        private long duration;
        private long startTime;


        public Node(V value, long duration, long startTime) {
            this.value = value;
            this.duration = duration;
            this.startTime = startTime;
        }
    }

    private LinkedList<Node> list;
    HashMap<K, Node> map;

    public ExpiringMap() {
        map = new HashMap<>();
        list = new LinkedList<>();
    }


    public void put(K key, V value, long duration) {
        long startTime = System.currentTimeMillis();
        Node node = new Node(value, duration, startTime);
        map.put(key, node);
        checkTime();
        addToHead(value, duration, startTime);
    }

    public V get(K key) {
        checkTime();
        if (map.containsKey(key)) {
            Node node = map.get(key);
            return (V) node.value;
        } else {
            return null;
        }
    }

    public void checkTime() {

        long curTime = System.currentTimeMillis();
        while (!list.isEmpty()) {
            Node node = list.getLast();
            if (node.duration + node.startTime > curTime) {
                list.removeLast();
                map.remove(node.key);
            } else {
                break;
            }
        }
    }
    // TreeMap
    //public K floorKey(K key)
   // entrySet();
    // subMap
    // public NavigableMap<K,V> tailMap(K fromKey,boolean inclusive)


    public void addToHead(V value, long duration, long startTime) {
        Node node = new Node(value, duration, startTime);
        list.addFirst(node);
    }

    @Override
    public void run() {
        checkTime();
    }

    public static void main(String[] args) {
        Timer timer = new Timer(true);
        ExpiringMap expiringMap = new ExpiringMap();
        timer.schedule(expiringMap, 0, 1000);//1000毫秒检查一次
    }
}
```

## Indeed 6 Find peak

```java
package Pratice;

public class FindPeakElement {
   /* Input: nums = [1,2,1,3,5,6,4]
    Output: 5
    Explanation: Your function can return either index number
     1 where the peak element is 2, or index number 5 where the 
     peak element is 6.*/
   public int findPeakElement(int[] nums) {
       int N = nums.length;
       int left = 0, right = N - 1;
       while (right - left > 1) {
           int mid = left + (right - left) / 2;
           if (nums[mid] < nums[mid + 1]) {
               left = mid + 1;
           } else {
               right = mid;
           }
       }
       return (left == N - 1 || nums[left] > nums[left + 1]) ? left : right;
   }

    public int findPeakElement2(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[mid + 1]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
}

```

## indeed7: git commit

```java
package Pratice;

import java.util.*;
import java.util.LinkedList;

public class GitCommit {
    class Node {
        int id;
        List<Node> parents;
        public Node(int id) {
            this.id = id;
            this.parents = new ArrayList<>();
        }
    }

    public List<Node> getAllParents(Node root) {
        List<Node> res = new ArrayList<>();
        if (root == null || root.parents.size() == 0)
            return res;

        Queue<Node> queue = new java.util.LinkedList<>();
        Set<Node> visited = new HashSet<>();
        queue.add(root);
        visited.add(root);

        while (!queue.isEmpty()) {
            Node cur = queue.poll();
            for (Node each : cur.parents) {
                if (!visited.contains(each)) {
                    queue.add(each);
                    visited.add(each);
                    res.add(each);
                }
            }
        }
        return res;
    }

    public Node getLCA(Node n1, Node n2) {
        if (n1 == null || n2 == null || n1 == n2)
            return null;

        Set<Node> visited1 = new HashSet<>();
        Queue<Node> q1 = new java.util.LinkedList<>();
        q1.add(n1);
        visited1.add(n1);

        Set<Node> visited2 = new HashSet<>();
        Queue<Node> q2 = new LinkedList<>();
        q2.add(n2);
        visited2.add(n2);

        while (!q1.isEmpty() || !q2.isEmpty()) {
            int size1 = q1.size();  // traverse by level
            for (int i = 0; i < size1; i++) {
                Node cur1 = q1.remove();
                for (Node each : cur1.parents) {
                    if (visited2.contains(each))
                        return each;

                    if (!visited1.contains(each)) {
                        visited1.add(each);
                        q1.add(each);
                    }
                }
            }

            int size2 = q2.size();
            for (int i = 0; i < size2; i++) {
                Node cur2 = q2.remove();
                for (Node each : cur2.parents) {
                    if (visited1.contains(each))
                        return each;

                    if (!visited2.contains(each)) {
                        visited2.add(each);
                        q2.add(each);
                    }
                }
            }
        }
        return null;
    }
}
```

## indeed8: git version with parent node

```java
package Pratice;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

public class GitVersionWithParentNode {
	 public List<Integer> findCommits(GitNode root) {  
	        List<Integer> result = new ArrayList<>();
	        Set<GitNode> visited = new HashSet<>();
	        Queue<GitNode> queue = new LinkedList<>();
	         
	        findCommitsHelper(root, visited, queue, result);
	         
	        return result;
	    }
	     
	    private void findCommitsHelper(GitNode root, Set<GitNode> visited, 
	          Queue<GitNode> queue, List<Integer> result) {
	        if (root == null) {
	            return;
	        }
	         
	        queue.offer(root);
	         
	        while (!queue.isEmpty()) {
	            GitNode curr = queue.poll();
	            if (!visited.contains(curr)) {
	                visited.add(curr);
	                result.add(curr.val);
	                 
	                for (GitNode neighbor : curr.neighbors) {
	                    queue.offer(neighbor);
	                }
	            }
	        }
	    }
	 
	    public int findLCA(GitNode node1, GitNode node2) {
	        if (node1 == null || node2 == null) {
	            throw new NullPointerException();
	        }
	         
	        List<Integer> result1 = new ArrayList<>();
	        bfs(node1, result1);
	         
	        List<Integer> result2 = new ArrayList<>();
	        bfs(node2, result2);
	         
	        int i = result1.size() - 1;
	        int j = result2.size() - 1;
	         
	        for (; i >= 0 && j >= 0; i--, j--) {
	            if (result1.get(i) == result2.get(j)) {
	                continue;
	            } else {
	                break;
	            }
	        }
	         
	        return result1.get(i + 1);
	    }
	     
	    
	   

		private void bfs(GitNode root, List<Integer> result) {
	        if (root == null) {
	            return;
	        }
	         
	        Set<GitNode> visited = new HashSet<>();
	        Queue<GitNode> queue = new LinkedList<>();
	         
	        queue.offer(root);
	         
	        while (!queue.isEmpty()) {
	            GitNode curr = queue.poll();
	             
	            if (!visited.contains(curr)) {
	                result.add(curr.val);
	                visited.add(curr);
	                 
	                for (GitNode parent : curr.parents) {
	                    queue.offer(parent);
	                }
	            }
	        }
	    }
	     
	         
	        // Step 2: find out the root of the graph
	        GitNode root = null;
	        Map<GitNode, Integer> inDegree = new HashMap<>();
	        for (GitNode node : map.values()) {
	            if (!inDegree.containsKey(node)) {
	                inDegree.put(node, 0);
	            }
	             
	            for (GitNode neighbor : node.neighbors) {
	                if (inDegree.containsKey(neighbor)) {
	                    int degree = inDegree.get(neighbor);
	                    inDegree.put(neighbor, degree + 1);
	                } else {
	                    inDegree.put(neighbor, 1);
	                }
	            }
	        }
	         
	        for (GitNode node : inDegree.keySet()) {
	            if (inDegree.get(node) == 0) {
	                root = node;
	                break;
	            }
	        }
	         
	        System.out.println("Root is " + root.val);
	        node1 = map.get(3);
	        node2 = map.get(4);
	        List<Integer> result = sol.findCommits(root);
	        int lca = sol.findLCA(node1, node2);
	         
	        System.out.println("LCA is " + lca);
	         
	        for (Integer elem : result) {
	            System.out.println(elem);
	        }
	    }
	     
	    static class GitNode {
	        int val;
	        List<GitNode> neighbors;
	        List<GitNode> parents;
	         
	        public GitNode(int val) {
	            this.val = val;
	            this.neighbors = new ArrayList<>();
	            this.parents = new ArrayList<>();
	        }
	    }
}

```

## indeed9 job storage

```java
package Pratice;

import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

class Job_Storage{

    private static final int DEFAULT_SIZE = 2 << 24;
    private static final int[] seeds = new int[] { 5, 7, 11, 13, 31, 37, 61 };
    private BitSet bits = new BitSet(DEFAULT_SIZE);
    private SimpleHash[] func = new SimpleHash[seeds.length];
    public Job_Storage() {
        for (int i = 0; i < seeds.length; i++) {
            func[i] = new SimpleHash(DEFAULT_SIZE, seeds[i]);
        }
    }
    public void add(String value) {
        for (SimpleHash f : func) {
            bits.set(f.hash(value), true);
        }
    }
    public boolean contains(String value) {
        if (value == null) {
            return false;
        }
        boolean ret = true;
        for (SimpleHash f : func) {
            ret = ret && bits.get(f.hash(value));
        }
        return ret;
    }
    // 内部类，simpleHash
    public static class SimpleHash {
        private int cap;
        private int seed;
        public SimpleHash(int cap, int seed) {
            this.cap = cap;
            this.seed = seed;
        }
        public int hash(String value) {
            int result = 0;
            int len = value.length();
            for (int i = 0; i < len; i++) {
                result = seed * result + value.charAt(i);
            }
            return (cap - 1) & result;
        }
    }
    public void getBitSets(){
        BitSet bs = new BitSet();
        long value = 12345;
        long[] longs = new long[]{12345,23456,234567};
        BitSet bsl = bs.valueOf(longs);



    }


}
```

## indeed 10: knight move

```java
package Pratice;

import java.util.BitSet;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;

// A* is an informed search algorithm, or a best-first search,
// At each iteration of its main loop, A* needs to determine which 
//of its paths to extend. It does so based on the
// cost of the path and an estimate of the cost required to 
//extend the path all the way to the goal.
// Specifically, A* selects the path that minimizes
//
//{\displaystyle f(n)=g(n)+h(n)}f(n)=g(n)+h(n)
//where n is the next node on the path, g(n) is the cost of the 
//path from the start node to n, and h(n)
// is a heuristic function that estimates the cost of the cheapest 
//path from n to the goal.
// A* terminates when the path it chooses to extend is a path from
// start to goal or if there are no paths
// eligible to be extended. The heuristic function is problem-specific.
// If the heuristic function is admissible,
// meaning that it never overestimates the actual cost to get to the goal,
// A* is guaranteed to return a least-cost path from start to goal.
public class Knight {
    int[] dx = new int[]{-2, -1, 1, 2, 2, 1, -1, -2};
    int[] dy = new int[]{1, 2, 2, 1, -1, -2, -2, -1};
    HashSet<String> visited = new HashSet<>();
    public int minKnightMoves(int x, int y) {
        // no need to work with negatives, sames # steps in any quadrant
        int k = Math.abs(x);
        int s = Math.abs(y);
        // special case dips negative, return early
        if (k == 1 && s == 1) return 2;


        Queue<int[]> queue = new LinkedList();
        queue.add(new int[]{0, 0});
        int steps = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i=0; i < size; i++) {
                int[] pos = queue.remove();
                int pX = pos[0];
                int pY = pos[1];
                String pr = pX + "," + pY;
                if (pX == x && pY == y) return steps;
                if (visited.contains(pr)) continue;
                // need put in the outside of logic
                visited.add(pr);
                for (int d = 0; d < 8; d ++) {
                    queue.add(new int[]{pX+ dx[d], pY + dy[d]});
                }
            }
            steps++;
        }
        return steps;
    }

    public static void main(String[] args) {
      //  270, -21
        Knight kg = new Knight();
        System.out.println(kg.minKnightMoves(270,-21));
        String a = "abc";
        String b = "abc";
        System.out.println(a.hashCode() +",,,"+b.hashCode());
        BitSet bs = new BitSet();
        int k = -100;
        int s = -20;
        System.out.println(k +"," + s);

        long guid = Long.MAX_VALUE - 2;
        int l =   Long.hashCode(guid);
        System.out.println(l);

    }
}

```

## indeed11: MatchQuery

```java
package Pratice;

import java.util.*;

public class MachQueryRecommendation {

    class Pair {
        int maxCount;
        List<String> words;
        public Pair() {
            maxCount = 0;
            words = new ArrayList<>();
        }
    }
    Map<String, Set<String>> users;
    Map<String, Map<String, Integer>> wordRelevance;
    Map<String, Pair> ans;

    public MachQueryRecommendation() {
        users = new HashMap<>();
        wordRelevance = new HashMap<>();
        ans = new HashMap<>();
    }

    public String search(String user, String word) {
        if (!users.containsKey(user))
            users.put(user, new HashSet<>());
        if (!wordRelevance.containsKey(word))
            wordRelevance.put(word, new HashMap<>());
        if (!ans.containsKey(word))
            ans.put(word, new Pair());

        Pair max = ans.get(word);

        // update maps
        for (String each : users.get(user)) {
            // update word -> each
            if (!wordRelevance.get(word).containsKey(each))
                wordRelevance.get(word).put(each, 1);
            else
        wordRelevance.get(word).put(each, wordRelevance.get(word).get(each) + 1);
            if (ans.get(word).maxCount == wordRelevance.get(word).get(each))
           ans.get(word).words.add(each);
            else if (ans.get(word).maxCount < wordRelevance.get(word).get(each)) {
                ans.put(word, new Pair());
                ans.get(word).maxCount = wordRelevance.get(word).get(each);
                ans.get(word).words.add(each);
            }

            // update each -> word
            if (!wordRelevance.get(each).containsKey(word))
                wordRelevance.get(each).put(word, 1);
            else
     wordRelevance.get(each).put(word, wordRelevance.get(each).get(word) + 1);

            if (ans.get(each).maxCount == wordRelevance.get(each).get(word))
                ans.get(each).words.add(word);
            else if (ans.get(each).maxCount < wordRelevance.get(each).get(word)) {
                ans.put(each, new Pair());
                ans.get(each).maxCount = wordRelevance.get(each).get(word);
                ans.get(each).words.add(word);
            }
        }
        users.get(user).add(word);  // do not forget this

        StringBuilder sb = new StringBuilder();
        sb.append(max.maxCount);
        for (String each : max.words) {
            sb.append(" " + each);
        }
        return sb.toString();
    }

    public static void main(String[] args) {
       MachQueryRecommendation qr = new MachQueryRecommendation();
        Scanner sc = new Scanner(System.in);
        int numLines = Integer.parseInt(sc.nextLine());
        for (int i = 0; i < numLines; i++) {
            String[] parts = sc.nextLine().split(" ");
            System.out.println(qr.search(parts[0], parts[1]));
        }
    }
}


```

## indeed12: Merge K sorted Streams

```java
package Pratice;

import java.util.*;
/*Given n sorted stream, and a constant number k. The stream type is like iterator
and it has two functions, move() and getValue(), find a list of numbers that each
of them appears at least k times in these streams. Duplicate numbers in a stream
should be counted as once.
(1,1,1,3,4,5,6,6)
{1,2,7,7,8} K = 2 only 1 return since 6, 7 , duplicate in their own stream


Note: In the interview, we should use min heap method
 =============================================================================
code
=============================================================================*/

public class Merge_K_Sorted_Streams {
    static class Stream{
        Iterator<Integer> iterator;
        public Stream(List<Integer> list){
            this.iterator = list.iterator();
        }
        public boolean move(){
            return iterator.hasNext();
        }
        public int getValue(){
            return iterator.next();
        }
    }
    class Num{
        int val;
        Stream stream;
        public Num(Stream stream){
            this.stream = stream;
            this.val = stream.getValue();
        }
    }
    public List<Integer> getNumberInAtLeastKStream(List<Stream> lists, int k){
        List<Integer> res = new ArrayList<>();
        if (lists == null || lists.size() == 0) return res;
        PriorityQueue<Num> minHeap = new PriorityQueue<>((a,b) -> a.val - b.val);
        //先把所有的stream放进heap里面
        for (Stream s: lists) {
            if (s.move()){ //这里先判断一下要不就崩了
                minHeap.offer(new Num(s));
            }
        }

        while (!minHeap.isEmpty()){
            Num cur = minHeap.poll();
            int curValue = cur.val;
            int count = 1;
            while (cur.stream.move()){
                int nextVal = cur.stream.getValue();
                if (nextVal == curValue){
                    continue;
                }
                else {
                    cur.val = nextVal;
                    minHeap.offer(cur);
                    break;
                }
            }
            //更新其他stream的头部，就是把指针往后挪，相同的数字就计数了。
            while (!minHeap.isEmpty() && curValue == minHeap.peek().val){
                count++;
                Num num = minHeap.poll();
//                int numVal = num.val;

                while (num.stream.move()){
                    int nextVal = num.stream.getValue();
                    if (curValue == nextVal){
                        continue;
                    }
                    else {
                        num.val = nextVal;
                        minHeap.offer(num);
                        break;
                    }
                }
            }

            if (count >= k){
                res.add(curValue);
            }
        }
        return res;
    }

    public static void main(String[] args) {
        Merge_K_Sorted_Streams test = new Merge_K_Sorted_Streams();
        Integer[] arr1 = {1,2,3,4};
        Integer[] arr2 = {2,5,6};
        Integer[] arr3 = {2,5,7};

        List<Integer> l1 = Arrays.asList(arr1);
        List<Integer> l2 = Arrays.asList(arr2);
        List<Integer> l3 = Arrays.asList(arr3);

        List<Stream> lists = new ArrayList<>();
        lists.add(new Stream(l1));
        lists.add(new Stream(l2));
        lists.add(new Stream(l3));

        List<Integer> res = test.getNumberInAtLeastKStream(lists, 2);
        System.out.println(res);
    }


}
```

## indeed13 moving average

```java
package Pratice;

import java.util.Deque;
import java.util.LinkedList;
import java.util.Queue;

/*Given a stream of input, and a API int getNow() to get the current time stamp,
        Finish two methods:

        1. void record(int val) to save the record.
        2. double getAvg() to calculate the averaged value of all the records 
        in 5 minutes.*/
public class Moving_Average {
    private Queue<Event> queue = new LinkedList<>();
    private int sum = 0;

    //这个是每次记录读进来的时间的,这个不用自己写,就是直接返回当前系统时间
    private int getNow(){
        return 0; //暂时写个0，苟活
    }

    private boolean isExpired(int curTime, int preTime){
        return curTime - preTime > 300;
    }
    private void removeExpireEvent(){
        while (!queue.isEmpty() && isExpired(getNow(), queue.peek().time)){
            Event curE = queue.poll();
            sum -= curE.val;
        }
    }
    public void record(int val){
        Event event = new Event(getNow(), val);
        queue.offer(event);
        sum += val;

        removeExpireEvent();
    }

    public double getAvg(){
        removeExpireEvent();
        if (!queue.isEmpty()){
            return (double) sum/queue.size();
        }

        return 0.0;
    }
}
class Event{
    int val;
    int time;
    public Event(int val, int time){
        this.val = val;
        this.time = time;
    }
}

public class Movingaverage {
    private Queue<Event> queue = new LinkedList<>();
    private int sum = 0;

    // record an event
    public void record(int val) {
        Event event = new Event(getNow(), val);
        queue.offer(event);
        sum += event.val;

        removeExpiredRecords();
    }

    private int getNow() {
        return 0;
    }

    private void removeExpiredRecords() {
        while (!queue.isEmpty() && expired(getNow(), queue.peek().time)) {
            Event curr = queue.poll();
            sum -= curr.val;
        }
    }
    private double getAvg() {
        removeExpiredRecords();
        if (queue.isEmpty()) {
            return 0;
        } else {
            return (double) sum / queue.size();
        }
    }

    private boolean expired(int currTime, int prevTime) {
        return currTime - prevTime > 5;
    }

    class Event {
        int time;
        int val;

        public Event (int time, int val) {
            this.time = time;
            this.val = val;
        }
    }
}

public class Moving_Average2 {
    //queue的容量被限制
    private Deque<Event> queue = new LinkedList<>(); //改成deque的话，可以从后面查
    private long sum = 0; //改用long显得严谨点儿
    int dataNum = 0;

    //这个是每次记录读进来的时间的,这个不用自己写,就是直接返回当前系统时间
    //假设它返回的是秒
    private int getNow(){
        return 0;
    }

    private boolean isExpired(int curTime, int preTime){
        return curTime - preTime > 300;
    }
    private void removeExpireEvent(){
        while (!queue.isEmpty() && isExpired(getNow(), queue.peekFirst().time)){
            Event curE = queue.poll();
            sum -= curE.val;
            dataNum -= curE.size;
        }
    }
    public void record(int val){ //其实就是record这里有了两种办法，
    //一种是建个新的，另一种就是合起来
        Event last = queue.peekLast();
        if (getNow() - last.time < 10){
            last.size += 1;
            last.val += val;
        }
        else {
            Event event = new Event(getNow(), val);
            queue.offer(event);
        }
        dataNum += 1;
        sum += val;
        removeExpireEvent();
    }

    public double getAvg(){
        removeExpireEvent();
        if (!queue.isEmpty()){
            return (double) sum/dataNum;
        }
        return 0.0;
    }
}
class Event2{
    int val;
    int time;
    int size;
    public Event2(int val, int time){
        this.val = val;
        this.time = time;
        this.size = 1;
    }


    //实现find Median，其实O1操作的话，要始终维护两个heap，这样塞进去会很慢
//原有基础上实现的话，那就直接quick select的办法了。
//复杂度是On，因为每次average case是去掉一半，就是O(n)+O(n/2)+O(n/4)+... 最后出来是O(2n)
    //那这个需要把整个queue给倒出来再塞回去。
    public double getMedian(){
        removeExpireEvent();
        int[] temp = new int[queue.size()];
        for (int i = 0; i<queue.size(); i++){
            temp[i] = queue.poll().val;
        }
        //这里还得把queue还原回去,先不写了。
        int len = temp.length;
        if (len % 2 == 0){
            return 0.5*(findKth(temp, len/2, 0, len-1) +
             findKth(temp, len/2-1, 0, len-1));
        }
        return (double)findKth(temp, len/2, 0, len-1);
    }
    public int findKth(int[] temp, int k, int start, int end){
        int pivot = temp[start];
        int left = start, right = end;
        while (left < right){
            while (temp[right] > pivot && left < right){
                right--;
            }
            while (temp[left] <= pivot && left < right){
                left++;
            }
            swap(temp, left, right);
        }
        swap(temp, start, right);
        if (k == right){
            return pivot;
        }
        else if (k < right){
            return findKth(temp, k, start, right-1);
        }

        return findKth(temp, k, right+1, end);
    }
    public void swap(int[] temp, int left, int right){
        int i = temp[left];
        temp[left] = temp[right];
        temp[right] = i;
    }
}

```

## indeed14: normalizedTitle

```java
package Pratice;

public class Normalized_Title {
    public String getHighestTitle(String rawTitle, String[] cleanTitles) {
        String res = "";
        int highScore = 0;
        for (String ct : cleanTitles) {
            int curScore = getScore(rawTitle, ct);
            if (curScore > highScore) {
                highScore = curScore;
                res = ct;
            }
        }
        return res;
    }

    //思路非常简单,两个title分别去查一下就行了。
    //这个下面有问题，比如a b c和d c的例子，那只能开二维矩阵去搜最高分。
    //不考虑顺序的话，就用map来记录词和位置吧。（而且它说没有重复的词，也是暗示用map）
    public int getScore(String raw, String ct) {
        int s = 0, temp = 0;
        int rIdx = 0, cIdx = 0;
        String[] rA = raw.split(" ");
        String[] cA = ct.split(" ");
        while (rIdx < rA.length) {
            String rCur = rA[rIdx];
            String cCur = cA[cIdx];
            if (!rCur.equals(cCur)) {
                cIdx = 0;
                temp = 0;
            } else {
                temp++;
                cIdx++;
            }
            rIdx++;
            s = Math.max(s, temp);
            if (cIdx == cA.length) break;
        }

        return s;
    }

    public static void main(String[] args) {
        Normalized_Title test = new Normalized_Title();
        String rawTitle = "senior software engineer";
        String[] cleanTitles = {
                "software engineer",
                "mechanical engineer",
                "senior software engineer"};

        String result = test.getHighestTitle(rawTitle, cleanTitles);
        System.out.println(result);
    }


    /* =============================================================================
    Follow Up
    =============================================================================
    raw title和clean title中有duplicate word怎么办
    比如raw = "a a a b", clean = "a a b"
        这样的话，靠指针就抓不出第二个a开始的aab，因为查第二个a的时候，是当做第二个a来算的。
        这个case
        应该返回3而不是2。
        那还想什么，开二维矩阵走DP吧。
=============================================================================*/
    public String getHightestTitleWithDup(String rawTitle, String[] cleanTitles) {
        String res = "";
        int highScore = 0;
        String[] rA = rawTitle.split(" ");
        for (String ct : cleanTitles) {
            String[] cA = ct.split(" ");
            int temp = getScoreWithDup(rA, cA);
            System.out.println("temp is " + temp);
            if (temp > highScore) {
                highScore = temp;
                res = ct;
            }
        }
        return res;
    }

    //二维矩阵里面每个位置都要查,因为不一定是从哪个位置开始匹配的,反正复杂度都是一样的。
    public int getScoreWithDup(String[] rA, String[] cA) {
        int col = rA.length;
        int row = cA.length;
        int res = 0;
        int[][] dp = new int[row][col];
        for (int i = 0; i < row; i++) {
            String cCur = cA[i];
            for (int j = 0; j < col; j++) {
                String rCur = rA[j];
                if (rCur.equals(cCur)) {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = Math.max(1, dp[i - 1][j - 1] + 1);
                    }
                }
                res = Math.max(res, dp[i][j]);
            }
        }
        return res;
    }
}
```

## indeed16 : min cost root

```java

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

class Edge{
    Node2 node; //表示这个edge的尾巴指向哪里。
    int cost;
    public Edge(Node2 n, int cost) {
        this.node = n;
        this.cost = cost;
    }
}
// like 129. Sum Root to Leaf Numbers

class Node2 {
    List<Edge> edges; //表示从这个头出发的所有edge
    public Node2(){
        this.edges = new ArrayList<>();
    }
}

public class Root_to_Leaf_Min_Cost{
    int minCost = Integer.MAX_VALUE;
    //返回最短路径上面的所有Edge
    public List<Edge> getMinPath(Node2 root){
        List<Edge> res = new ArrayList<>();
        List<Edge> temp = new ArrayList<>();
        dfs(res, temp, root, 0);
        return res;
    }
    //就是普通的DFS
    public void dfs(List<Edge> res, List<Edge> temp, Node2 root, int curCost){
        if (root == null){
            return;
        }
        if (root.edges.size() == 0){
            if (curCost < minCost){
                minCost = curCost;
                res.clear();
                res.addAll(temp);
                return;
            }
        }
        for (Edge e : root.edges){
            Node2 next = e.node;
            temp.add(e);
            dfs(res, temp, next, curCost+e.cost);
            temp.remove(temp.size()-1);
        }
    }
    //这个只返回个最小cost
    public int getMinCost(Node2 root){
        if (root == null) {
            return 0;
        }
        helper(root, 0);
        return minCost;
    }
    public void helper(Node2 root, int curCost){
        if (root.edges.size() == 0){
            minCost = Math.min(minCost, curCost);
            return;
        }
        for (Edge e : root.edges){
            Node2 next = e.node;
            helper(next, curCost + e.cost);
        }
    }
    public static void main(String[] args) {
        Root_to_Leaf_Min_Cost test = new Root_to_Leaf_Min_Cost();
        /*
         *       n1
         *   e1 /  \ e3
         *     n2   n3
         * e2 /
         *   n4
         *
         * */
        Node2 n1 = new Node2();
        Node2 n2 = new Node2();
        Node2 n3 = new Node2();
        Node2 n4 = new Node2();
        Edge e1 = new Edge(n2,1);
        Edge e2 = new Edge(n4,2);
        Edge e3 = new Edge(n3,5);
        n1.edges.add(e1);
        n1.edges.add(e3);
        n2.edges.add(e2);
        int res = test.getMinCost(n1);
        System.out.println(res);
        System.out.println(test.getMinPathInGraph(n1).size());
       // System.out.println("3 = "+res);
    }

    int miniCost = Integer.MAX_VALUE;
    Map<Node2, Integer> dist = new HashMap<>();
    public List<Edge> getMinPathInGraph(Node2 root){
        List<Edge> res = new ArrayList<>();
        List<Edge> temp = new ArrayList<>();
        dfsInGraph(res, temp, root, 0);
        return res;
    }
public void dfsInGraph(List<Edge> res, List<Edge> temp, Node2 node, int curCost){
        if (node == null) return;
        if (dist.containsKey(node)) return;
        dist.put(node, curCost);
        if (node.edges.size() == 0){
            if (curCost < miniCost){
                miniCost = curCost;
                res.clear();
                res.addAll(temp);
            }
            return;
        }

        for (Edge e : node.edges){
            Node2 next = e.node;
            temp.add(e);
            dfsInGraph(res, temp, next, curCost + e.cost);
            temp.remove(temp.size()-1);
        }
    }
}
```

## indeed17 word distance

```java

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class STWordDistance {
    public int shortestDistance(String[] words, String word1, String word2) {
        if (words == null || words.length == 0) {
            return 0;
        }
        int index1 = -1, index2 = -1;
        int distance = Integer.MAX_VALUE;
        for (int i = 0; i < words.length; i++) {
            if (words[i].equals(word1)) {
                index1 = i;
            }
            if (words[i].equals(word2)) {
                index2 = i;
            }
            if (index1 != -1 && index2 != -1) {
                distance = Math.min(distance, Math.abs(index1 - index2));
            }
        }
        return distance;
    }

        private Map<String, List<Integer>> map;

        public STWordDistance(String[] words) {
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

## indeed18: unrolled linked list

```java
package Pratice;
/*Given a LinkedList, every node contains a array. 
Every element of the array is char
		implement two functions
		1. get(int index) find the char at the index
		2. insert(char ch, int index) insert the char to the index
        3:删除一个数怎么处理，需要注意的地方也就是如果node空了就删掉吧。
		那就需要记录前一个node了，这样比较好删掉当前node。*/
class UnrolledLinkedList {
	class Node {
		char[] chars = new char[5];
		Node next;
		int len;
	}

	Node head;
	int totalLen;

	public UnrolledLinkedList(Node head, int totalLen) {
		this.head = head;
		this.totalLen = totalLen;
	}

	public char get(int index) {
		if (index < 0 || index >= totalLen)
			return ' ';

		Node cur = head;
		while (cur != null) {
			if (index >= cur.len)
				index -= cur.len;
			else {
				return cur.chars[index];
			}
			cur = cur.next;
		}
		return ' ';
	}

	public void insert(int index, char c) {
		if (index < 0 || index > totalLen)
			return;

		Node prev = new Node();
		prev.next = head;
		Node cur = head;
		while (cur != null) {
			if (index >= cur.len) {
				index -= cur.len;
			} else {
				// node is full
				if (cur.len == 5) {
					Node newNode = new Node();
					newNode.chars[0] = cur.chars[4];
					newNode.len = 1;
					newNode.next = cur.next;
					cur.next = newNode;
					cur.len--;
				}

				// normal case
				cur.len++;
				for (int i = cur.len - 1; i > index; i--)
					cur.chars[i] = cur.chars[i-1];
				cur.chars[index] = c;
				break;
			}
			prev = cur;
			cur = cur.next;
		}

		// node is null
		if (cur == null) {
			Node newNode = new Node();
			newNode.chars[0] = c;
			newNode.len = 1;
			prev.next = newNode;
			// case 4: insert 1st element
			if (head == null)
				head = prev.next;
		}
		totalLen++;
	}

	public void delete(int index) {
		if (index < 0 || index >= totalLen)
			return;

		Node prev = new Node();
		prev.next = head;
		Node cur = head;
		while (cur != null) {
			if (index >= cur.len) {
				index -= cur.len;
			} else {
				if (cur.len == 1) {
					prev.next = cur.next;
				} else {
					for (int i = index; i < cur.len - 1; i++) {
						cur.chars[i] = cur.chars[i+1];
					}
					cur.len--;
				}
				break; // DO NOT forget this
			}
			prev = cur;
			cur = cur.next;
		}
		// corner case: only one element in the linked list
		head = prev.next;
		totalLen--;
	}

}



```

## Indeed 19: validate python

```java
package Pratice;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class Validate_Python_Indentation {
    public boolean validate(String[] lines){
        //就用stack来存之前的line就行
        Stack<String> stack = new Stack<>();
        for (String line : lines){
            int level = getIndent(line);
            //先检查是不是第一行
            if (stack.isEmpty()){
                if (level != 0) {
                    System.out.println(line);
                    return false;
                }
            }
            //再检查上一行是不是control statement
            else if (stack.peek().charAt(stack.peek().length()-1) ==':'){
                if (getIndent(stack.peek()) + 1 != level){
                    System.out.println(line);
                    return false;
                }
            }
            else {
                while (!stack.isEmpty() && getIndent(stack.peek()) > level){
                    stack.pop();
                }
                if (getIndent(stack.peek()) != level){
                    System.out.println(line);
                    return false;
                }
            }
            stack.push(line);
        }
        return true;
    }
    //这里如果它说n个空格算一次tab的话，就最后返回的时候res/n好了。
    public int getIndent(String line){
        int res = 0;
        for (int i = 0; i < line.length(); i++){
            if (line.charAt(i) == ' '){
                res++;
            }
            else break;
        }
        return res;
    }
    public static void main(String[] args) {
        Validate_Python_Indentation test = new Validate_Python_Indentation();
        String[] lines = {
                "def:",
                " abc:",
                "  b2c:",
                "   cc",
                " b5c",
                "b6c"
        };
        System.out.println(test.validate(lines));
        //先这样吧，应该行了。
    }


/*============= Following Code Credit to Zhu Siyao ===============*/

    public static boolean valid_python_indentation(List<String> inputs){
        Stack<Integer> stack = new Stack<>();
        for(int i=0;i<inputs.size();i++){
            String str =  inputs.get(i);
            String abbr = getAbbr(str);
            int level = str.length()-abbr.length();

            if(i!=0 && inputs.get(i-1).charAt(inputs.get(i-1).length()-1)==':'){
                if(level<=stack.peek()) return false;
            }else{
                while(!stack.isEmpty() && level<stack.peek()) stack.pop();
                if(!stack.isEmpty() && level!=stack.peek()) return false;

            }
            stack.push(level);
            System.out.println(level);
        }

        return true;

    }

    private static String getAbbr(String str) {
        String result = str.trim();
        return result;
    }



}
```


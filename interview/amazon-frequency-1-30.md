# Amazon Frequency 1~ 30

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


# Karat

## Sub Domain visits

```java
class Solution {
    public List<String> subdomainVisits(String[] cpdomains) {
        HashMap<String, Integer> map = new HashMap<>();
        for(int i = 0; i < cpdomains.length; i ++) {
            String cur =  cpdomains[i];
            String[] doms = cur.split("\\s+");
            int value  = Integer.valueOf(doms[0]);
            String[] subs = doms[1].split("\\.");
            String pre = "";
            for(int j = subs.length - 1; j >= 0; j --) {
               String combine = subs[j] + pre;
               map.put(combine, map.getOrDefault(combine, 0) + value);
               pre = "." + combine;
            }
        }
        ArrayList<String> res = new ArrayList<>();
        for(String key : map.keySet()){
            StringBuilder sb = new StringBuilder();
            sb.append(map.get(key)).append(" ").append(key);
            res.add(sb.toString());
        }
        return res;
    }
}

```

## Longest common history

```java
public static List<String> longestCommonContinuousHistory(String[] history1,
                     String[] history2) {
        int m = history1.length; int n = history2.length;
        int count = -1;
        int index = -1;
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= history1.length; i++) {
            for (int j = 1; j <= history2.length; j++) {
                if (history1[i - 1] == history2[j - 1]) {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                    if(dp[i][j] > count){
                      count = dp[i][j];
                      index = i - 1;
                    }
                }else {
                    dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
                }
            }
        }
        List<String> res = new ArrayList<>();
        while (index + 1 - count >= 0){
            res.add(history1[index]);
        }
        return res;
    }

}
```

## Ads Conversion Rate

```python
    public static String[] AdsConversion(String[] completedUserId,
              String[] adClicks, String[] allUserIp) {
        HashSet<String> userIdSet = new HashSet<>();
        for (String user : completedUserId) {
            userIdSet.add(user);
        }
        Map<String, List<String>> adTextMap = new HashMap<>();
        Map<String, String> ipUserMap = new HashMap<>();
        for (String adClick : adClicks) {
            String[] parsed = adClick.split(",");
            String text = parsed[2];
            String ip = parsed[0];
            adTextMap.putIfAbsent(text, new ArrayList<>());
            adTextMap.get(text).add(ip);
        }

        for (String userIp : allUserIp) {
            String[] userAndIp = userIp.split(",");
            ipUserMap.putIfAbsent(userAndIp[1], userAndIp[0]);
        }

        String[] result = new String[adTextMap.keySet().size()];
        int i = 0;
        for (String text : adTextMap.keySet()) {
            List<String> clicks = adTextMap.get(text);
            int buyer = 0;
            for (String user : clicks) {
                if (userIdSet.contains(ipUserMap.get(user))) {
                    buyer++;
                }
            }
            result[i++] = buyer + " of " + clicks.size() + " " + text;
        }
        return result;
    }
```

## Student course overlap

```python
/*
Students may decide to take different "tracks" or sequences of courses in
 the Computer Science curriculum. There may be more than one track that 
 includes the same course, but each student follows a single linear track
  from a "root" node to a "leaf" node. In the graph below, 
  their path always moves left to right.

Write a function that takes a list of (source, destination) pairs, 
and returns the name of all of the courses that the students could be 
taking when they are halfway through their track of courses.

Sample input:
all_courses = [
    ["Logic", "COBOL"],
    ["Data Structures", "Algorithms"],
    ["Creative Writing", "Data Structures"],
    ["Algorithms", "COBOL"],
    ["Intro to Computer Science", "Data Structures"],
    ["Logic", "Compilers"],
    ["Data Structures", "Logic"],
    ["Creative Writing", "System Administration"],
    ["Databases", "System Administration"],
    ["Creative Writing", "Databases"],
    ["Intro to Computer Science", "Graphics"],
]

Sample output (in any order):
          ["Data Structures", "Creative Writing", "Databases", 
          "Intro to Computer Science"]

All paths through the curriculum (midpoint *highlighted*):

*Intro to C.S.* -> Graphics
Intro to C.S. -> *Data Structures* -> Algorithms -> COBOL
Intro to C.S. -> *Data Structures* -> Logic -> COBOL
Intro to C.S. -> *Data Structures* -> Logic -> Compiler
Creative Writing -> *Databases* -> System Administration
*Creative Writing* -> System Administration
Creative Writing -> *Data Structures* -> Algorithms -> COBOL
Creative Writing -> *Data Structures* -> Logic -> COBOL
Creative Writing -> *Data Structures* -> Logic -> Compilers

Visual representation:

                    ____________
                    |          |
                    | Graphics |
               ---->|__________|
               |                          ______________
____________   |                          |            |
|          |   |    ______________     -->| Algorithms |--\     _____________
| Intro to |   |    |            |    /   |____________|   \    |           |
| C.S.     |---+    | Data       |   /                      >-->| COBOL     |
|__________|    \   | Structures |--+     ______________   /    |___________|
                 >->|____________|   \    |            |  /
____________    /                     \-->| Logic      |-+      _____________
|          |   /    ______________        |____________|  \     |           |
| Creative |  /     |            |                         \--->| Compilers |
| Writing  |-+----->| Databases  |                              |___________|
|__________|  \     |____________|-\     _________________________
               \                    \    |                       |
                \--------------------+-->| System Administration |
                                         |_______________________|

Complexity analysis variables:

n: number of pairs in the input

*/

'use strict';

const prereqs_courses1 = [
  ['Data Structures', 'Algorithms'],
  ['Foundations of Computer Science', 'Operating Systems'],
  ['Computer Networks', 'Computer Architecture'],
  ['Algorithms', 'Foundations of Computer Science'],
  ['Computer Architecture', 'Data Structures'],
  ['Software Design', 'Computer Networks'],
];

const prereqs_courses2 = [
  ['Data Structures', 'Algorithms'],
  ['Algorithms', 'Foundations of Computer Science'],
  ['Foundations of Computer Science', 'Logic'],
];

const prereqs_courses3 = [['Data Structures', 'Algorithms']];

const allCourses = [
  ['Logic', 'COBOL'],
  ['Data Structures', 'Algorithms'],
  ['Creative Writing', 'Data Structures'],
  ['Algorithms', 'COBOL'],
  ['Intro to Computer Science', 'Data Structures'],
  ['Logic', 'Compilers'],
  ['Data Structures', 'Logic'],
  ['Creative Writing', 'System Administration'],
  ['Databases', 'System Administration'],
  ['Creative Writing', 'Databases'],
  ['Intro to Computer Science', 'Graphics'],
];

q1
public static Map<String[], String[]> findPairs(String[][] coursePairs) {
        Map<String, HashSet<String>> map = new HashMap<>();
        Map<String[], String[]> result = new HashMap<>();
        for (String[] coursesPair : coursePairs) {
            if (!map.containsKey(coursesPair[0])) {
                map.put(coursesPair[0], new HashSet<>());
            }
            map.get(coursesPair[0]).add(coursesPair[1]);
        }

        List<String> students = new ArrayList<>(map.keySet());
        for (int i = 0; i < students.size(); i++) {
            for (int j = i + 1; j < students.size(); j++) {
                String[] key = new String[] { students.get(i), students.get(j) };
                List<String> courses = new ArrayList<>();
                for (String c1 : map.get(key[0])) {
                    if (map.get(key[1]).contains(c1)) {
                        courses.add(c1);
                    }
                }
                String[] value = new String[courses.size()];
                for (int k = 0; k < value.length; k++) {
                    value[k] = courses.get(k);
                }
                result.put(key, value);
            }
        }
        return result;
    }
    
    q2
     public static char findMediumCourse(char[][] courses) {
        int[] count = new int[26];
        Map<Character, Character> map = new HashMap<>();
        for (char[] course : courses) {
            count[course[0] - 'A']++;
            count[course[1] - 'A']++;
            map.put(course[0], course[1]);
        }
        char start = 'A';
        for (int i = 0; i < 26; i++) {
            if (count[i] == 1) {
                start = (char) ('A' + i);
                break;
            }
        }
        int middleCourse = map.keySet().size() / 2;
        while (middleCourse-- > 0) {
            start = map.get(start);
        }
        return start;
    }
    
    q3
     public static Set<String> halfWayLessons(String[][] courses) {
        Set<String> result = new HashSet<>();
        Map<String, Integer> inorder = new HashMap<>();
        Map<String, List<String>> graph = new HashMap<>();
        Map<String, Boolean> visited = new HashMap<>();

        for (String[] course : courses) {
            String source = course[0];
            String des = course[1];
            visited.put(source, false);
            visited.put(des, false);
            if (!graph.containsKey(source)) {
                graph.put(source, new ArrayList<>());
            }
            if (!graph.containsKey(des)) {
                graph.put(des, new ArrayList<>());
            }
            graph.get(source).add(des);
            if (!inorder.containsKey(source)) {
                inorder.put(source, 0);
            }
            inorder.put(des, inorder.getOrDefault(des, 0) + 1);
        }

        for (String key : inorder.keySet()) {
            if (inorder.get(key).equals(0)) {
                LinkedList<String> temp = new LinkedList<>();
                temp.add(key);
                backtrack(key, graph, temp, result);
            }
        }

        return result;
    }

    public static void backtrack(String start, Map<String, List<String>> graph, List<String> temp, Set<String> result) {
        int size = graph.get(start).size();
        if (size == 0) {
            result.add(temp.get((temp.size() + 1) / 2 - 1));
            return;
        }
        for (int i = 0; i < size; i++) {
            String next = graph.get(start).get(i);
            temp.add(next);
            backtrack(next, graph, temp, result);
            temp.remove(temp.size() - 1);
        }

    }
```

## Find rectangle

```python
    int[][] findOneRectangle(int[][] board) {
        int[][] result = new int[2][2];
        if ( board.length == 0 || board[0].length == 0) {
            return result;
        }

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (board[i][j] == 0) {
                    result[0][0] = i;
                    result[0][1] = j;
                    int height = 1, width = 1;
                    while (i + height < board.length && board[i + height][j] == 0) {
                        height++;
                    }
                    while (j + width < board[0].length && board[i][j + width] == 0) {
                        width++;
                    }
                    result[1][0] = i + height - 1; 
                    result[1][1] = j + width - 1;
                  
                    break;
                }
                if (result.length != 0) {
                    break;
                }
            }
        }
        return result;
    }
    

function findMultipleRectangle(board) {
  if (!board || board.length === 0 || board[0].length === 0) {
    return [];    
  }
  const result = [];
  for (let i = 0; i < board.length; i++) {
    for (let j = 0; j < board[0].length; j++) {
      if (board[i][j] === 0) {
        const rectangle = [[i, j]];
        board[i][j] = 1;
        let height = 1, width = 1;
        while (i + height < board.length && board[i + height][j] === 0) {
          height++;
        }
        while (j + width < board[0].length && board[i][j + width] === 0) {
          width++;
        }
        for (let h = 0; h < height; h++) {
          for (let w = 0; w < width; w++) {
            board[i + h][j + w] = 1;
          }
        }
        rectangle.push([i + height - 1, j + width - 1]);
        result.push(rectangle);
      }
    }
  }
  return result;
}
// dfs
function findMultipleShapes(board) {
  if (!board || board.length === 0 || board[0].length === 0) {
    return [];    
  }
  const result = [];
  const floodFillDFS = (x, y, shape) => {
    if (x < 0 || x >= board.length || y < 0 || y >= board[0].length ||
     board[x][y] === 1) {
      return;
    }
    board[x][y] = 1;
    shape.push([x, y]);
    floodFillDFS(x + 1, y, shape);
    floodFillDFS(x - 1, y, shape);
    floodFillDFS(x, y - 1, shape);
    floodFillDFS(x, y + 1, shape);
  };
  for (let i = 0; i < board.length; i++) {
    for (let j = 0; j < board[0].length; j++) {
      if (board[i][j] === 0) {
        shape = [];
        floodFillDFS(i, j, shape);
        result.push(shape);
      }
    }
  }
  return result;
}
```

## calculator

```python
  public static int basicCalculator(String expression) {
        if (expression == null || expression.length() == 0)
            return 0;
        char[] expressionChar = expression.toCharArray();
        int num = 0;
        int sign = 1;
        for (int i = 0; i < expressionChar.length; i++) {
            if (expressionChar[i] == '+')
                sign = 1;
            else if (expressionChar[i] == '-')
                sign = -1;
            else if (expressionChar[i] >= '0' && expressionChar[i] <= '9') {
                int temp = expressionChar[i] - '0';
                while (i + 1 < expressionChar.length
                        && (expressionChar[i + 1] >= '0' && expressionChar[i + 1] <= '9')) {
                    temp *= 10;
                    temp += expressionChar[++i] - '0';
                }
                num += temp * sign;
            }
        }
        return num;
    }

public static int basicCalculator2(String expression) {
        if (expression == null || expression.length() == 0)
            return 0;
        char[] expressionChar = expression.toCharArray();
        Stack<Integer> stack = new Stack<>();
        int num = 0;
        int sign = 1;
        for (int i = 0; i < expressionChar.length; i++) {
            if (expressionChar[i] == '+')
                sign = 1;
            else if (expressionChar[i] == '-')
                sign = -1;
            else if (expressionChar[i] >= '0' && expressionChar[i] <= '9') {
                int temp = expressionChar[i] - '0';
                while (i + 1 < expressionChar.length
                        && (expressionChar[i + 1] >= '0' && expressionChar[i + 1] <= '9')) {
                    temp *= 10;
                    temp += expressionChar[++i] - '0';
                }
                num += temp * sign;
            } else if (expressionChar[i] == '(') {
                stack.push(num);
                stack.push(sign);
                num = 0;
                sign = 1;
            } else if (expressionChar[i] == ')') {
                num = num * stack.pop() + stack.pop();
            }
        }
        return num;
    }
    
```

## Word wrap & word processor

```python
public static List<String> wrapLines1(String[] words, int maxLength){
    List<String> ans = new ArrayList<>();
    StringBuilder sb = new StringBuilder();
    int p = 0;
    while(p < words.length){
        if(sb.length() == 0)
            // assume all words length no exceed to maxLength
            sb.append(words[p++]);

        else if(sb.length() + 1 + words[p].length() <= maxLength){
            sb.append('-');
            sb.append(words[p++]);
        }
        else{
            ans.add(sb.toString());
            sb.setLength(0);
        }
    }
    if(sb.length() != 0) ans.add(sb.toString());
    return ans;
}


public static List<String> wrapLines2(String[] lines, int maxLength){
    List<String> unbalanced = new ArrayList<>();
    List<String> words = new ArrayList<>();
    for(String line : lines){
        String[] word_collection = line.split(" ", -1);
        Collections.addAll(words, word_collection);
    }
    StringBuilder sb = new StringBuilder();
    int p = 0;
    while(p < words.size()){
        if(sb.length() == 0)
            // assume all words length no exceed to maxLength
            sb.append(words.get(p++));

        else if(sb.length() + 1 + words.get(p).length() <= maxLength){
            sb.append('-');
            sb.append(words.get(p++));
        }
        else{
            unbalanced.add(sb.toString());
            sb.setLength(0);
        }
    }
    if(sb.length() != 0) unbalanced.add(sb.toString());
    //now we have un-balanced result, then balance it
    List<String> balanced = new ArrayList<>();
    for(String line : unbalanced){
        StringBuilder cur_line = new StringBuilder(line);
        int num_needed = maxLength - cur_line.length();
        if(!cur_line.toString().contains("-")){
            balanced.add(cur_line.toString());
            continue;
        };
        while(num_needed > 0){
            int i = 0;
            while(i < cur_line.length() - 1){
                if(cur_line.charAt(i) == '-' && cur_line.charAt(i + 1) != '-'){
                    cur_line.insert(i + 1, '-');
                    num_needed--;
                    i++;
                    if(num_needed == 0) break;
                }
                i++;
            }
        }
        balanced.add(cur_line.toString());
    }
    return balanced;
}

```

## Is valid matrix

```python
 public static boolean validMatrix1(int[][] matrix) {
        int N = matrix.length;
        for (int i = 0; i < N; i++) {
            HashSet<Integer> rowSet = new HashSet<>();
            HashSet<Integer> colSet = new HashSet<>();
            for (int j = 0; j < N; j++) {
                int curRow = matrix[i][j];
                if (curRow < 1 || curRow > N || !rowSet.add(curRow)) {
                    return false;
                }
                int curCol = matrix[j][i];
                if (curCol < 1 || curCol > N || !colSet.add(curCol)) {
                    return false;
                }
            }
        }
        return true;
    }

 public static boolean checkNonogram(int[][] matrix, int[][] rows, int[][] cols) {
        return checkRows(matrix, rows) && checkCols(matrix, cols);
    }

    private static boolean checkRows(int[][] matrix, int[][] rows) {
        if (matrix.length != rows.length)
            return false;
        for (int i = 0; i < matrix.length; i++) {
            List<Integer> pattern = new ArrayList<>();
            int num = 0;
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] == 1 && num != 0) {
                    pattern.add(num);
                    num = 0;
                }
                if (matrix[i][j] == 0) {
                    num += 1;
                }
            }
            if (num != 0) {
                pattern.add(num);
            }

            if (pattern.size() == 0 && rows[i].length == 0) {
                continue;
            }

            if (pattern.size() != rows[i].length) {
                return false;
            }

            for (int k = 0; k < pattern.size(); k++) {
                if (pattern.get(k) != rows[i][k])
                    return false;
            }
        }
        return true;
    }

    private static boolean checkCols(int[][] matrix, int[][] cols) {
        if (matrix[0].length != cols.length)
            return false;
        for (int i = 0; i < matrix[0].length; i++) {
            List<Integer> pattern = new ArrayList<>();
            int num = 0;
            for (int j = 0; j < matrix.length; j++) {
                if (matrix[j][i] == 1 && num != 0) {
                    pattern.add(num);
                    num = 0;
                }
                if (matrix[j][i] == 0) {
                    num += 1;
                }
            }
            if (num != 0) {
                pattern.add(num);
            }

            if (pattern.size() == 0 && cols[i].length == 0) {
                continue;
            }

            if (pattern.size() != cols[i].length) {
                return false;
            }
            for (int k = 0; k < pattern.size(); k++) {
                if (pattern.get(k) != cols[i][k])
                    return false;
            }

        }
        return true;
    }
```

## Node ancestor

```python
 public static List<Integer> zeroOrOneAncestor(int[][] edges) {
        List<Integer> result = new ArrayList<>();
        if (edges == null || edges.length == 0)
            return result;

        // Build a graph using a map
        Map<Integer, HashSet<Integer>> graph = new HashMap<>();
        for (int[] edge : edges) {
            graph.putIfAbsent(edge[1], new HashSet<>());
            graph.putIfAbsent(edge[0], new HashSet<>());
            graph.get(edge[1]).add(edge[0]);
        }
        // loop the keySet of the map, to find the nodes 
        //who has less or equal to 1
        // parent.
        for (int key : graph.keySet()) {
            if (graph.get(key).size() <= 1) {
                result.add(key);
            }
        }
        return result;
    }

 public static boolean hasCommonAncestor(int[][] edges, int x, int y) {
        if (edges == null || edges.length == 1)
            return false;
        Map<Integer, HashSet<Integer>> graph = new HashMap<>();
        for (int[] edge : edges) {
            graph.putIfAbsent(edge[1], new HashSet<>());
            graph.putIfAbsent(edge[0], new HashSet<>());
            graph.get(edge[1]).add(edge[0]);
        }
        HashSet<Integer> parent1 = new HashSet<>();
        HashSet<Integer> parent2 = new HashSet<>();
        // find all of the parents of given two nodes
        // loop one of the set to find whether there is an overlap
        findParents(x, parent1, graph);
        findParents(y, parent2, graph);
        for (int parent : parent1) {
            if (parent2.contains(parent))
                return true;
        }
        return false;
    }

    // Add the parents of a given node into a set
    public static void findParents(int cur, HashSet<Integer> parents, 
         Map<Integer, HashSet<Integer>> graph) {
        for (int parent : graph.get(cur)) {
            if (parents.add(parent)) {
                findParents(parent, parents, graph);
            }
        }
    }
    
       public static int findFarAncestor(int[][] edges, int x) {
        if (edges == null || edges.length == 0)
            return 0;
        Map<Integer, HashSet<Integer>> graph = new HashMap<>();
        for (int[] edge : edges) {
            if (!graph.containsKey(edge[1])) {
                graph.put(edge[1], new HashSet<>());
            }
            if (!graph.containsKey(edge[0])) {
                graph.put(edge[0], new HashSet<>());
            }
            graph.get(edge[1]).add(edge[0]);
        }
        // max[0] is used to keep the maximum level so far;
        int[] max = new int[] { Integer.MIN_VALUE };
        // result[0] is used to keep the farest parent
        int[] result = new int[] { 0 };
        dfs(x, 0, max, result, graph);
        return result[0];
    }

    // dfs to find a parent with max levels
    private static void dfs(int cur, int level, int[] max, int[] result,
                   Map<Integer, HashSet<Integer>> graph) {
        if (graph.get(cur).size() == 0) {
            if (level > max[0]) {
                max[0] = level;
                result[0] = cur;
            }
        } else {
            for (int parent : graph.get(cur)) {
                dfs(parent, level + 1, max, result, graph);
            }
        }
    }

```

## Entering gate without badge

```python
function invalidBadgeRecords(records) {
  if (!records || records.length === 0) {
    return [];
  }
  const result = [[], []];
  // 0 for exited, 1 for entered
  const state = new Map();
  const invalidEnter = new Set();
  const invalidExit = new Set();
  for (const [name, action] of records) {
    !state.has(name) && state.set(name, 0);
    if (action === 'enter') {
      if (state.get(name) === 0) {
        state.set(name, 1);
      } else {
        invalidEnter.add(name);
      }
    } else {
      if (state.get(name) === 1) {
        state.set(name, 0);
      } else {
        invalidExit.add(name);
      }
    }
  }
  for (const [name, s] of state) {
    if (s === 1) {
      invalidEnter.add(name);
    }
  }
  for (const name of invalidEnter) {
    result[0].push(name);
  }
  for (const name of invalidExit) {
    result[1].push(name);
  }
  return result;
}

//question 2
function frequentAccess(records) {
  if (!records || records.length === 0) {
    return [];
  }
  const result = [];
  const times = new Map();
  for (const [name, timestamp] of records) {
    if (times.has(name)) {
      times.get(name).push(timestamp);
    } else {
      times.set(name, [timestamp]);
    }
  }
  for (const [name, timestamps] of times) {
    timestamps.sort(timeDifference);
    let i = 0;
    let timewindow = [timestamps[i]];
    for (let j = 1; j < timestamps.length; j++) {
      if (timeDifference(timestamps[i], timestamps[j]) < 60) {
        timewindow.push(timestamps[j]);
      } else {
        timewindow = [timestamps[j]];
        i = j;
      }
    }
    if (timewindow.length >= 3) {
      result.push([name, timewindow]);
    }
  }
  return result;
}

function timeDifference(a, b) {
  const aHour = Math.floor(a / 100);
  const bHour = Math.floor(b / 100);
  const aMinute = a % 100;
  const bMinute = b % 100;
  return aHour * 60 + aMinute - (bHour * 60 + bMinute);
}
```

## meeting room sparse time

```python
    public static boolean canSchedule(int[][] meetings, int start, int end) {
        for (int[] meeting : meetings) {
            if (start < meeting[1] && end > meeting[0]) {
                return false;
            }
        }
        return true;
    }
    
    public static List<int[]> spareTime(int[][] meetings) {
        Arrays.sort(meetings, ((a, b) -> a[0] - b[0]));
        List<int[]> result = mergeMeeting(meetings);
        List<int[]> output = new ArrayList<>();
        int start = 0;
        for (int i = 0; i < result.size(); i++) {
            if (result.get(i)[0] > start) {
                output.add(new int[] { start, result.get(i)[0] });
            }
            start = result.get(i)[1];
        }
        if (start < 2400) {
            output.add(new int[] { start, 2400 });
        }
        return output;
    }

    private static List<int[]> mergeMeeting(int[][] meetings) {

        List<int[]> result = new ArrayList<>();
        int[] curMeeting = meetings[0];
        for (int[] meeting : meetings) {
            if (curMeeting[1] >= meeting[0]) {
                curMeeting[1] = Math.max(curMeeting[1], meeting[1]);
            } else {
                result.add(curMeeting);
                curMeeting = meeting;
            }
        }
        result.add(curMeeting);
        return result;
    }
    
```

## Sparse vector

```python
class IndexOutOfRangeError extends Error {
  constructor(message) {
    super(message);
  }
}

function Node(val, next, index) {
  this.val = val;
  this.next = next;
  this.index = index;
}

function SparseVector(n) {
  this.length = n;
  this.head = null;
}

SparseVector.prototype.set = function SparseVectorSet(index, val) {
  if (index >= this.length) {
    throw new IndexOutOfRangeError(
      `Index out of range: ${index} of ${this.length}`
    );
  }
  let curr = this.head;
  if (!curr) {
    const node = new Node(val, null, index);
    this.head = node;
    return;
  }
  let prev = new Node();
  prev.next = curr;
  while (curr && curr.index < index) {
    prev = curr;
    curr = curr.next;
  }
  if (curr) {
    if (curr.index === index) {
      curr.val = val;
    } else {
      const node = new Node(val, curr, index);
      prev.next = node;
    }
  } else {
    prev.next = new Node(val, null, index);
  }
};

SparseVector.prototype.get = function SparseVectorGet(index) {
  if (index >= this.length) {
    throw new IndexOutOfRangeError(
      `Index out of range: ${index} of ${this.length}`
    );
  }
  let curr = this.head;
  while (curr && curr.index !== index) {
    curr = curr.next;
  }
  return curr ? curr.val : 0;
};

SparseVector.prototype.toString = function SparseVectorToString() {
  const result = [];
  let curr = this.head;
  for (let i = 0; i < this.length; i++) {
    if (!curr || i < curr.index) {
      result.push(0);
    } else if (i === curr.index) {
      result.push(curr.val);
    } else {
      curr = curr.next;
      i--;
    }
  }
  return '[' + result.toString() + ']';
};

//add doc cos

SparseVector.prototype.add = function SparseVectorAdd(v2) {
  if (this.length !== v2.length) {
    throw new Error('length mismatch');
  }
  const result = [];
  for (let i = 0; i < this.length; i++) {
    result.push(this.get(i) + v2.get(i));
  }
  return result;
};

SparseVector.prototype.dot = function SparseVectorDot(v2) {
  if (this.length !== v2.length) {
    throw new Error('length mismatch');
  }
  let result = 0;
  for (let i = 0; i < this.length; i++) {
    result += this.get(i) * v2.get(i);
  }
  return result;
};

SparseVector.prototype.norm = function SparseVectorNorm() {
  let sum = 0;
  for (let i = 0; i < this.length; i++) {
    const val = this.get(i);
    sum += val * val;
  }
  return Math.sqrt(sum);
};

SparseVector.prototype.cos = function SparseVectorCos(v2) {
  return this.dot(b) / (this.norm() * v2.norm());
};

class SparseVector {
     Map<Integer, Integer> indexMap = new HashMap<>();
     int n;
    // int sizeZ;
     SparseVector(int[] nums) {
         n = nums.length;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0){
             indexMap.put(i, nums[i]);
         //    sizeZ ++;
            }
                
        }
    }  
   
    // Return the dotProduct of two sparse vectors
    public int dotProduct(SparseVector vec) {
        if (indexMap.size() == 0 || vec.indexMap.size() == 0) return 0;
        if (indexMap.size() > vec.indexMap.size())
            return vec.dotProduct(this);
        int productSum = 0;
        for (Map.Entry<Integer, Integer> entry : indexMap.entrySet()) {
            int index = entry.getKey();
            if(vec.indexMap.containsKey(index)){
                productSum += (entry.getValue() * vec.indexMap.get(index));
            }
        }
        return productSum;
    }
}

```

## Find treasure

```python
    public static List<int[]> findLegalMoves(int i, int j, int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return new ArrayList<>();
        }
        List<int[]> result = new ArrayList<>();
    int[][] directions = new int[][] { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
        for (int[] direction : directions) {
            int x = i + direction[0];
            int y = i + direction[1];
            if (x >= 0 && x < grid.length && y >= 0 &&
             y < grid[0].length && grid[x][y] == 0) {
                result.add(new int[] { x, y });
            }
        }
        return result;
    }
//P2
  public static boolean FindLegalMoves(int[][] grid, int row, int col) {
        boolean[][] visited = new boolean[grid.length][grid[0].length];
        dfs(grid, row, col, visited);
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == 0 && visited[i][j] == false)
                    return false;
            }
        }
        return true;
    }

    public static void dfs(int[][] grid, int i, int j, boolean[][] visited) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid.length || 
              grid[i][j] == -1 || visited[i][j] == true)
            return;
        visited[i][j] = true;
        dfs(grid, i - 1, j, visited);
        dfs(grid, i + 1, j, visited);
        dfs(grid, i, j - 1, visited);
        dfs(grid, i, j + 1, visited);
    }
    //P3
    public static List<List<int[]>> findShortestPaths(int[][] grid, 
                        int[] start, int[] end) {
        if (grid == null || grid.length == 0 || grid[0].length == 0)
            return new ArrayList<>();
        List<List<int[]>> temp = new ArrayList<>();
    int[][] directions = new int[][] { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
        int treasure = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1)
                    treasure += 1;
            }
        }
        backtrack(grid, start[0], start[1], end, treasure, new LinkedList<>(), 
                            directions, temp);
        List<List<int[]>> result = new ArrayList<>();
        int min = Integer.MAX_VALUE;
        for (List<int[]> path : temp) {
            if (path.size() < min) {
                result = new ArrayList<>();
                result.add(path);
                min = path.size();
            } else if (path.size() == min) {
                result.add(path);
            }
        }
        return result;
    }

    private static void backtrack(int[][] grid, int row, int col, 
                 int[] end, int treasure, LinkedList<int[]> path,
            int[][] directions, List<List<int[]>> result) {
        if (row < 0 || row >= grid.length || col < 0 || col >= grid[0].length 
            || grid[row][col] == -1
                || grid[row][col] == 2) {
            return;
        }

        int temp = grid[row][col];
        grid[row][col] = 2;
        if (temp == 1)
            treasure--;
        path.offer(new int[] { row, col });
        if (row == end[0] && col == end[1] && treasure == 0) {
            result.add(new ArrayList<>(path));
            grid[row][col] = temp;
            path.removeLast();
        } else {
            for (int[] direction : directions) {
                int x = row + direction[0];
                int y = col + direction[1];
                backtrack(grid, x, y, end, treasure, path, directions, result);
            }
            grid[row][col] = temp;
            path.removeLast();

        }
    }
```

##

##

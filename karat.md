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

3: Ads Conversion Rate

```python
function adsConversionRate(completedPurchaseUserIds, adClicks, allUserIps) {
  const userIds = new Set(completedPurchaseUserIds);
  const conversion = new Map();
  const ipToUserId = new Map();
  for (const userIp of allUserIps) {
    const [userId, ip] = userIp.split(',');
    ipToUserId.set(ip, userId);
  }
  for (const click of adClicks) {
    const [ip,, adText] = click.split(',');
    if (conversion.has(adText)) {
      conversion.get(adText)[1]++;
      if (userIds.has(ipToUserId.get(ip))) {
        conversion.get(adText)[0]++;
      }
    } else {
      const bought = userIds.has(ipToUserId.get(ip)) ? 1 : 0;
      conversion.set(adText, [bought, 1]);
    }
  }
  for (const [adText, ratio] of conversion) {
    console.log(`${ratio[0]} of ${ratio[1]}  ${adText}`);
  }
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

function findAllMidway(prereqs) {
  const graph = formGraph(prereqs);
  const paths = [];
  const backtracking = (path, curr) => {
    if (graph.get(curr).length === 0) {
      paths.push([...path]);
      return;
    }
    for (const next of graph.get(curr)) {
      backtracking(path, next);
    }
    path.pop(curr);
  };
  const firstCourses = findAllFirstCourses(prereqs);
  for (const course of firstCourses) {
    backtracking([], course);
  }
  const result = new Set();
  for (const path of paths) {
    if (path.length % 2 === 0) {
      result.add(path[path.length / 2 - 1]);
    } else {
      result.add(path[Math.floor(path.length / 2)]);
    }
  }
  return result;
}

function findAllFirstCourses(prereqs) {
  const result = [];
  const courses = new Set();
  const coursesHavePrereq = new Set();
  for (const prereq of prereqs) {
    courses.add(prereq[0]);
    courses.add(prereq[1]);
    coursesHavePrereq.add(prereq[1]);
  }
  for (const course of courses) {
    if (!coursesHavePrereq.has(course)) {
      result.push(course);
    }
  }
  return result;
}

function formGraph(prereqs) {
  const result = new Map();
  for (const prereq of prereqs) {
    if (result.has(prereq[0])) {
      result.get(prereq[0]).add(prereq[1]);
    } else {
      result.set(prereq[0], new Set([prereq[1]]));
    }
  }
  for (const prereq of prereqs) {
    if (!result.has(prereq[1])) {
      result.set(prereq[1], new Set());
    }
  }
  return result;
}

console.log(findAllMidway(allCourses));

function findMidway(prereqs) {
  const result = [];
  const coursesHavePrereq = new Set();
  for (const prereq of prereqs) {
    coursesHavePrereq.add(prereq[1]);
  }
  let curr;
  for (const prereq of prereqs) {
    if (!coursesHavePrereq.has(prereq[0])) {
      curr = prereq[0];
      break;
    }
  }
  const courses = [...coursesHavePrereq, curr];
  while (result.length !== courses.length) {
    result.push(curr);
    for (const prereq of prereqs) {
      if (curr === prereq[0]) {
        curr = prereq[1];
        break;
      }
    }
  }
  if (result.length % 2 === 0) {
    return result[result.length / 2 - 1];
  }
  return result[Math.floor(result.length / 2)];
}

const studentCoursePairs1 = [
  ['58', 'Linear Algebra'],
  ['94', 'Art History'],
  ['94', 'Operating Systems'],
  ['17', 'Software Design'],
  ['58', 'Mechanics'],
  ['58', 'Economics'],
  ['17', 'Linear Algebra'],
  ['17', 'Political Science'],
  ['94', 'Economics'],
  ['25', 'Economics'],
  ['58', 'Software Design'],
];

const studentCoursePairs2 = [
  ['42', 'Software Design'],
  ['0', 'Advanced Mechanics'],
  ['9', 'Art History'],
];

function findAllOverlaps(studentCoursePairs) {
  const map = new Map();
  for (const [studentId, course] of studentCoursePairs) {
    if (map.has(studentId)) {
      map.get(studentId).push(course);
    } else {
      map.set(studentId, [course]);
    }
  }
  const studentIds = [...map.keys()];
  const result = new Map();
  for (let i = 0; i < studentIds.length; i++) {
    for (let j = i + 1; j < studentIds.length; j++) {
      const key = `${studentIds[i]},${studentIds[j]}`;
      const overlaps = [];
      for (const course1 of map.get(studentIds[i])) {
        for (const course2 of map.get(studentIds[j])) {
          if (course1 === course2) {
            overlaps.push(course1);
          }
        }
      }
      result.set(key, overlaps);
    }
  }
  return result;
}

// console.log(findAllOverlaps(studentCoursePairs1));
// console.log(findAllOverlaps(studentCoursePairs2));
```

```python
I used BFS to find ancestor of each node and compare if there's any common ones.
   I just modified the code in 2 to return the last number in the queue from BFS as 
  the earliest ancestor.
  
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

## Find rectangle

```python
function findOneRectangle(board) {
  if (!board || board.length === 0 || board[0].length === 0) {
    return [];
  }
  const result = [];
  for (let i = 0; i < board.length; i++) {
    for (let j = 0; j < board[0].length; j++) {
      if (board[i][j] === 0) {
        result.push([i, j]);
        let height = 1, width = 1;
        while (i + height < board.length && board[i + height][j] === 0) {
          height++;
        }
        while (j + width < board[0].length && board[i][j + width] === 0) {
          width++;
        }
        result.push([i + height - 1, j + width - 1]);
        break;
      }
      if (result.length !== 0) {
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
function basicCalculator(expression) {
  if (!expression || expression.length === 0) {
    return 0;
  }
  let result = 0;
  let sign = 1;
  for (let i = 0; i < expression.length; i++) {
    if (isNumeric(expression[i])) {
      let num = expression[i];
      while (i + 1 < expression.length && isNumeric(expression[i + 1])) {
        num += expression[++i];
      }
      num = parseInt(num);
      result += num * sign;
    } else if (expression[i] === '+') {
      sign = 1;
    } else if (expression[i] === '-') {
      sign = -1;
    }
  }
  return result;
}

function isNumeric(c) {
  return c >= '0' && c <= '9';
}

function basicCalculator(expression) {
  if (!expression || expression.length === 0) {
    return 0;
  }
  let result = 0;
  let sign = 1;
  let stack = [];
  for (let i = 0; i < expression.length; i++) {
    if (isNumeric(expression[i])) {
      let num = expression[i];
      while (i + 1 < expression.length && isNumeric(expression[i + 1])) {
        num += expression[++i];
      }
      num = parseInt(num);
      result += num * sign;
    } else if (expression[i] === '+') {
      sign = 1;
    } else if (expression[i] === '-') {
      sign = -1;
    } else if (expression[i] === '(') {
      stack.push(result);
      stack.push(sign);
      result = 0;
      sign = 1;
    } else if (expression[i] === ')') {
      result = result * stack.pop() + stack.pop();
    }
  }
  return result;
}

function isNumeric(c) {
  return c >= '0' && c <= '9';
}
```

## Word wrap & word processor

```python
function wordWrap(words, maxLen) {
  if (!words || words.length === 0) {
    return [];
  }
  const result = [];
  let i = 0;
  while (i < words.length) {
    let remain = maxLen;
    let count = 0;
    while (i < words.length) {
      if (remain - words[i].length < 0) {
        break;
      }
      count++;
      remain -= words[i++].length + 1;
    }
    result.push(words.slice(i - count, i).join('-'));
  }
  return result;
}

function reflowAndJustify(lines, maxLen) {
  if (!lines || lines.length === 0) {
    return [];
  }
  const words = lines.join(' ').split(' ');
  const result = [];
  let i = 0;
  while (i < words.length) {
    // split words into lines first
    let remain = maxLen;
    let count = 0;
    while (i < words.length) {
      if (remain - words[i].length < 0) {
        break;
      }
      count++;
      remain -= words[i++].length + 1;
    }
    const line = words.slice(i - count, i);

    // after splitting into lines, calculate the required dashes 
    between each word
    const n = line.reduce((n, word) => n + word.length, 0);
    let reflowed = ''; // line result with padded dashes
    const baseDash = '-'.repeat(parseInt((maxLen - n) / (line.length - 1)));
    let extra = (maxLen - n) % (line.length - 1); //
    // some dashes at the beginning has one extra dash
    for (let j = 0; j < line.length; j++) {
      if (j === line.length - 1) {
        reflowed += line[j];
      } else {
        reflowed +=
          extra-- <= 0 ? line[j] + baseDash : line[j] + baseDash + '-';
      }
    }
    result.push(reflowed);
  }
  return result;
}
```

## Is valid matrix

```python
function isValidMatrix(matrix) {
  if (!matrix || matrix.length === 0 || matrix[0].length === 0) {
    return false;
  }
  const n = matrix.length;
  for (let i = 0; i < n; i++) {
    const rowSet = new Set();
    const colSet = new Set();
    let rowMin = Number.POSITIVE_INFINITY, rowMax = Number.NEGATIVE_INFINITY;
    let colMin = rowMin, colMax = rowMax;
    for (let j = 0; j < n; j++) {
      if (!rowSet.has(matrix[i][j])) {
        rowSet.add(matrix[i][j]);
        rowMin = Math.min(rowMin, matrix[i][j]);
        rowMax = Math.max(rowMax, matrix[i][j]);
      } else {
        return false;
      }
      if (!colSet.has(matrix[j][i])) {
        colSet.add(matrix[j][i]);
        colMin = Math.min(colMin, matrix[j][i]);
        colMax = Math.max(colMax, matrix[j][i]);
      } else {
        return false;
      }
    }
    if (rowMin !== 1 || colMin !== 1 || rowMax !== n || colMax !== n) {
      return false;
    }
  }
  return true;
}
```


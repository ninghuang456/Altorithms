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

function isValidNonogram(matrix, rows, cols) {
  if (!matrix || !rows || !cols) {
    return false;
  }
  const n = matrix.length;
  const m = matrix[0].length;
  if (n === 0 || n !== rows.length || m !== cols.length) {
    return false;
  }
  return (
    isNonogramRowsValid(matrix, rows, n, m) &&
    isNonogramColsValid(matrix, cols, n, m)
  );
}

function isNonogramRowsValid(matrix, rows, n, m) {
  for (let i = 0; i < n; i++) {
    let rowIndex = 0;
    for (let j = 0; j < m; j++) {
      if (matrix[i][j] === 0) {
        if (rows[i].length === 0) {
          return false;
        }
        for (let k = 0; k < rows[rowIndex]; k++) {
          if (j + k >= m || matrix[i][j + k] !== 0) {
            return false;
          }
        }
        j += rows[i][rowIndex++];
      }
    }
    if (rowIndex !== rows[i].length) {
      return false;
    }
  }
  return true;
}

function isNonogramColsValid(matrix, cols, n, m) {
  for (let i = 0; i < m; i++) {
    let colIndex = 0;
    for (let j = 0; j < n; j++) {
      if (matrix[j][i] === 0) {
        if (cols[i].length === 0) {
          return false;
        }
        for (let k = 0; k < cols[colIndex]; k++) {
          if (j + k >= n || matrix[j + k][i] !== 0) {
            return false;
          }
        }
        j += cols[i][colIndex++];
      }
    }
    if (colIndex !== cols[i].length) {
      return false;
    }
  }
  return true;
}
```

## Node ancestor

```python
function findNodesWithZeroOrOneParent(edges) {
  if (!edges || edges.length === 0) {
    return [];
  }
  const result = [];
  const map = new Map();
  for (const [parent, child] of edges) {
    if (map.has(child)) {
      map.get(child).add(parent);
    } else {
      map.set(child, new Set([parent]));
    }
  }
  for (const [child, parentSet] of map) {
    if (parentSet.length === 0 || parentSet.length === 1) {
      result.push(child);
    }
  }
  return result;
}

function hasCommonAncestor(edges, x, y) {
  if (!edges || edges.length === 0) {
    return false;
  }
  const directParents = new Map();
  for (const [parent, child] of edges) {
    if (directParents.has(child)) {
      directParents.get(child).add(parent);
    } else {
      directParents.set(child, new Set([parent]));
    }
  }
  const findAllParents = (e) => {
    const result = new Set();
    const stack = [];
    stack.push(e);
    while (stack.length !== 0) {
      const curr = stack.pop();
      const parents = directParents.get(curr);
      if (!parents) {
        continue;
      }
      for (const parent of parents) {
        if (result.has(parent)) {
          continue;
        }
        result.add(parent);
        stack.push(parent);
      }
    }
    return result;
  };
  const parentsOfX = findAllParents(x);
  const parentsOfY = findAllParents(y);
  for (const parentOfX of parentsOfX) {
    if (parentsOfY.has(parentOfX)) {
      return true;
    }
  }
  return false;
}

// earliestAncestor
function Queue() {
  this.firstStack = [];
  this.secondStack = []; 
  this.length = 0;
}

Queue.prototype.push = function(x) {
  this.firstStack.push(x);
  this.length++;
};

Queue.prototype.pop = function() {
  if (this.secondStack.length === 0) {
    while (this.firstStack.length !== 0) {
      this.secondStack.push(this.firstStack.pop());
    }
  }
  this.length--;
  return this.secondStack.pop();
};

function earliestAncestor(parentChildPairs, x) {
  const directParents = new Map();
  for (const [parent, child] of parentChildPairs) {
    if (directParents.has(child)) {
      directParents.get(child).add(parent);
    } else {
      directParents.set(child, new Set([parent]));
    }
  }
  let currlayer = new Queue();
  let prevlayer;
  const visited = new Set();
  currlayer.push(x);
  while (currlayer.length !== 0) {
    prevlayer = new Queue();
    let size = currlayer.length;
    while (size--) {
      const curr = currlayer.pop();
      prevlayer.push(curr);
      const parents = directParents.get(curr);
      if (!parents) {
        continue;
      }
      for (const parent of parents) {
        if (visited.has(parent)) {
          continue;
        }    
        currlayer.push(parent);
        visited.add(parent);
      }
    }
  }
  return prevlayer.pop();
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
function canSchedule(meetings, start, end) {
  for (const meeting of meetings) {
    if (
      (start >= meeting[0] && start < meeting[1]) ||
      (end > meeting[0] && end <= meeting[1]) ||
      (start < meeting[0] && end > meeting[1])
    ) {
      return false;
    }
  }
  return true;
}


function spareTime(meetings) {
  if (!meetings || meetings.length === 0) {
    return [];
  }
  meetings = mergeMeetings(meetings);
  const result = [];
  let start = 0;
  for (let i = 0; i < meetings.length; i++) {
    result.push([start, meetings[i][0]]);
    start = meetings[i][1];
  }
  return result;
}

function mergeMeetings(meetings) {
  const result = [];
  meetings.sort((a, b) => a[0] - b[0]);
  let [start, end] = meetings[0];
  for (const meeting of meetings) {
    if (start < meeting[1]) {
      end = Math.max(end, meeting[1]);
    } else {
      result.push(start, end);
      start = meeting[0];
      end = meeting[0];
    }
  }
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
class Solution {
    int[][] dis = new int[][]{{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) return 0;
        int total = 0;
        for (int r = 0; r < grid.length; r ++) {
            for (int c = 0; c < grid[0].length; c++) {
                if (grid[r][c] == '1') {
                    searchArea(grid, r , c);
                    total ++;
                }
            }
        }
        return total;
    }
    
    private void searchArea(char[][] grid, int r, int c){
        if(!inArea(grid,r,c)){
            return;
        }
        if(grid[r][c] != '1'){
            return;
        }
        grid[r][c] = '2';
        for (int i = 0; i < 4; i ++){
            int nextR = r + dis[i][0];
            int nextC = c + dis[i][1];
            searchArea(grid, nextR, nextC);
        }
            
        
    }
    
    private boolean inArea(char[][] grid, int r, int c){
        return r >= 0 && r < grid.length && c >= 0 && c < grid[0].length;
    }
}

function findLegalMoves(matrix, i, j) {
  if (!matrix || matrix.length === 0) {
    return false;
  }
  const visited = Array.from({ length: matrix.length }, () =>
    Array.from({ length: matrix[0].length }, () => false)
  );
  const floodFillDFS = (x, y) => {
    if (x < 0 || x >= matrix.length || y < 0 || y >= matrix[0].length || matrix[x][y] === -1 || visited[x][y]) {
      return;
    } 
    visited[x][y] = true;
    floodFillDFS(x - 1, y);
    floodFillDFS(x + 1, y);
    floodFillDFS(x, y - 1);
    floodFillDFS(x, y + 1);
  };
  floodFillDFS(i, j);
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[0].length; j++) {
      if (!visited[i][j] && matrix[i][j] === 0) {
        return false;
      }
    }
  }
  return true;
}

function findAllTreasures(board, start, end) {
  if (!board) {
    return [];
  }
  let numTreasures = 0;
  for (let i = 0; i < board.length; i++) {
    for (let j = 0; j < board[0].length; j++) {
      if (board[i][j] === 1) {
        numTreasures++;
      }
    }
  }
  const paths = [];
  const dfs = (x, y, path, remainTreasure) => {
    if (
      x < 0 ||
      x >= board.length ||
      y < 0 ||
      y >= board[0].length ||
      board[x][y] === -1 ||
      board[x][y] === 2
    ) {
      return;
    }
    path.push([x, y]);
    const temp = board[x][y];
    if (temp === 1) {
      remainTreasure--;
    }
    if (x === end[0] && y === end[1] && remainTreasure === 0) {
      paths.push([...path]);
      path.pop();
      board[x][y] = temp;
      return;
    }
    board[x][y] = 2;
    dfs(x + 1, y, path, remainTreasure);
    dfs(x - 1, y, path, remainTreasure);
    dfs(x, y + 1, path, remainTreasure);
    dfs(x, y - 1, path, remainTreasure);
    board[x][y] = temp;
    path.pop();
  };
  dfs(start[0], start[1], [], numTreasures);
  if (paths.length === 0) {
    return [];
  }
  let minPaths = paths[0].length;
  for (let i = 0; i < paths.length; i++) {
    minPaths = Math.min(minPaths, paths[i].length);
  }
  return paths.filter((path) => path.length === minPaths);
}
```

## 

## 


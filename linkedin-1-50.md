# Linkedin 1 \~ 50

## 244 Shortest Word Distance II

```
Input
["WordDistance", "shortest", "shortest"]
[[["practice", "makes", "perfect", "coding", "makes"]], ["coding", "practice"], ["makes", "coding"]]
Output
[null, 3, 1]

Explanation
WordDistance wordDistance = new WordDistance(["practice", "makes", "perfect", "coding", "makes"]);
wordDistance.shortest("coding", "practice"); // return 3
wordDistance.shortest("makes", "coding");    // return 1

class WordDistance {
    
    private Map<String, List<Integer>> map;

    public WordDistance(String[] words) {
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
        // list 1 and list 2 already sorted!
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

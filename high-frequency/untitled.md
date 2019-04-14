# String 1

1.Unique Email Addresses

Every email consists of a local name and a domain name, separated by the @ sign.

For example, in `alice@leetcode.com`, `alice` is the local name, and `leetcode.com` is the domain name.

Besides lowercase letters, these emails may contain `'.'`s or `'+'`s.

If you add periods \(`'.'`\) between some characters in the **local name** part of an email address, mail sent there will be forwarded to the same address without dots in the local name.  For example, `"alice.z@leetcode.com"` and `"alicez@leetcode.com"` forward to the same email address.  \(Note that this rule does not apply for domain names.\)

If you add a plus \(`'+'`\) in the **local name**, everything after the first plus sign will be **ignored**. This allows certain emails to be filtered, for example `m.y+name@email.com` will be forwarded to `my@email.com`.  \(Again, this rule does not apply for domain names.\)

It is possible to use both of these rules at the same time.

Given a list of `emails`, we send one email to each address in the list.  How many different addresses actually receive mails? 

```text
class Solution {
    public int numUniqueEmails(String[] emails) {
        if (emails == null || emails.length == 0) return 0;
        Set<String> emailSet = new HashSet<>();
        for (String email : emails){
            String[] parts = email.split("@");
            String[] first = parts[0].split("\\+");//斜杠方向
            String firstHalf = first[0].replace(".","");
            String total = firstHalf + "@" + parts[1];
            emailSet.add(total);
        }
        return emailSet.size();    
    }
}
```

2. Longest Palindromic Substring

Given a string **s**, find the longest palindromic substring in **s**. You may assume that the maximum length of **s** is 1000.

**Example 1:**

```text
Input: "babad"
Output: "bab"
Note: "aba" is also a valid answer.
```

**Example 2:**

```text
Input: "cbbd"
Output: "bb"
```

```text
class Solution {
     private int lo, maxLen;

    public String longestPalindrome(String s) {
        int len = s.length();
        if (len < 2)
            return s;

        for (int i = 0; i < len-1; i++) {
            extendPalindrome(s, i, i);  //assume odd length, try to extend Palindrome as possible
            extendPalindrome(s, i, i+1); //assume even length.
        }
        return s.substring(lo, lo + maxLen);
    }

    private void extendPalindrome(String s, int left, int right) {//字符串和两个指针
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        //注意上面while循环是在左右不等才停止的，当前的maxLen是成员变量，需要维持奇偶中大的一个（较小的不进循环）
        if (maxLen < right - left - 1) {
            lo = left + 1;//回文子字符串的下标
            maxLen = right - left - 1;//回文子字符串的上标
        }
    }
}
```

3.Longest Substring Without Repeating Characters

Given a string, find the length of the **longest substring** without repeating characters.

**Example 1:**

```text
Input: "abcabcbb"
Output: 3 
Explanation: The answer is "abc", with the length of 3. 
```

**Example 2:**

```text
Input: "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
```

**Example 3:**

```text
Input: "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3. 
             Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
```

```text
class Solution {
    public int lengthOfLongestSubstring(String s) {
          if (s.length() == 0) {
            return 0;
        }
        HashMap<Character, Integer> map = new HashMap<>();
        int result = 0;
        for (int i = 0, j = 0; i < s.length(); i++) {//右边索引遍历字符串,左边记录窗口左边
            if (map.containsKey(s.charAt(i))) {//如果滑动窗口出现重复的字符
                j = Math.max(j, map.get(s.charAt(i)) + 1); // j 只能前行 如果不对比 会在abba的时候j 从2 变回1；
            }
            map.put(s.charAt(i), i);//不管是否移动左边的索引，都将当前的字符存入hashmap
            result = Math.max(result, i - j + 1);
        }
        return result;
        
    }
}
```


# 1~10 classic

```text
1. Two Sum
class Solution {
    public int[] twoSum(int[] nums, int target) {
        if (nums == null || nums.length < 2) return new int[0];
        HashMap<Integer,Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
             if (map.containsKey(target - nums[i])){
                 int index = map.get(target - nums[i]);
                 return new int[]{index, i};   
             } else {
                 map.put(nums[i], i);
             }
        }
        return new int[0];
    }
}

2. Maximum Subarray
class Solution {
    public int maxSubArray(int[] a) {
    int maxSum = 0, thisSum = 0, max=a[0];
    for(int i=0; i<a.length; i++) {
        if(a[i]>max) max =a[i];
        thisSum += a[i];
        if(thisSum > maxSum)
            maxSum = thisSum;
        else if(thisSum < 0)
            thisSum = 0;
    }
    if (maxSum==0) return max;
    return maxSum;
        
    }
}

3. Best Time to Buy and Sell Stock
buy one and sell one share of the stock),
design an algorithm to find the maximum profit.

class Solution {
    public int maxProfit(int[] prices) {
         if (prices.length == 0) {
			 return 0 ;
		 }		
		 int max = 0 ;
		 int sofarMin = prices[0] ;
	     for (int i = 0 ; i < prices.length ; ++i) {
	    	 if (prices[i] > sofarMin) {
	    		 max = Math.max(max, prices[i] - sofarMin) ;
	    	 } else{
	    		sofarMin = prices[i];  
	    	 }
	     }	     
	    return  max ;
        
    }
}

4. Merge Sorted Array
public class Solution {
    public void merge(int A[], int m, int B[], int n) {
        int i = m - 1, j = n - 1, k = m + n - 1;
        while(i >= 0 && j >= 0) {
            A[k--] = A[i] > B[j] ? A[i--] : B[j--];
        }
        while(j >= 0) {
            A[k--] = B[j--];
        }
    }
}

5. Move Zeroes
class Solution {
    public void moveZeroes(int[] nums) {
        if (nums == null || nums.length == 0) return ;
        int nonZeroIndex = 0;
        for(int num : nums) {
            if (num != 0) {
                nums[nonZeroIndex] = num;
                nonZeroIndex ++;
            }
        } 
        
        while (nonZeroIndex < nums.length){
             nums[nonZeroIndex] = 0;
             nonZeroIndex ++;
        }
    }
}
6. Valid Parentheses
class Solution {
    public boolean isValid(String s) {
           Stack<Character> stack = new Stack<Character>();
        // Iterate through string until empty
        for(int i = 0; i<s.length(); i++) {
            // Push any open parentheses onto stack
            if(s.charAt(i) == '(' || s.charAt(i) == '[' || s.charAt(i) == '{')
                stack.push(s.charAt(i));
            // Check stack for corresponding closing parentheses, false if not valid
            else if(s.charAt(i) == ')' && !stack.empty() && stack.peek() == '(')
                stack.pop();
            else if(s.charAt(i) == ']' && !stack.empty() && stack.peek() == '[')
                stack.pop();
            else if(s.charAt(i) == '}' && !stack.empty() && stack.peek() == '{')
                stack.pop();
            else
                return false;
        }
        // return true if no open parentheses left in stack
        return stack.empty();
         
        
    }
}

7. Unique Email Addresses
class Solution {
    public int numUniqueEmails(String[] emails) {
        if (emails == null || emails.length == 0) return 0;
        Set<String> emailSet = new HashSet<>();
        for(String email : emails) {
            String[] parts = email.split("@");
            String[] first = parts[0].split("\\+"); 
            String firstHalf = first[0].replace(".","");
            emailSet.add(firstHalf + "@" + parts[1]);
        }
       return emailSet.size();
    }
}

8. Reverse String
class Solution {
    public String reverseString(String s) {
         char[] word = s.toCharArray();
        int i = 0;
        int j = s.length() - 1;
        while (i < j) {
            char temp = word[i];
            word[i] = word[j];
            word[j] = temp;
            i++;
            j--;
        }
        return new String(word);
        
    }
}

9. Valid Palindrome
class Solution {
    public boolean isPalindrome(String s) {
        if (s == null || s.length() == 0) {
            return true;
            }
        char[] total = s.toCharArray();
        int left = 0;
        int right = s.length()- 1;
        while (left < right) {  
            char l = total[left];
            char r = total [right];
            if (!Character.isLetterOrDigit(l)){
               left ++;
           } else if (!Character.isLetterOrDigit(r)) {
               right --;
           }   else {
               if(Character.toLowerCase(l) == Character.toLowerCase(r)) {
                   left ++;
                   right--;
               } else {
                   return false;
               }
           } 
        }
        return true;
    }
}

10. First Unique Character in a String
class Solution {
    public int firstUniqChar(String s) {
         int freq [] = new int[26];
        for(int i = 0; i < s.length(); i ++)
            freq [s.charAt(i) - 'a'] ++;
        for(int i = 0; i < s.length(); i ++)
            if(freq [s.charAt(i) - 'a'] == 1)
                return i;
        return -1;
        
    }
}


```


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
               res.add((start==end) ? String.valueOf(start) : String.valueOf(start) + "->" + String.valueOf(end));
               start = nums[i];
               end = nums[i];
            }
          
        }
        res.add( (start==end) ? String.valueOf(start) : String.valueOf(start) + "->" + String.valueOf(end));
        return res;
    }
}


```




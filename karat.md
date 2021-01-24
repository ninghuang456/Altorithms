# Karat

1- **Subdomain Visit Count**

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

**2- Longest Common Continuous Subarray**

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

## 


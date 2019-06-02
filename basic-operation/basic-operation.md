# Basic Operation

```text
ArrayList

public class ArrayListDemo {
    public static void main(String[] srgs){
         ArrayList<Integer> arrayList = new ArrayList<Integer>();
         System.out.printf("Before add:arrayList.size() = %d\n",arrayList.size());
         arrayList.add(1);
         arrayList.add(3);
         arrayList.add(5);
         arrayList.add(7);
         arrayList.add(9);
         // 三种遍历方式打印元素
         // 第一种：通过迭代器遍历
         Iterator<Integer> it = arrayList.iterator();
         while(it.hasNext()){
             System.out.print(it.next() + " ");
         }
         System.out.println();

         // 第二种：通过索引值遍历
         for(int i = 0; i < arrayList.size(); i++){
             System.out.print(arrayList.get(i) + " ");
         }
         System.out.println();

         // 第三种：for循环遍历
         for(Integer number : arrayList){
             System.out.print(number + " ");
         }
         // toArray用法
         // 第一种方式(最常用)
         Integer[] integer = arrayList.toArray(new Integer[0]);
         // 第二种方式(容易理解)
         Integer[] integer1 = new Integer[arrayList.size()];
         arrayList.toArray(integer1);
         // 抛出异常，java不支持向下转型
         //Integer[] integer2 = new Integer[arrayList.size()];
         //integer2 = arrayList.toArray();
         System.out.println();
         // 在指定位置添加元素
         arrayList.add(2,2);
         // 删除指定位置上的元素
         arrayList.remove(2);    
         // 删除指定元素
         arrayList.remove((Object)3);
         // 判断arrayList是否包含5
         System.out.println("ArrayList contains 5 is: " + arrayList.contains(5));
         // 清空ArrayList
         arrayList.clear();
         // 判断ArrayList是否为空
         System.out.println("ArrayList is empty: " + arrayList.isEmpty());
    }
}
===================================================================================
HashMap 基本用法
创建HashMap对象
     HashMap<String,Integer> hashMap = new HashMap<>();
1.添加键值对
添加元素时，如果key已经存在，则返回旧value，并将新的value存到该key中；如果key不存在，则返回null
    hashMap.put("aa",1);
    hashMap.put("bb",2);
    hashMap.put("cc",3);
put方法会覆盖原有的value，而另一种put方法不会覆盖：putIfAbsent(key,value)
    hashMap.putIfAbsent("aa",4);
该方法首先会判断key是否存在，如果存在且value不为null，则不会覆盖原有的value，
并返回原来的value；如果key不存在或者key的value为null，则会put进新值，并返回null。
另外，两种方法当key=null时，并不会抛出异常，而是按照一个特殊的方法进行存储。

2.删除元素
如果之前对相同key多次put，则可以移除key对应的旧value，而最新的value不受影响。(×)
remove(key):删除成功(存在key)，返回被删除的key对应的value，否则返回null。
remove(key,value):删除成功（存在entry），返回true，否则返回false。
    hashMap.remove("bb");
    hashMap.remove("aa",5);
3.获取元素
对于获取元素，有get(key)和getOrDefault(key,defaultValue)（1.8之后）两种方法.
    hashMap.get("cc")
getOrDefault在key不存在时,返回一个defaultValue。在没有该方法前需要这样写：
    Integer bbValue = hashMap.containsKey("bb")?hashMap.get("bb"):-1;
有了getOrDefault可以这样写：
    getOrDefault("aa",-1)//key=aa不存在，所以返回默认value -1

4.元素遍历
        Iterator iterator = hashMap.keySet().iterator();
        while (iterator.hasNext()){
            String key = (String)iterator.next();
            System.out.println(key+"="+hashMap.get(key));
        }
第二种遍历方法：
Iterator iterator1 = hashMap.entrySet().iterator();
        while (iterator1.hasNext()){
            Map.Entry entry = (Map.Entry) iterator1.next();
            String key = (String) entry.getKey();
            Integer value = (Integer) entry.getValue();
            System.out.println(key+"="+value);
        }
5.判断key或value是否存在
    hashMap.containsKey("aa");
    hashMap.containsValue(1);
6.替换元素
replace方法用来替换元素。
    hashMap.replace("ff",5);
对于存在的key，调用replace方法，会替换原来的value，并返回旧value，
这和put的效果是一样的；对于不存在的key，replace方法什么都不做。
这就是他和put的区别（put在key不存在时将新key-value加入map）。

===============================================================================
Queue 操作：
offer 添加一个元素并返回true 如果队列已满，则返回false
poll 移除并返问队列头部的元素 如果队列为空，则返回null
peek 返回队列头部的元素 如果队列为空，则返回null
add 增加一个元索 如果队列已满，则抛出一个IIIegaISlabEepeplian异常
remove 移除并返回队列头部的元素 如果队列为空，则抛出一个NoSuchElementException异常
element 返回队列头部的元素 如果队列为空，则抛出一个NoSuchElementException异常
put 添加一个元素 如果队列满，则阻塞
take 移除并返回队列头部的元素

Stack类
栈：桶型或箱型数据类型，后进先出，相对堆Heap为二叉树类型，可以快速定位并操作
public class Stack extends Vector
Stack的方法调用的Vector的方法，被synchronized修饰，为线程安全(Vector也是)
Stack methods：
push : 把项压入堆栈顶部 ，并作为此函数的值返回该对象
pop : 移除堆栈顶部的对象，并作为此函数的值返回该对象
peek : 查看堆栈顶部的对象，，并作为此函数的值返回该对象，但不从堆栈中移除它
empty : 测试堆栈是否为空
search : 返回对象在堆栈中的位置，以 1 为基数
===============================================================================


```


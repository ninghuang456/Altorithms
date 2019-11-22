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
Deque类
适用场景:
一般场景,LinkedList,链表双端队列,允许元素为null；ArrayDeque,数组双端队列,不允许元素为null.
并发场景下,使用BlockingDeque接口下的一些实现.

    1) 是Queue的子接口，表示双端队列，即两端（队尾和队首）都能插入和删除的特殊队列；

    2) 当然，Deque可以使用Queue的全部方法，但是自己也扩展了很多方法，主要用于操作两个端口的：
特点就是Queue的每种方法都多了一个操作队尾的版本
（同时也提供了一个名称上呼应的操作队首的版本（first（队首）、last（队尾）），
和Queue的完全相对应（就是Queue的方法多了两个First和Last的后缀而已）：

         i. 插入：add/offer

异常不安全：

            a. void add(E ele);  // Queue

            b. void addFirst(E ele); // Deque，从队首加，和a.等价

            c. void addLast(E ele);  // Deque，从队尾加

异常安全：

            a. boolean offer(E ele);  // Queue

            b. boolean offerFirst | offerLast(E ele); // Deque

         ii. 取出并删除：remove/poll

异常不安全：E remove | removeFirst | removeLast();

异常安全：E poll | pollFirst | pollLast();

         iii. 查看但不删除：element（Queue）/get（Deque）
         /peek，有点儿特殊，不安全的查看没有直接沿用Queue的名称（element），而是get！

异常不安全：

            a. E element(); // Queue，查看队首

            b. E getFirst | getLast();  // Deque

异常安全：E peek | peekFirst | peekLast();

！！其中，Queue的版本和Deque的First版本完全等价，只是为了命名上和Last对称罢了！例如poll和pollFirst完全等价；

    3) 特殊的，Deque也有自己特有的一些常用方法：

         i. 找到并删除指定元素：boolean removeFirstOccurrence | removeLastOccurrence(E e);  
         // 删除队列中第一个出现的/最后一个出现的e，
如果不存在或者空或其它返回false，成功删除（即改变了队列）则返回true

         ii. Deque不仅有普通的正向迭代器dq.iterator，也有反向迭代器：Iterator<E> descendingIterator();  
         // 得到的同样是Iterator类型的迭代器，只不过是反向的

！！即起始位置在队尾，其hasNext、next是朝队首迭代的！

         

2. 用Deque实现栈：

    1) Java1.0的时候就提供了一个Stack类，是Vector的子类，用于实现栈这种数据结构（FILO），
    即一种只能在一端进出元素的数据结构；

    2) 但是和Vector一样，Stack过于古老，并且实现地非常不好，问题多多，因此现在基本已经不用了；

    3) Deque是双端队列，两端都可以进出，因此可以直接用Deque来代替栈了（只操作一端就是栈了，
    只有First方法或者只用Last方法就变成栈了）；

    4) 但是为了迎合Stack的风格，Deque还是提供了push/pop方法，让它用起来感觉是栈：
    只不过这两个方法默认的端口是队首

         i. void push(E e);  // 等价于void addFirst(E e);

         ii. E pop();  // 等价于E removeFirst();

！！可以看到提供的两个方法都是一场不安全的版本，
如果有异常安全的需求那就自行使用offer/poll(First/Last)方法



3. ArrayDeque：

    1) 顾名思义，就是用数组实现的Deque；

    2) 既然是底层是数组那肯定也可以指定其capacity，但不过由于双端队列本身数据结构的限制，
    要求只能在初始化的时候指定capacity，
因此没有像ArrayList那样随时都可以调整的ensureCapacity方法了，
只能在构造器中指定其初始的capacity：ArrayDeque(int numElements);   // 初始元素个数，默认长度是16

！！同样由于本身数据结构的限制，ArrayDeque没有trimToSize方法可以为自己瘦身！！

    3) ArrayDeque的使用方法就是上面的Deque的使用方法，基本没有对Deque拓展什么方法；



4. LinkedList：

    1) 是一种双向链表数据结构，由于双向链表的特性，它既具有线性表的特性也具有双端队列的特性；

    2) 在Java的集合中，LinkedList同时（直接）继承了List和Deque；

    3) 由于它是用链表实现的，因此不像数组那样可以指定初始长度，它只有一个空的构造器；

    4) 其余就是它可以使用List和Deque的全部方法，从功能的广度上（覆盖面）来将它是最强大的！

    5) 具体方法就不介绍了，既可以当成List使，也可以当成Deque使；



5. 各类线性表的性能比较以及选择：

    1) 性能主要是看底层是如何实现，不过就是数组和链表两种，因此性能也是围绕这两种数据结构展开的；

    2) 链表遍历、插入、删除高效（相对的数组不行），而数组随机访问、批量处理高效（相对的链表不行），
链表使用迭代器迭代高效（数组不行，数组直接随机访问遍历更快）；

```


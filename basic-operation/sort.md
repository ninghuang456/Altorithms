---
description: 冒泡 选择 插入 归并 快速 基数 计数 桶 时间复杂度是否原地是否基于交换是否稳定（同样的数据排序后位置有没有发生变化）
---

# Sort

```text
private static void swap(Object[] arr, int i, int j) {
        Object t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
    }
    
public static void selectSort(int[] arr){

        int n = arr.length;
        for( int i = 0 ; i < n ; i ++ ){
            int minIndex = i;
            for( int j = i + 1 ; j < n ; j ++ )
                if( arr[j] < arr[minIndex] )
                    minIndex = j;

            swap( arr , i , minIndex);
        }
    }
}

public static void InsertionSort(Comparable[] arr){

        int n = arr.length;
        for (int i = 0; i < n; i++) {
            for( int j = i; j > 0 && arr[j].compareTo(arr[j-1]) < 0 ; j--)
                swap(arr, j, j-1);

        }
    }


```


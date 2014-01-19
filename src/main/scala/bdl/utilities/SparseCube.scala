package utilities
import scala.collection.mutable.{HashMap, HashSet, ArrayBuffer}

class SparseCube (
    val ids1: Array[Int], val ptr1: Array[Int], val map1: Array[Int], 
    val ids2: Array[Int], val ptr2: Array[Array[Int]], val map2: Array[Int], 
    val ids3: Array[Int], val ptr3: Array[Array[Int]], val map3: Array[Int],
    val value: Array[Float]) extends Serializable {
  
  //factor matrices are N-by-K
  def getSE(
      factorMat1: Array[Array[Float]], 
      factorMat2: Array[Array[Float]],
      factorMat3: Array[Array[Float]]) = {
    val numFactors = factorMat1(0).length
    var se = 0f; var d = 0
    while (d < ptr1.length-1) {
      var i = ptr1(d)
      val factor1 = factorMat1(d)
      while (i < ptr1(d+1)) {
        val factor2 = factorMat2(ids2(i))
        val factor3 = factorMat3(ids3(i))
        var k = 0; var sum = 0f
        while (k < numFactors) {
          value(i) -= factor1(k)*factor2(k)*factor3(k)
          k += 1
        }
        se += value(i)*value(i)
        i += 1
      }
      d += 1
    }
    se
  }
}

object SparseCube {
  def apply(triplets: Array[Triplet], multicore: Boolean = false) = {
    val nnz = triplets.length
    val ids1 = new Array[Int](nnz)
    val ids2 = new Array[Int](nnz); val ids3 = new Array[Int](nnz) 
    val value = new Array[Float](nnz)
    val set1 = new HashSet[Int]; val set2 = new HashSet[Int]
    val set3 = new HashSet[Int]
    
    for (triplet <- triplets) {
      set1.add(triplet._1)
      set2.add(triplet._2)
      set3.add(triplet._3)
    }
    val size1 = set1.size
    val size2 = set2.size
    val size3 = set3.size
    val ptr1 = new Array[Int](size1+1)
    val ptr2_buf = if (multicore) Array.fill(size2)(new ArrayBuffer[Int]) else null
    val ptr3_buf = if (multicore) Array.fill(size3)(new ArrayBuffer[Int]) else null
    val array1 = set1.toArray.sorted
    val array2 = set2.toArray.sorted
    val array3 = set3.toArray.sorted
    val map1 = new HashMap[Int, Int]; val map2 = new HashMap[Int, Int]
    val map3 = new HashMap[Int, Int]
    array1.view.zipWithIndex.foreach(pair=>map1.put(pair._1, pair._2))
    array2.view.zipWithIndex.foreach(pair=>map2.put(pair._1, pair._2))
    array3.view.zipWithIndex.foreach(pair=>map3.put(pair._1, pair._2))
    
    var i = 0
    while (i < nnz) {
      val triplet = triplets(i)
      ids1(i) = map1.getOrElse(triplet._1, -1)
      ids2(i) = map2.getOrElse(triplet._2, -1)
      ids3(i) = map3.getOrElse(triplet._3, -1)
      value(i) = triplet.value
      ptr1(ids1(i)+1) += 1
      i += 1
    }
    for (d <- 1 to size1) ptr1(d) += ptr1(d-1)
    
    def swap(i: Int, j: Int) = {
      var tmp_int = ids1(i); ids1(i) = ids1(j); ids1(j) = tmp_int
      tmp_int = ids2(i); ids2(i) = ids2(j); ids2(j) = tmp_int
      tmp_int = ids3(i); ids3(i) = ids3(j); ids3(j) = tmp_int
      var tmp_float = value(i); value(i) = value(j); value(j) = tmp_float
    }
    
    def lessThan(i: Int, j: Int) = {
      (ids1(i)<ids1(j)) || (ids1(i)==ids1(j)&&ids2(i)<ids2(j)) || 
          (ids1(i)==ids1(j) && ids2(i)==ids2(j)&&ids3(i)<ids3(j))
    }
    
    def quickSort(start: Int, end: Int) : Unit = {
      if (start < end) {
        val pivot = (start+end)/2
        swap(start, pivot)
        var left = start+1
        var right = end
        while (left < right) {
          while (left < right && lessThan(left, start)) left += 1
          while (right > left && (!lessThan(right, start))) right -= 1
          if (left < right) swap(left, right)
        }
        if (lessThan(left, start)) {
          swap(start, left)
          quickSort(start, left-1)
          quickSort(left+1, end)
        }
        else {
          swap(start, left-1)
          quickSort(start, left-2)
          quickSort(left, end)
        }
      }
    }
    
    quickSort(0, nnz-1)
    if (multicore) {
      //to faciliate O(1) access to the entries through all three dimensions
      i = 0
      while (i < nnz) {
        ptr2_buf(ids2(i)) += i
        ptr3_buf(ids3(i)) += i
        i += 1
      }
    }
    val ptr2 = if (multicore) ptr2_buf.map(_.toArray) else null
    val ptr3 = if (multicore) ptr3_buf.map(_.toArray) else null
    new SparseCube(ids1, ptr1, array1, ids2, ptr2, array2, ids3, ptr3, array3, value)
  }
}
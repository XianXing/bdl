package utilities

import scala.collection.mutable.{HashMap, HashSet}

class CSRMatrix(val indices: Array[Array[Int]], val values: Array[Array[Float]], 
    val rowMap: Array[Int], val numCols: Int) 
  extends Serializable {
  val numRows = rowMap.length
}

object CSRMatrix {
  
  def apply(featureMatrix: Array[SparseVector]) = {
    //sparse matrix implemented with compressed sparse row (CSR) format
    val numCols = featureMatrix.length
    val isBinary = featureMatrix(0).isBinary
    val rowSet = new HashSet[Int]
    var nnz = 0
    var n = 0
    while (n < numCols) {
      val rowIndices = featureMatrix(n).getIndices
      val length = rowIndices.length
      var i = 0
      while (i < length) {
        rowSet.add(rowIndices(i))
        nnz += 1
        i += 1
      }
      n += 1
    }
    val numRows = rowSet.size
    val row_ptr = Array.ofDim[Int](numRows)
    val rowMap = new HashMap[Int, Int]
    //need them to be sorted for an easy reverse operation (see Model.toLocal function)
    val rowArray = rowSet.toArray.sorted
    var p = 0 
    while(p < numRows) {
      rowMap.put(rowArray(p), p)
      p += 1
    }
    n = 0
    while (n < numCols) {
      val rowIndices = featureMatrix(n).getIndices
      val values = if (isBinary) null else featureMatrix(n).getValues
      val length = rowIndices.length
      var l = 0
      while (l < length) {
        val p = rowMap.getOrElse(rowIndices(l), -1)
        row_ptr(p) += 1
        l += 1
      }
      n += 1
    }
    val indices = new Array[Array[Int]](numRows)
    val values = if (isBinary) null else new Array[Array[Float]](numRows)
    p = 0
    while (p < numRows) { 
      indices(p) = new Array[Int](row_ptr(p))
      if (!isBinary) values(p) = new Array[Float](row_ptr(p))
      row_ptr(p) = 0
      p += 1
    }
    n = 0
    while (n < numCols) {
      val rowIndices = featureMatrix(n).getIndices
      val rowValues = if (isBinary) null else featureMatrix(n).getValues
      val length = rowIndices.length
      var l = 0
      while (l < length) {
        val p = rowMap.getOrElse(rowIndices(l), -1)
        val i = row_ptr(p)
        row_ptr(p) += 1
        indices(p)(i) = n
        if (!isBinary) values(p)(i) = rowValues(l)
        l += 1
      }
      n += 1
    }
    new CSRMatrix(indices, values, rowArray, numCols)
  }
}

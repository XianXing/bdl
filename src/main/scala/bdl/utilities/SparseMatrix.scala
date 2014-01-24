package utilities

import scala.collection.mutable.{HashMap, HashSet}

class SparseMatrix (
    val row_ptr : Array[Int], val col_idx : Array[Int], val value_r : Array[Float],
    val col_ptr : Array[Int], val row_idx : Array[Int], val value_c : Array[Float],
    val rowMap: Array[Int], val colMap: Array[Int], val numRows: Int, val numCols: Int
    ) extends Serializable {
  
  def getSE(rowFactors: Array[Array[Float]], colFactors: Array[Array[Float]],
    transpose: Boolean = false) = {
    var se = 0f
    val ss : Tuple3[Int, Int, Int] = (1, 1, 1)
    if (transpose) {
      //row/col factors are rank*M/rank*N respectively
      var k = 0; val rank = rowFactors.length
      val res = value_r
      val numRows = rowFactors(0).length
      while (k < rank) {
        val rowFactor = rowFactors(k)
        val colFactor = colFactors(k)
        var r = 0; var i = 0; 
        while (r < numRows) {
          i = row_ptr(r)
          while (i < row_ptr(r+1)) {
            res(i) -= rowFactor(r)*colFactor(col_idx(i))
            i += 1
          }
          r += 1
        }
        k += 1
      }
      var i = 0; val l = res.length 
      while (i < l) { se += res(i)*res(i); i+=1 }
    }
    else {
      var r = 0; var i = 0
      val numRows = rowFactors.length
      while (r < numRows) {
        i = row_ptr(r)
        val rowFactor = rowFactors(r)
        val l = rowFactor.length
        while (i < row_ptr(r+1)) {
          val colFactor = colFactors(col_idx(i))
          var k = 0; var sum = 0f; 
          while (k < l) {
            sum += rowFactor(k)*colFactor(k)
            k += 1
          }
          se += (sum - value_r(i))*(sum - value_r(i))
          i += 1
        }
        r += 1
      }
    }
    se
  }
  
  def getPred(rowFactors: Array[Array[Float]], colFactors: Array[Array[Float]],
    transpose: Boolean = false) = {
    var se = 0f
    val result = Array.ofDim[((Int, Int), (Float, Float, Int))](value_r.length)
    if (transpose) {
      //row/col factors are rank*M/rank*N respectively
      var k = 0; val rank = rowFactors.length
      val pred = Array.ofDim[Float](value_r.length)
      val numRows = rowFactors(0).length
      while (k < rank) {
        val rowFactor = rowFactors(k)
        val colFactor = colFactors(k)
        var r = 0; var i = 0; 
        while (r < numRows) {
          i = row_ptr(r)
          while (i < row_ptr(r+1)) {
            pred(i) += rowFactor(r)*colFactor(col_idx(i))
            i += 1
          }
          r += 1
        }
        k += 1
      }
      var r = 0; var i = 0; 
      while (r < numRows) {
        i = row_ptr(r)
        while (i < row_ptr(r+1)) {
          result(i) = ((rowMap(r), colMap(col_idx(i))), (value_r(i), pred(i), 1))
          i += 1
        }
        r += 1
      }
    }
    else {
      var r = 0; var i = 0
      val numRows = rowFactors.length
      while (r < numRows) {
        i = row_ptr(r)
        while (i < row_ptr(r+1)) {
          val c = col_idx(i)
          val rowFactor = rowFactors(r)
          val colFactor = colFactors(c)
          var k = 0; var pred = 0f; val l = rowFactor.length
          while (k < l) {
            pred += rowFactor(k)*colFactor(k)
            k += 1
          }
          result(i) = ((rowMap(r), colMap(col_idx(i))), (value_r(i), pred, 1))
          i += 1
        }
        r += 1
      }
    }
    result
  }
}

object SparseMatrix {
  def apply(records: Seq[Record]) = {
    //build a sparse matrix implemented with both compressed sparse row (CSR) 
    //and compressed sparse column (CSC) formats
    val rowSet = new HashSet[Int]; val colSet = new HashSet[Int]
    var maxRowIdx = 0
    var maxColIdx = 0
    var nnz = 0
    records.foreach(record => {
      maxRowIdx = math.max(record.rowIdx, maxRowIdx)
      maxColIdx = math.max(record.colIdx, maxColIdx)
      rowSet.add(record.rowIdx); colSet.add(record.colIdx)
      nnz += 1
    })
    println("begin to build the sparse matrix,  nnz: " + nnz)
    val row_idx = new Array[Int](nnz); val col_idx = new Array[Int](nnz)
    val value_r = new Array[Float](nnz); val value_c = new Array[Float](nnz)
    println("all large arrays initialized")
    val numRows = rowSet.size
    val numCols = colSet.size
    println("numRows: " + numRows + " maxRowIdx: " + maxRowIdx)
    println("numCols: " + numCols + " maxColIdx: " + maxColIdx)
    val row_ptr = Array.ofDim[Int](numRows+1)
    val col_ptr = Array.ofDim[Int](numCols+1)
    val rowMap = new HashMap[Int, Int]
    val colMap = new HashMap[Int, Int]
    //need them to be sorted for an easy reverse operation (see Model.toLocal function)
    val rowArray = rowSet.toArray.sorted; val colArray = colSet.toArray.sorted
    var r = 0; while(r<numRows) {rowMap.put(rowArray(r), r); r+=1}
    var c = 0; while(c<numCols) {colMap.put(colArray(c), c); c+=1}
    
    var i = 0
    records.foreach(record => {
      val rowInx = record.rowIdx
      val colInx = record.colIdx
      row_idx(i) = rowMap.getOrElse(rowInx, -1)
      col_idx(i) = colMap.getOrElse(colInx, -1)
      row_ptr(row_idx(i)+1) += 1
      col_ptr(col_idx(i)+1) += 1
      value_r(i) = record.value
      i += 1
    })
    
    r = 1; while (r <= numRows) { row_ptr(r) += row_ptr(r-1); r += 1}
    c = 1; while (c <= numCols) { col_ptr(c) += col_ptr(c-1); c += 1}
    
    def swap(i: Int, j: Int) = {
      var tmp_int = col_idx(i); col_idx(i) = col_idx(j); col_idx(j) = tmp_int
      tmp_int = row_idx(i); row_idx(i) = row_idx(j); row_idx(j) = tmp_int
      var tmp_float = value_r(i); value_r(i) = value_r(j); value_r(j) = tmp_float
    }
    
    def lessThan(i: Int, j: Int) = {
      //actually here we could only compare the row indices, because we only
      //need to group cols with same row indices together, the relative position
      //of col indices doesn't really matter
      (row_idx(i) < row_idx(j)) || (row_idx(i)==row_idx(j)&&col_idx(i)<=col_idx(j))
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
    
    // Generate CRS format
    quickSort(0, nnz-1)
    
    // Transpose CRS into CCS matrix
    r = 0
    while (r < numRows) {
      i = row_ptr(r)
      while (i < row_ptr(r+1)) {
        val c = col_idx(i)
        row_idx(col_ptr(c)) = r
        value_c(col_ptr(c)) = value_r(i)
        col_ptr(c) += 1
        i += 1
      }
      r += 1
    }
    c = numCols; while(c > 0) {col_ptr(c) = col_ptr(c-1); c -= 1}
    col_ptr(0) = 0
    println("finished sparse matrix initializing")
    new SparseMatrix(row_ptr, col_idx, value_r, col_ptr, row_idx, value_c, 
        rowArray, colArray, numRows, numCols)
  }
  
  def apply(array: Array[(Array[Int], Array[Int], Array[Float])]) : SparseMatrix = {
    val sizes = array.map(_._1.length)
    val nnz = sizes.sum
    val row_idx = new Array[Int](nnz)
    val col_idx = new Array[Int](nnz)
    val value_r = new Array[Float](nnz)
    val length = array.length
    var l = 0
    var count = 0
    while (l < length) {
      val _1 = array(l)._1
      val _2 = array(l)._2
      val _3 = array(l)._3
      System.arraycopy(_1, 0, row_idx, count, sizes(l))
      System.arraycopy(_2, 0, col_idx, count, sizes(l))
      System.arraycopy(_3, 0, value_r, count, sizes(l))
      count += sizes(l)
      //release the memory now for garbage collection
      array(l) = null
      l += 1
    }
    SparseMatrix(row_idx, col_idx, value_r)
  }
  
  def apply(row_idx: Array[Int], col_idx: Array[Int], value_r: Array[Float]) = {
    val nnz = row_idx.length
    val rowSet = new HashSet[Int]; val colSet = new HashSet[Int]
    var maxRowIdx = 0
    var maxColIdx = 0
    var i = 0
    while (i < nnz) {
      maxRowIdx = math.max(row_idx(i), maxRowIdx)
      maxColIdx = math.max(col_idx(i), maxColIdx)
      rowSet.add(row_idx(i)); colSet.add(col_idx(i))
      i += 1
    }
    println("begin to build the sparse matrix,  nnz: " + nnz)
    val numRows = rowSet.size
    val numCols = colSet.size
    println("numRows: " + numRows + " maxRowIdx: " + maxRowIdx)
    println("numCols: " + numCols + " maxColIdx: " + maxColIdx)
    val row_ptr = new Array[Int](numRows+1)
    val col_ptr = new Array[Int](numCols+1)
    val rowMap = new HashMap[Int, Int]
    val colMap = new HashMap[Int, Int]
    //need them to be sorted for an easy reverse operation (see Model.toLocal function)
    val rowArray = rowSet.toArray.sorted; val colArray = colSet.toArray.sorted
    var r = 0; while(r<numRows) {rowMap.put(rowArray(r), r); r+=1}
    var c = 0; while(c<numCols) {colMap.put(colArray(c), c); c+=1}
    rowSet.clear
    colSet.clear
    i = 0
    while (i < nnz) {
      row_idx(i) = rowMap.getOrElse(row_idx(i), -1)
      col_idx(i) = colMap.getOrElse(col_idx(i), -1)
      row_ptr(row_idx(i)+1) += 1
      col_ptr(col_idx(i)+1) += 1
      i += 1
    }
    rowMap.clear
    colMap.clear
    r = 1; while (r <= numRows) { row_ptr(r) += row_ptr(r-1); r += 1}
    c = 1; while (c <= numCols) { col_ptr(c) += col_ptr(c-1); c += 1}
    
    def swap(i: Int, j: Int) = {
      var tmp_int = col_idx(i); col_idx(i) = col_idx(j); col_idx(j) = tmp_int
      tmp_int = row_idx(i); row_idx(i) = row_idx(j); row_idx(j) = tmp_int
      var tmp_float = value_r(i); value_r(i) = value_r(j); value_r(j) = tmp_float
    }
    
    def lessThan(i: Int, j: Int) = {
      //actually here we could only compare the row indices, because we only
      //need to group cols with same row indices together, the relative position
      //of col indices doesn't really matter
      (row_idx(i) < row_idx(j)) || (row_idx(i)==row_idx(j)&&col_idx(i)<=col_idx(j))
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
    // Generate CRS format
    quickSort(0, nnz-1)
    
    // Transpose CRS into CCS matrix
    val value_c = new Array[Float](nnz)
    r = 0
    while (r < numRows) {
      i = row_ptr(r)
      while (i < row_ptr(r+1)) {
        val c = col_idx(i)
        row_idx(col_ptr(c)) = r
        value_c(col_ptr(c)) = value_r(i)
        col_ptr(c) += 1
        i += 1
      }
      r += 1
    }
    c = numCols; while(c > 0) {col_ptr(c) = col_ptr(c-1); c -= 1}
    col_ptr(0) = 0
    println("finished sparse matrix initializing")
    new SparseMatrix(row_ptr, col_idx, value_r, col_ptr, row_idx, value_c, 
        rowArray, colArray, numRows, numCols)
  }
  
  def apply(featureMatrix: Array[SparseVector]) = {
    //sparse matrix implemented with compressed sparse row (CSR) format
    val numCols = featureMatrix.length
    println("num of data: " + numCols)
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
    println("nnz: " + nnz)
    val numRows = rowSet.size
    val row_ptr = Array.ofDim[Int](numRows+1)
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
        row_ptr(p+1) += 1
        l += 1
      }
      n += 1
    }
    rowSet.clear
    rowMap.clear
    p = 0
    while (p < numRows) { 
      row_ptr(p+1) += row_ptr(p)
      p += 1
    }
    val col_idx = new Array[Int](nnz)
    val value_r = if (isBinary) null else new Array[Float](nnz)
    n = 0
    while (n < numCols) {
      val rowIndices = featureMatrix(n).getIndices
      val values = if (isBinary) null else featureMatrix(n).getValues
      val length = rowIndices.length
      var l = 0
      while (l < length) {
        val p = rowMap.getOrElse(rowIndices(l), -1)
        val i = row_ptr(p)
        row_ptr(p) += 1
        col_idx(i) = n
        if (!isBinary) value_r(i) = values(l)
        l += 1
      }
      n += 1
    }
    p = numRows
    while (p > 0) {
      row_ptr(p) = row_ptr(p-1)
      p -= 1
    }
    row_ptr(0) = 0
    new SparseMatrix(row_ptr, col_idx, value_r, null, null, null, rowArray, null, 
        numRows, numCols)
  }
}
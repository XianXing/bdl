package utilities

import scala.collection.mutable.ArrayBuilder

class SparseVector(val indices : Array[Int], val values : Array[Float]) 
  extends Serializable {
  
  def this(indices : Array[Int]) = this(indices, null)
  def isBinary = values == null || values.length == 0
  def size = indices.length
  def contains(index : Int) = indices.indexWhere(i => i == index) >= 0 
  
  def getIndices = indices
  def getValues = if (isBinary) Array.fill(size)(1.0f) else values
  
  def getZippedPairs = indices.zip(values)
  
  def toArray(length: Int = size) = {
    val array = new Array[Float](length)
    val values = getValues
    var l = 0
    while (l < size) {
      array(indices(l)) = values(l)
      l += 1
    }
    array
  }
  
  def toArray(array: Array[Float]) = {
    var l = 0
    while (l < size) {
      array(indices(l)) = values(l)
      l += 1
    }
  }
  
  def apply(index : Int) = {
    val pos = indices.indexWhere(i => i == index)
    if (pos >= 0) {
      if (isBinary) 1.0f
      else values(pos)
    }
    else 0.0f
  }
  
  def + (that: SparseVector) : SparseVector = {
    var this_idx = 0
    var that_idx = 0
    val this_length = this.size
    val that_length = that.size
    val this_indices = this.getIndices
    val this_values = this.getValues
    val that_indices = that.getIndices
    val that_values = that.getValues
    
    val res_indices = new ArrayBuilder.ofInt
    val res_values = new ArrayBuilder.ofFloat
    while(this_idx < this_length && that_idx < that_length) {
      if (this_indices(this_idx) == that_indices(that_idx)) {
        res_indices += this_indices(this_idx) 
        res_values += this_values(this_idx) + that_values(that_idx)
        this_idx += 1
        that_idx += 1
      }
      else if (this_indices(this_idx) > that_indices(that_idx)) {
        res_indices += that_indices(that_idx)
        res_values += that_values(that_idx)
        that_idx += 1
      }
      else {
        res_indices += this_indices(this_idx)
        res_values += this_values(this_idx)
        this_idx += 1
      }
    }
    while(this_idx < this_length) {
      res_indices += this_indices(this_idx)
      res_values += this_values(this_idx)
      this_idx += 1
    }
    while(that_idx < that_length) {
      res_indices += that_indices(that_idx)
      res_values += that_values(that_idx)
      that_idx += 1
    }
    SparseVector(res_indices.result, res_values.result)
  }
  
  def + (that: Vector) : Vector = {
    var this_idx = 0
    var that_idx = 0
    val this_length = this.size
    val that_length = that.length
    val this_indices = this.getIndices
    val this_values = this.getValues
    
    val res_values = new Array[Float](that_length)
    
    while (this_idx < this_length) {
      if (this_indices(this_idx) == that_idx) {
        res_values(that_idx) = this_values(this_idx) + that(that_idx)
        this_idx += 1
        that_idx += 1
      }
      else {
        res_values(that_idx) = that(that_idx)
        that_idx += 1
      }
    }
    while(that_idx < that_length) {
      res_values(that_idx) = that(that_idx)
      that_idx += 1
    }
    Vector(res_values)
  }
  
  def - (that: SparseVector) : SparseVector = {
    var this_idx = 0
    var that_idx = 0
    val this_length = this.size
    val that_length = that.size
    val this_indices = this.getIndices
    val this_values = this.getValues
    val that_indices = that.getIndices
    val that_values = that.getValues
    
    val res_indices = new ArrayBuilder.ofInt
    val res_values = new ArrayBuilder.ofFloat
    while(this_idx < this_length && that_idx < that_length) {
      if (this_indices(this_idx) == that_indices(that_idx)) {
        res_indices += this_indices(this_idx) 
        res_values += this_values(this_idx) - that_values(that_idx)
        this_idx += 1
        that_idx += 1
      }
      else if (this_indices(this_idx) > that_indices(that_idx)) {
        res_indices += that_indices(that_idx)
        res_values += (-that_values(that_idx))
        that_idx += 1
      }
      else {
        res_indices += this_indices(this_idx)
        res_values += this_values(this_idx)
        this_idx += 1
      }
    }
    while(this_idx < this_length) {
      res_indices += this_indices(this_idx)
      res_values += this_values(this_idx)
      this_idx += 1
    }
    while(that_idx < that_length) {
      res_indices += that_indices(that_idx)
      res_values += (-that_values(that_idx))
      that_idx += 1
    }
    SparseVector(res_indices.result, res_values.result)
  }
  
  def - (that: Vector) : Vector = {
    var this_idx = 0
    var that_idx = 0
    val this_length = this.size
    val that_length = that.length
    val this_indices = this.getIndices
    val this_values = this.getValues
    
    val res_values = new Array[Float](that_length)
    
    while (this_idx < this_length) {
      if (this_indices(this_idx) == that_idx) {
        res_values(that_idx) = this_values(this_idx) - that(that_idx)
        this_idx += 1
        that_idx += 1
      }
      else {
        res_values(that_idx) = (-that(that_idx))
        that_idx += 1
      }
    }
    while(that_idx < that_length) {
      res_values(that_idx) = (-that(that_idx))
      that_idx += 1
    }
    Vector(res_values)
  }
  
  def * (that: SparseVector) : SparseVector = {
    var this_idx = 0
    var that_idx = 0
    val this_length = this.size
    val that_length = that.size
    val this_indices = this.getIndices
    val this_values = this.getValues
    val that_indices = that.getIndices
    val that_values = that.getValues
    
    val res_indices = new ArrayBuilder.ofInt
    res_indices.sizeHint(size)
    val res_values = new ArrayBuilder.ofFloat
    res_values.sizeHint(size)
    while(this_idx < this_length && that_idx < that_length) {
      if (this_indices(this_idx) == that_indices(that_idx)) {
        res_indices += this_indices(this_idx) 
        res_values += this_values(this_idx) * that_values(that_idx)
        this_idx += 1
        that_idx += 1
      }
      else if (this_indices(this_idx) > that_indices(that_idx)) {
        that_idx += 1
      }
      else {
        this_idx += 1
      }
    }
    SparseVector(res_indices.result, res_values.result)
  }
  
  def * (that: Float) : SparseVector = {
    val res_values = new Array[Float](size)
    val values = this.getValues
    var i = 0
    while (i < indices.length) {
      res_values(i) = values(i)*that
      i += 1
    }
    SparseVector(indices, res_values)
  }
  
  def dot(that: SparseVector) : Float = {
    var this_idx = 0
    var that_idx = 0
    val this_length = this.size
    val that_length = that.size
    val this_indices = this.getIndices
    val this_values = this.getValues
    val that_indices = that.getIndices
    val that_values = that.getValues
    
    var res = 0.0f
    while(this_idx < this_length && that_idx < that_length) {
      if (this_indices(this_idx) == that_indices(that_idx)) {
        res += this_values(this_idx) * that_values(that_idx)
        this_idx += 1
        that_idx += 1
      }
      else if (this_indices(this_idx) < that_indices(that_idx)) {
        this_idx += 1
      }
      else {
        that_idx += 1
      }
    }
    res
  }
  
  def dot(that: Vector) : Float = {
    var res = 0.0f
    var i = 0
    val values = this.getValues
    while (i < size) {
      res += values(i)*that(indices(i))
      i += 1
    }
    res
  }
  
  def squaredL2Norm = {
    if (values != null) values.map(ele => ele*ele).reduce(_+_)
    else indices.length
  }
  def l2Norm = math.sqrt(squaredL2Norm)
  
  def l1Norm = {
    if (values != null) values.map(ele => math.abs(ele)).reduce(_+_)
    else indices.length
  }
  
  def squaredL2Dist (that: SparseVector) : Float = {
    var this_idx = 0
    var that_idx = 0
    val this_length = this.size
    val that_length = that.size
    val this_indices = this.getIndices
    val this_values = this.getValues
    val that_indices = that.getIndices
    val that_values = that.getValues
    
    var res = 0.0f
    while(this_idx < this_length && that_idx < that_length) {
      if (this_indices(this_idx) == that_indices(that_idx)) {
        val diff = this_values(this_idx) - that_values(that_idx)
        res += diff*diff
        this_idx += 1
        that_idx += 1
      }
      else if (this_indices(this_idx) < that_indices(that_idx)) {
        val diff = this_values(this_idx)
        res += diff*diff
        this_idx += 1
      }
      else {
        val diff = - that_values(that_idx)
        res += diff*diff
        that_idx += 1
      }
    }
    res
  }
  
  def l1Dist (that: SparseVector) : Float = {
    var this_idx = 0
    var that_idx = 0
    val this_length = this.size
    val that_length = that.size
    val this_indices = this.getIndices
    val this_values = this.getValues
    val that_indices = that.getIndices
    val that_values = that.getValues
    
    var res = 0.0f
    while(this_idx < this_length && that_idx < that_length) {
      if (this_indices(this_idx) == that_indices(that_idx)) {
        res += math.abs(this_values(this_idx) - that_values(that_idx))
        this_idx += 1
        that_idx += 1
      }
      else if (this_indices(this_idx) < that_indices(that_idx)) {
        res += math.abs(this_values(this_idx))
        this_idx += 1
      }
      else {
        res += math.abs(that_values(this_idx))
        that_idx += 1
      }
    }
    res
  }
//  override def toString = 
//    if (isBinary) indices.mkString(" ")
//    else indices.zip(values).mkString("(", ": ", ")")
}

object SparseVector {
  class Multiplier(num: Float) {
    def * (vec: SparseVector) = vec * num
  }

  implicit def doubleToMultiplier(num: Float) = new Multiplier(num)
  
  def apply(indices: Array[Int]) = new SparseVector(indices)
  
  def apply(indices: Array[Int], sorted : Boolean) = {
    def quickSort(indices: Array[Int]) : Array[Int] = {
      if (indices.length <= 1) indices
      else {
        val pivot = indices(indices.length/2)
        Array.concat(quickSort(indices.filter(pivot > )), 
            indices.filter(pivot == ), 
            quickSort(indices.filter(pivot < )))
      }
    }
    if (sorted) new SparseVector(indices)
    else new SparseVector(quickSort(indices))
  }
  
  def apply(indices: Array[Int], values: Array[Float]) 
    = new SparseVector(indices, values)
  
  def apply(indices: Array[Int], values: Array[Float], sorted : Boolean) = {
    def quickSort(pairs : Array[(Int, Float)]) : Array[(Int, Float)] = {
      if (pairs.length <= 1) pairs
      else {
        val pivot = pairs(pairs.length/2)
        Array.concat(quickSort(pairs.filter(pair => pair._1 < pivot._1)), 
            pairs.filter(pair => pair._1 == pivot._1), 
            quickSort(pairs.filter(pair => pair._1 > pivot._1)))
      }
    }
    if (sorted) new SparseVector(indices, values)
    else {
      val sortedPair = quickSort(indices.zip(values))
      var i = 0
      while (i < indices.length) {
        indices(i) = sortedPair(i)._1
        values(i) = sortedPair(i)._2
        i += 1
      }
      new SparseVector(indices, values)
    }
  }
  
  def apply(indices : Array[Int], vec : Vector) = {
    val length = indices.length
    val values = new Array[Float](length)
    var i = 0
    while (i < length) {
      values(i) = vec(indices(i))
      i += 1
    }
    new SparseVector(indices, values)
  }
  
  def apply(indices : Array[Int], value : Float) = {
    val length = indices.length
    val values = new Array[Float](length)
    var i = 0
    while (i < length) {
      values(i) = value
      i += 1
    }
    new SparseVector(indices, values)
  }
  
  def minusAndTimes(sv1 : SparseVector, sv2: SparseVector, sc: Float) 
    : SparseVector = {
    //calculate (sv1-sv2)*sc
    var this_idx = 0
    var that_idx = 0
    val this_length = sv1.size
    val that_length = sv2.size
    val this_indices = sv1.getIndices
    val this_values = sv1.getValues
    val that_indices = sv2.getIndices
    val that_values = sv2.getValues
        
    val res_indices = new ArrayBuilder.ofInt
    res_indices.sizeHint(math.max(sv1.size, sv2.size))
    val res_values = new ArrayBuilder.ofFloat
    res_values.sizeHint(math.max(sv1.size, sv2.size))
    
    while(this_idx < this_length && that_idx < that_length) {
      if (this_indices(this_idx) == that_indices(that_idx)) {
        res_indices += this_indices(this_idx) 
        res_values += sc*(this_values(this_idx) - that_values(that_idx))
        this_idx += 1
        that_idx += 1
      }
      else if (this_indices(this_idx) > that_indices(that_idx)) {
        res_indices += that_indices(that_idx)
        res_values += sc*(-that_values(that_idx))
        that_idx += 1
      }
      else {
        res_indices += this_indices(this_idx)
        res_values += sc*this_values(this_idx)
        this_idx += 1
      }
    }
    while(this_idx < this_length) {
      res_indices += this_indices(this_idx)
      res_values += this_values(this_idx)
      this_idx += 1
    }
    while(that_idx < that_length) {
      res_indices += that_indices(that_idx)
      res_values += (-that_values(that_idx))
      that_idx += 1
    }
    SparseVector(res_indices.result, res_values.result)
  }
  
  def addAndTimes(sv1 : SparseVector, sv2: SparseVector, sc: Float): SparseVector = {
    //calculate (sv1+sv2)*f
    var this_idx = 0
    var that_idx = 0
    val this_length = sv1.size
    val that_length = sv2.size
    val this_indices = sv1.getIndices
    val this_values = sv1.getValues
    val that_indices = sv2.getIndices
    val that_values = sv2.getValues
        
    val res_indices = new ArrayBuilder.ofInt
    res_indices.sizeHint(math.max(sv1.size, sv2.size))
    val res_values = new ArrayBuilder.ofFloat
    res_values.sizeHint(math.max(sv1.size, sv2.size))
    
    while(this_idx < this_length && that_idx < that_length) {
      if (this_indices(this_idx) == that_indices(that_idx)) {
        res_indices += this_indices(this_idx) 
        res_values += sc*(this_values(this_idx) + that_values(that_idx))
        this_idx += 1
        that_idx += 1
      }
      else if (this_indices(this_idx) > that_indices(that_idx)) {
        res_indices += that_indices(that_idx)
        res_values += sc*(that_values(that_idx))
        that_idx += 1
      }
      else {
        res_indices += this_indices(this_idx)
        res_values += sc*this_values(this_idx)
        this_idx += 1
      }
    }
    while(this_idx < this_length) {
      res_indices += this_indices(this_idx)
      res_values += sc*this_values(this_idx)
      this_idx += 1
    }
    while(that_idx < that_length) {
      res_indices += that_indices(that_idx)
      res_values += sc*that_values(that_idx)
      that_idx += 1
    }
    SparseVector(res_indices.result, res_values.result)
  }
}
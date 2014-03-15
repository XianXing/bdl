package utilities

import scala.math

class IntVector(val elements: Array[Int]) extends Serializable {
  def length = elements.length
  def toArray = elements
  def toArray(array: Array[Int]) = {
    assert(array.length == length)
    var l = 0
    while (l < length) {
      array(l) = elements(l)
      l += 1
    }
  }
  def apply(index: Int) = elements(index)
  def update(index : Int, value : Int) = elements(index) = value
  
  def + (other: IntVector): IntVector = {
    if (length != other.length)
      throw new IllegalArgumentException("Vectors of different length")
    return IntVector(length, i => this(i) + other(i))
  }
  
  def * (other: IntVector): IntVector = {
    if (length != other.length)
      throw new IllegalArgumentException("Vectors of different length")
    return IntVector(length, i => this(i) * other(i))
  }
 
  def / (other: IntVector): IntVector = {
    if (length != other.length)
      throw new IllegalArgumentException("Vectors of different length")
    return IntVector(length, i => this(i) / other(i))
  }  

  def add(other: IntVector) = this + other

  def - (other: IntVector): IntVector = {
    if (length != other.length)
      throw new IllegalArgumentException("Vectors of different length")
    return IntVector(length, i => this(i) - other(i))
  }

  def subtract(other: IntVector) = this - other

  def dot(other: IntVector): Int = {
    if (length != other.length)
      throw new IllegalArgumentException("Vectors of different length")
    var ans = 0
    var i = 0
    while (i < length) {
      ans += this(i) * other(i)
      i += 1
    }
    return ans
  }
  
  /**
   * return (this + plus) dot other, but without creating any intermediate storage
   * @param plus
   * @param other
   * @return
   */
  def plusDot(plus: IntVector, other: IntVector): Int = {
    if (length != other.length)
      throw new IllegalArgumentException("Vectors of different length")
    if (length != plus.length)
      throw new IllegalArgumentException("Vectors of different length")
    var ans = 0
    var i = 0
    while (i < length) {
      ans += (this(i) + plus(i)) * other(i)
      i += 1
    }
    return ans
  }
  
  def plusDot(plus: IntVector, other: Int): Int = {
    if (length != plus.length)
      throw new IllegalArgumentException("Vectors of different length")
    var ans = 0
    var i = 0
    while (i < length) {
      ans += (this(i) + plus(i)) * other
      i += 1
    }
    return ans
  }
  
  def plusTimes(plus: IntVector, other: Int): IntVector = {
    if (length != plus.length)
      throw new IllegalArgumentException("Vectors of different length")
    return IntVector(length, i => (this(i) + plus(i)) * other)
  }

  def += (other: IntVector): IntVector = {
    if (length != other.length)
      throw new IllegalArgumentException("Vectors of different length")
    var ans = 0.0
    var i = 0
    while (i < length) {
      elements(i) += other(i)
      i += 1
    }
    this
  }
  
  def getMean() = elements.sum/length
  def getVariance() = {
    val mean = getMean()
    elements.map(e=>{(e-mean)*(e-mean)}).sum/length
  }

  def addInPlace(other: IntVector) = this += other

  def + (scalar: Int): IntVector = IntVector(length, i => this(i) + scalar)
  def * (scale: Int): IntVector = IntVector(length, i => this(i) * scale)

  def multiply (d: Int) = this * d

  def / (d: Int): IntVector = this * (1 / d)

  def divide (d: Int) = this / d

  def unary_- = this * -1

  def sum = elements.reduceLeft(_ + _)

  def squaredDist(other: IntVector): Double = {
    var ans = 0.0
    var i = 0
    while (i < length) {
      ans += (this(i) - other(i)) * (this(i) - other(i))
      i += 1
    }
    return ans
  }

  def dist(other: IntVector): Double = math.sqrt(squaredDist(other))
  
  def squaredL2Norm = elements.map(ele => ele*ele).reduce(_+_)
  def l2Norm = math.sqrt(squaredL2Norm)
  
  override def toString = elements.mkString("(", ", ", ")")
}

object IntVector {
  def apply(elements: Array[Int]) = new IntVector(elements)

  def apply(elements: Int*) = new IntVector(elements.toArray)

  def apply(length: Int, initializer: Int => Int): IntVector = {
    val elements = new Array[Int](length)
    var i = 0
    while(i < length) {
      elements(i) = initializer(i)
      i += 1
    }
    return new IntVector(elements)
  }
  
  def apply(arr : Array[(Int, Int)]): IntVector = {
    val elements = new Array[Int](arr.length)
    arr.foreach(pair => elements(pair._1) = pair._2)
    return new IntVector(elements)
  }
  
  def apply(index: Array[Int], length: Int): IntVector = {
    val elements = new Array[Int](length)
    index.foreach(i => elements(i) = 1)
    return new IntVector(elements)
  }
  
  def apply(arr : Array[(Int, Int)], P : Int): IntVector = {
    val elements = new Array[Int](P)
    arr.foreach(pair => elements(pair._1) = pair._2)
    return new IntVector(elements)
  }
  
  def apply(matrix : Array[Array[Int]]) : Array[IntVector] = {
    matrix.map(array => IntVector(array))
  }
  
  def getElements(matrix : Array[IntVector]) = {
    matrix.map(vector => vector.elements)
  }
  
  def plusAndTimes(v1: IntVector, v2: IntVector, sc: Int): IntVector = {
    if (v1.length != v2.length)
      throw new IllegalArgumentException("Vectors of different length")
    return IntVector(v1.length, i => (v1(i) + v2(i)) * sc)
  }
  
  def zeros(length: Int) = new IntVector(new Array[Int](length))

  def ones(length: Int) = IntVector(length, _ => 1)

  class Multiplier(num: Int) {
    def * (vec: IntVector) = vec * num
  }

  implicit def doubleToMultiplier(num: Int) = new Multiplier(num)

  implicit object VectorAccumParam 
    extends org.apache.spark.AccumulatorParam[IntVector] {
    def addInPlace(t1: IntVector, t2: IntVector) = t1.addInPlace(t2)
    def zero(initialValue: IntVector) = IntVector.zeros(initialValue.length)
  }

}

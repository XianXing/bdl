package utilities

import scala.math

class DoubleVector(val elements: Array[Double]) extends Serializable {
  def length = elements.length
  def toArray = elements
  def toArray(array: Array[Double]) = {
    assert(array.length == length)
    var l = 0
    while (l < length) {
      array(l) = elements(l)
      l += 1
    }
  }
  def apply(index: Int) = elements(index)
  def update(index : Int, value : Double) = elements(index) = value
  
  def + (other: DoubleVector): DoubleVector = {
    if (length != other.length)
      throw new IllegalArgumentException("Vectors of different length")
    return DoubleVector(length, i => this(i) + other(i))
  }
  
  def * (other: DoubleVector): DoubleVector = {
    if (length != other.length)
      throw new IllegalArgumentException("Vectors of different length")
    return DoubleVector(length, i => this(i) * other(i))
  }
 
  def / (other: DoubleVector): DoubleVector = {
    if (length != other.length)
      throw new IllegalArgumentException("Vectors of different length")
    return DoubleVector(length, i => this(i) / other(i))
  }  

  def add(other: DoubleVector) = this + other

  def - (other: DoubleVector): DoubleVector = {
    if (length != other.length)
      throw new IllegalArgumentException("Vectors of different length")
    return DoubleVector(length, i => this(i) - other(i))
  }

  def subtract(other: DoubleVector) = this - other

  def dot(other: DoubleVector): Double = {
    if (length != other.length)
      throw new IllegalArgumentException("Vectors of different length")
    var ans = 0.0
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
  def plusDot(plus: DoubleVector, other: DoubleVector): Double = {
    if (length != other.length)
      throw new IllegalArgumentException("Vectors of different length")
    if (length != plus.length)
      throw new IllegalArgumentException("Vectors of different length")
    var ans = 0.0
    var i = 0
    while (i < length) {
      ans += (this(i) + plus(i)) * other(i)
      i += 1
    }
    return ans
  }
  
  def plusDot(plus: DoubleVector, other: Double): Double = {
    if (length != plus.length)
      throw new IllegalArgumentException("Vectors of different length")
    var ans = 0.0
    var i = 0
    while (i < length) {
      ans += (this(i) + plus(i)) * other
      i += 1
    }
    return ans
  }
  
  def plusTimes(plus: DoubleVector, other: Double): DoubleVector = {
    if (length != plus.length)
      throw new IllegalArgumentException("Vectors of different length")
    return DoubleVector(length, i => (this(i) + plus(i)) * other)
  }

  def += (other: DoubleVector): DoubleVector = {
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

  def addInPlace(other: DoubleVector) = this += other

  def + (scalar: Double): DoubleVector = DoubleVector(length, i => this(i) + scalar)
  def * (scale: Double): DoubleVector = DoubleVector(length, i => this(i) * scale)

  def multiply (d: Double) = this * d

  def / (d: Double): DoubleVector = this * (1 / d)

  def divide (d: Double) = this / d

  def unary_- = this * -1

  def sum = elements.reduceLeft(_ + _)

  def squaredDist(other: DoubleVector): Double = {
    var ans = 0.0
    var i = 0
    while (i < length) {
      ans += (this(i) - other(i)) * (this(i) - other(i))
      i += 1
    }
    return ans
  }
  
  def dist(other: DoubleVector): Double = math.sqrt(squaredDist(other))
  
  def squaredL2Norm = elements.map(ele => ele*ele).reduce(_+_)
  def l2Norm = math.sqrt(squaredL2Norm)
  
  override def toString = elements.mkString("(", ", ", ")")
}

object DoubleVector {
  def apply(elements: Array[Double]) = new DoubleVector(elements)

  def apply(elements: Double*) = new DoubleVector(elements.toArray)

  def apply(length: Int, initializer: Int => Double): DoubleVector = {
    val elements = new Array[Double](length)
    var i = 0
    while(i < length) {
      elements(i) = initializer(i)
      i += 1
    }
    return new DoubleVector(elements)
  }
  
  def apply(arr : Array[(Int, Double)]): DoubleVector = {
    val elements = new Array[Double](arr.length)
    arr.foreach(pair => elements(pair._1) = pair._2)
    return new DoubleVector(elements)
  }
  
  def apply(index: Array[Int], length: Int): DoubleVector = {
    val elements = new Array[Double](length)
    index.foreach(i => elements(i) = 1.0)
    return new DoubleVector(elements)
  }
  
  def apply(arr : Array[(Int, Double)], P : Int): DoubleVector = {
    val elements = new Array[Double](P)
    arr.foreach(pair => elements(pair._1) = pair._2)
    return new DoubleVector(elements)
  }
  
  def apply(matrix : Array[Array[Double]]) : Array[DoubleVector] = {
    matrix.map(array => DoubleVector(array))
  }
  
  def getElements(matrix : Array[DoubleVector]) = {
    matrix.map(vector => vector.elements)
  }
  
  def plusAndTimes(v1: DoubleVector, v2: DoubleVector, sc: Double): DoubleVector = {
    if (v1.length != v2.length)
      throw new IllegalArgumentException("Vectors of different length")
    return DoubleVector(v1.length, i => (v1(i) + v2(i)) * sc)
  }
  
  def zeros(length: Int) = new DoubleVector(new Array[Double](length))

  def ones(length: Int) = DoubleVector(length, _ => 1)

  class Multiplier(num: Double) {
    def * (vec: DoubleVector) = vec * num
  }

  implicit def doubleToMultiplier(num: Double) = new Multiplier(num)

  implicit object VectorAccumParam 
    extends org.apache.spark.AccumulatorParam[DoubleVector] {
    def addInPlace(t1: DoubleVector, t2: DoubleVector) = t1.addInPlace(t2)
    def zero(initialValue: DoubleVector) = DoubleVector.zeros(initialValue.length)
  }

}

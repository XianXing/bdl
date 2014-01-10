package utilities

import scala.math

class Vector(val elements: Array[Float]) extends Serializable {
  def length = elements.length
  def toArray = elements
  def toArray(array: Array[Float]) = {
    assert(array.length == length)
    var l = 0
    while (l < length) {
      array(l) = elements(l)
      l += 1
    }
  }
  def apply(index: Int) = elements(index)
  def update(index : Int, value : Float) = elements(index) = value
  
  def + (other: Vector): Vector = {
    if (length != other.length)
      throw new IllegalArgumentException("Vectors of different length")
    return Vector(length, i => this(i) + other(i))
  }
  
  def * (other: Vector): Vector = {
    if (length != other.length)
      throw new IllegalArgumentException("Vectors of different length")
    return Vector(length, i => this(i) * other(i))
  }
 
  def / (other: Vector): Vector = {
    if (length != other.length)
      throw new IllegalArgumentException("Vectors of different length")
    return Vector(length, i => this(i) / other(i))
  }  

  def add(other: Vector) = this + other

  def - (other: Vector): Vector = {
    if (length != other.length)
      throw new IllegalArgumentException("Vectors of different length")
    return Vector(length, i => this(i) - other(i))
  }

  def subtract(other: Vector) = this - other

  def dot(other: Vector): Float = {
    if (length != other.length)
      throw new IllegalArgumentException("Vectors of different length")
    var ans = 0.0f
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
  def plusDot(plus: Vector, other: Vector): Float = {
    if (length != other.length)
      throw new IllegalArgumentException("Vectors of different length")
    if (length != plus.length)
      throw new IllegalArgumentException("Vectors of different length")
    var ans = 0.0f
    var i = 0
    while (i < length) {
      ans += (this(i) + plus(i)) * other(i)
      i += 1
    }
    return ans
  }
  
  def plusDot(plus: Vector, other: Float): Float = {
    if (length != plus.length)
      throw new IllegalArgumentException("Vectors of different length")
    var ans = 0.0f
    var i = 0
    while (i < length) {
      ans += (this(i) + plus(i)) * other
      i += 1
    }
    return ans
  }
  
  def plusTimes(plus: Vector, other: Float): Vector = {
    if (length != plus.length)
      throw new IllegalArgumentException("Vectors of different length")
    return Vector(length, i => (this(i) + plus(i)) * other)
  }

  def += (other: Vector): Vector = {
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

  def addInPlace(other: Vector) = this += other

  def + (scalar: Float): Vector = Vector(length, i => this(i) + scalar)
  def * (scale: Float): Vector = Vector(length, i => this(i) * scale)

  def multiply (d: Float) = this * d

  def / (d: Float): Vector = this * (1 / d)

  def divide (d: Float) = this / d

  def unary_- = this * -1

  def sum = elements.reduceLeft(_ + _)

  def squaredDist(other: Vector): Float = {
    var ans = 0.0f
    var i = 0
    while (i < length) {
      ans += (this(i) - other(i)) * (this(i) - other(i))
      i += 1
    }
    return ans
  }

  def dist(other: Vector): Float = math.sqrt(squaredDist(other)).toFloat
  
  def squaredL2Norm = elements.map(ele => ele*ele).reduce(_+_)
  def l2Norm = math.sqrt(squaredL2Norm)
  
  override def toString = elements.mkString("(", ", ", ")")
}

object Vector {
  def apply(elements: Array[Float]) = new Vector(elements)

  def apply(elements: Float*) = new Vector(elements.toArray)

  def apply(length: Int, initializer: Int => Float): Vector = {
    val elements = new Array[Float](length)
    var i = 0
    while(i < length) {
      elements(i) = initializer(i)
      i += 1
    }
    return new Vector(elements)
  }
  
  def apply(arr : Array[(Int, Float)]): Vector = {
    val elements = new Array[Float](arr.length)
    arr.foreach(pair => elements(pair._1) = pair._2)
    return new Vector(elements)
  }
  
  def apply(arr : Array[(Int, Float)], P : Int): Vector = {
    val elements = new Array[Float](P)
    arr.foreach(pair => elements(pair._1) = pair._2)
    return new Vector(elements)
  }
  
  def apply(matrix : Array[Array[Float]]) : Array[Vector] = {
    matrix.map(array => Vector(array))
  }
  
  def getElements(matrix : Array[Vector]) = {
    matrix.map(vector => vector.elements)
  }
  
  def plusAndTimes(v1: Vector, v2: Vector, sc: Float): Vector = {
    if (v1.length != v2.length)
      throw new IllegalArgumentException("Vectors of different length")
    return Vector(v1.length, i => (v1(i) + v2(i)) * sc)
  }
  
  def zeros(length: Int) = new Vector(new Array[Float](length))

  def ones(length: Int) = Vector(length, _ => 1)

  class Multiplier(num: Float) {
    def * (vec: Vector) = vec * num
  }

  implicit def doubleToMultiplier(num: Float) = new Multiplier(num)

  implicit object VectorAccumParam extends org.apache.spark.AccumulatorParam[Vector] {
    def addInPlace(t1: Vector, t2: Vector) = t1.addInPlace(t2)
    def zero(initialValue: Vector) = Vector.zeros(initialValue.length)
  }

}

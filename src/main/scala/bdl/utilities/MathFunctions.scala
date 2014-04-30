package utilities

import breeze.linalg._
import breeze.numerics._

object MathFunctions {
  def sigmoid(value : Double): Double = {
    if (value < -10) 4.5398e-05
    else if (value > 10) 1-1e-05
    else 1 / (1 + math.exp(-value))
  }
  
  def tetragamma(x: Double): Double = {
    if (x > 0 && x <= 1e-3) -2 / (x * x * x)
    else if (x >= 30) {
      val x1 = 1/x
      val x2 = x1*x1
      val x3 = x2*x1
      //using the asymptotic expansion for large x
      -x2 - x3*(1 + x1 * (0.5 - x2 * (1.0/6 - x2/6)))
    }
    else {
      tetragamma(x + 1) - 2 / (x * x * x)
    }
  }
  
  def dirExp(input: DenseVector[Double]): DenseVector[Double] = {
    digamma(input) :- digamma(sum(input))
  }
  
  def eDirExp(input: DenseVector[Double]): DenseVector[Double] = {
    exp(digamma(input) :- digamma(sum(input)))
  }
  
}
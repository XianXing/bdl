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
  
  def normalize(input: DenseMatrix[Double], multicore: Boolean, 
      output: DenseMatrix[Double]) = {
    val rows = input.rows
    if (multicore) {
      for (r <- (0 until rows).par) {
        output(r, ::) := input(r, ::) :/ sum(input(r, ::).t)
      }
    } else {
      var r = 0
      while (r < rows) {
        output(r, ::) := input(r, ::) :/ sum(input(r, ::).t)
        r += 1
      }
    }
  }
  
  def normalize(input: DenseMatrix[Double], multicore: Boolean): DenseMatrix[Double]= {
    val rows = input.rows
    val cols = input.cols
    val output = DenseMatrix.zeros[Double](rows, cols)
    normalize(input, multicore, output)
    output
  }
  
  def dirExp(input: DenseVector[Double]): DenseVector[Double] = {
    digamma(input) :- digamma(sum(input))
  }
  
  def dirExp(eta: DenseMatrix[Double], multicore: Boolean, 
      eLogBeta: DenseMatrix[Double]): Unit = {
    val numTopics = eta.rows
    if (multicore) {
      for (k <- (0 until numTopics).par) {
        eLogBeta(k, ::).t := dirExp(eta(k, ::).t)
      }
    } else {
      for (k <- 0 until numTopics) {
        eLogBeta(k, ::).t := dirExp(eta(k, ::).t)
      }
    }
  }
  
  def dirExp(eta: DenseMatrix[Double], multicore: Boolean): DenseMatrix[Double] = {
    val rows = eta.rows
    val cols = eta.cols
    val eLogBeta = DenseMatrix.zeros[Double](rows, cols)
    dirExp(eta, multicore, eLogBeta)
    eLogBeta
  }
  
  def eDirExp(input: DenseVector[Double]): DenseVector[Double] = {
    exp(digamma(input) :- digamma(sum(input)))
  }
  
  def eDirExp(eta: DenseMatrix[Double], multicore: Boolean, 
      expELogBeta: DenseMatrix[Double]): Unit = {
    val numTopics = eta.rows
    if (multicore) {
      for (k <- (0 until numTopics).par) {
        expELogBeta(k, ::).t := eDirExp(eta(k, ::).t)
      }
    } else {
      for (k <- 0 until numTopics) {
        expELogBeta(k, ::).t := eDirExp(eta(k, ::).t)
      }
    }
  }
  
  def eDirExp(eta: DenseMatrix[Double], multicore: Boolean): DenseMatrix[Double] = {
    val rows = eta.rows
    val cols = eta.cols
    val expELogBeta = DenseMatrix.zeros[Double](rows, cols)
    eDirExp(eta, multicore, expELogBeta)
    expELogBeta
  }
  
  def parallelSum(mat1: DenseMatrix[Double], mat2: DenseMatrix[Double], 
      res: DenseMatrix[Double]) {
    val par = (0 until mat1.cols).par
    for (n <- par) {
      res(::, n) := mat1(::, n) + mat2(::, n)
    }
  }
  
  def parallelSum(mat1: DenseMatrix[Double], mat2: DenseMatrix[Double])
    : DenseMatrix[Double]=  {
    val rows = mat1.rows
    val cols = mat1.cols
    val res = DenseMatrix.zeros[Double](rows, cols)
    val par = (0 until cols).par
    for (n <- par) {
      res(::, n) := mat1(::, n) + mat2(::, n)
    }
    res
  }
  
  def log_sum(log_a: Double, log_b: Double): Double = {
    if (log_a < log_b) {
      log_b+log(1 + exp(log_a-log_b))
    } else {
      log_a+log(1 + exp(log_b-log_a))
    }
  }
}
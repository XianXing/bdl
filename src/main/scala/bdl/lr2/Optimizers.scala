package lr2

import utilities.SparseMatrix
import utilities.Math
import classification.OptimizerType._
import classification.VariationalType._

object Optimizers {
   
  def runCGOrLBFGS(labels: Array[Byte], features: SparseMatrix, 
      numIter: Int, regPara: Double): Array[Double] = {
    val numFeatures = features.numRows
    val weights = new Array[Double](numFeatures)
    val priors = new Array[Double](numFeatures)
    val optimizerType = CG
    runCGOrLBFGS(labels, features, weights, priors, optimizerType,
      numIter, regPara)
  }
  
  def runCGOrLBFGS(labels: Array[Byte], features: SparseMatrix, 
      weights: Array[Double], optimizerType: OptimizerType,
      numIter: Int, regPara: Double): Array[Double] = {
    val numFeatures = features.numRows
    val priors = new Array[Double](numFeatures)
    runCGOrLBFGS(labels, features, weights, priors, optimizerType, 
      numIter, regPara)
  }
  
  //only supports L2 regularization
  def runCGOrLBFGS(
      labels: Array[Byte], features: SparseMatrix, 
      weights: Array[Double], priors: Array[Double], 
      optimizerType: OptimizerType,
      numIter: Int, regPara: Double): Array[Double] = {
    
    val numFeatures = features.numRows
    val gradientPrev = new Array[Double](numFeatures)
    val direction = new Array[Double](numFeatures)
    val gradient = new Array[Double](numFeatures)
    val deltaPara = 
      if (optimizerType == LBFGS) new Array[Double](numFeatures) 
      else null
    val updatedWeights = new Array[Double](numFeatures)
    Array.copy(weights, 0, updatedWeights, 0, numFeatures)
    var iter = 0
    while (iter < numIter) {
      Functions.getGrad(labels, features, updatedWeights, gradient)
      var p = 1 //no shrinkage for the intercept
      while (p < numFeatures) {
        gradient(p) -= regPara*(updatedWeights(p)-priors(p))
        p += 1
      }
      if (iter > 1) {
        if (optimizerType == CG) getCGDirection(gradient, gradientPrev, direction)
        else if (optimizerType == LBFGS) {
          getLBFGSDirection(deltaPara, gradient, gradientPrev, direction)
        }
      }
      else Array.copy(gradient, 0, direction, 0, numFeatures)
      val h = Functions.getHessian(features, updatedWeights, direction)
      p = 1 //no shrinkage for the intercept
      var gu = gradient(0)*direction(0)
      var uhu = 0.0
      while (p < numFeatures) {
        uhu += direction(p)*direction(p)
        gu += gradient(p)*direction(p)
        p += 1
      }
      uhu *= regPara
      uhu += h
      p = 0
      while (p < numFeatures) {
        gradientPrev(p) = gradient(p)
        //equation (17) in Tom Minka 2003
        val delta = gu/uhu*direction(p)
        if (optimizerType == LBFGS) deltaPara(p) = delta
        updatedWeights(p) += delta
        p += 1
      }
      iter += 1
    }
    updatedWeights
  }
  
  def runCD(labels: Array[Byte], features: SparseMatrix, weights: Array[Double], 
      emBayes: Boolean, numIter: Int, rho: Double, gamma: Double)
    : (Array[Double], Double, Double) = {
    val numFeatures = features.numRows
    val priors = new Array[Double](numFeatures)
    val varType = Taylor
    runCD(labels, features, weights, priors, emBayes, varType, numIter, rho, gamma)
  }
  
  def runCD(labels: Array[Byte], features: SparseMatrix, weights: Array[Double], 
      numIter: Int, rho: Double, gamma: Double)
    : (Array[Double], Double, Double) = {
    val numFeatures = features.numRows
    val priors = new Array[Double](numFeatures)
    val varType = Taylor
    val emBayes = true
    runCD(labels, features, weights, priors, emBayes, varType, numIter, rho, gamma)
  }
  
  def runCD(labels: Array[Byte], features: SparseMatrix, weights: Array[Double], 
      priors: Array[Double], numIter: Int, 
      rho: Double, gamma: Double): (Array[Double], Double, Double) = {
    val varType = Taylor
    val emBayes = true
    runCD(labels, features, weights, priors, emBayes, varType, numIter, rho, gamma)
  }
  
  def runCD(labels: Array[Byte], features: SparseMatrix, weights: Array[Double], 
      priors: Array[Double], emBayes: Boolean, numIter: Int, 
      rho: Double, gamma: Double): (Array[Double], Double, Double) = {
    val varType = Taylor
    runCD(labels, features, weights, priors, emBayes, varType, numIter, rho, gamma)
  }
  
  //re-use weights as output 
  def runCD(labels: Array[Byte], features: SparseMatrix, 
      weights: Array[Double], priors: Array[Double], emBayes: Boolean,
      varType: VariationalType, numIter: Int, rho: Double, gamma: Double)
    : (Array[Double], Double, Double) =  {
    val numFeatures = features.numRows
    val numData = features.numCols
    val coefA = new Array[Double](numData)
    val residual = new Array[Double](numData)
    val updatedWeights = new Array[Double](numFeatures)
    Array.copy(weights, 0, updatedWeights, 0, numFeatures)
    val prec = 
      if (varType == Bohning || varType == Jaakkola || emBayes) {
        Array.tabulate(numFeatures)(i => rho*gamma)
      }
      else null
    var iter = 0
    while (iter < numIter) {
      if (varType == Jaakkola) {
        jaakkola(labels, features, updatedWeights, prec, coefA, residual)
      }
      else if (varType == Bohning) {
        bohning(labels, features, updatedWeights, coefA, residual)
      }
      else taylor(labels, features, updatedWeights, coefA, residual)
      // O(nnz)
      updatePara(labels, features, updatedWeights, prec, priors, 
        coefA, residual, rho*gamma, varType, emBayes)
      iter += 1
    }
    
    if (emBayes) {
      val updatedGamma = updateGamma(updatedWeights, priors, prec, rho)
      var p = 0
      var se = 0.0
      while (p < numFeatures) {
        val diff = updatedWeights(p) - priors(p)
        se += diff*diff + 1/prec(p)
        p += 1
      }
      (updatedWeights, updatedGamma, se*updatedGamma)
    }
    else (updatedWeights, gamma, 0)
  }
  
  def getLBFGSDirection(delta_para: Array[Double], gradient: Array[Double], 
      gradient_old: Array[Double], direction: Array[Double]) = {
    val numFeatures = gradient.length
    var p = 0; var dwdg = 0.0; var dgdg = 0.0; var dwg = 0.0; var dgg = 0.0
    while (p < numFeatures) {
      val delta_g = gradient(p) - gradient_old(p)
      dwdg += delta_para(p)*delta_g
      dgdg += delta_g*delta_g
      dwg += delta_para(p)*gradient(p)
      dgg += delta_g*gradient(p)
      p += 1
    }
    val b = 1+dgdg/dwdg
    val a_g = dwg/dwdg
    val a_w = dgg/dwdg - b*a_g
    p = 0
    while (p < numFeatures) {
      val delta_g = gradient(p) - gradient_old(p)
      direction(p) = -gradient(p) + a_w*delta_para(p) + a_g*delta_g
      p += 1
    }
  }
  
  def getCGDirection(gradient: Array[Double], gradient_old: Array[Double], 
      direction: Array[Double]) = {
    val numFeatures = gradient.length
    var p = 0
    var deno = 0.0
    var nume = 0.0
    while (p < numFeatures) {
      //Hestenes-Stiefel formula:
      val delta = gradient(p) - gradient_old(p)
      nume += gradient(p)*delta
      deno += direction(p)*delta
      p += 1
    }
    val beta = nume/deno
    p = 0
    while (p < numFeatures) {
      direction(p) = gradient(p) - direction(p)*beta
      p += 1
    }
  }
  
  def jaakkola(labels: Array[Byte], features: SparseMatrix, weights: Array[Double],
      prec: Array[Double], coefA: Array[Double], residual: Array[Double]) {
    
    val ptr = features.row_ptr
    val idx = features.col_idx
    val value = features.value_r
    val numData = features.numCols
    val numFeatures = features.numRows
    var n = 0
    while (n < numData) {
      coefA(n) = 0f
      residual(n) = 0f
      n += 1
    }
    if (value == null) {
      var p = 0 
      while (p < numFeatures) {
        var i = ptr(p)
        while (i < ptr(p+1)) {
          val n = idx(i)
          coefA(n) += weights(p)
          residual(n) += 1/prec(p)
          i += 1
        }
        p += 1
      }
    }
    else {
      var p = 0 
      while (p < numFeatures) {
        var i = ptr(p)
        while (i < ptr(p+1)) {
          val n = idx(i)
          coefA(n) += weights(p)*value(i)
          residual(n) += value(i)*value(i)/prec(p)
          i += 1
        }
        p += 1
      }
    }
    n = 0
    while (n < numData) {
      val wtx = coefA(n)
      val xi = math.sqrt(wtx*wtx + residual(n)).toFloat
      coefA(n) = if (xi < 1e-3f) 0.25f else (Math.sigmoid(xi)-0.5f)/xi
      residual(n) =  labels(n)/2 - coefA(n)*wtx
      n += 1
    }
  }
  
  def bohning(labels: Array[Byte], features: SparseMatrix, weights: Array[Double],
      coefA: Array[Double], residual: Array[Double]) {
    val numData = labels.length
    var n = 0
    while (n < numData) {
      coefA(n) = 0.25f
      residual(n) = 0f
      n += 1
    }
    //calculate wTx and store it in residual
    Functions.getWTX(features, weights, residual)
    n = 0
    while (n < numData) {
      val wtx = residual(n)
      val exp = math.exp(-wtx).toFloat
      val sigmoid = 1/(1+exp)
      val y = if (labels(n) == 1) 1 else 0
      residual(n) = y-sigmoid
      n += 1
    }
  }
  
  def taylor(labels: Array[Byte], features: SparseMatrix, weights: Array[Double],
      coefA: Array[Double], residual: Array[Double]) {
    
    val numData = coefA.length
    var n = 0
    while (n < numData) {
      residual(n) = 0f
      n += 1
    }
    //calculate wTx and store it in residual
    Functions.getWTX(features, weights, residual)
    n = 0
    while (n < numData) {
      val wtx = residual(n)
      val exp = math.exp(-wtx).toFloat
      val sigmoid = 1/(1+exp)
      coefA(n) = sigmoid - sigmoid*sigmoid
      if (coefA(n) < 1e-5f) coefA(n) = 1e-5f
      val y = if (labels(n)==1) 1 else 0
      residual(n) = y-sigmoid
      n += 1
    }
  }
  
  def updatePara(
      labels: Array[Byte], features: SparseMatrix, 
      weights: Array[Double], prec: Array[Double],
      prior: Array[Double], coefA: Array[Double], residual: Array[Double],
      regPara: Double, varType: VariationalType, emBayes: Boolean) = {
    
    val ptr = features.row_ptr
    val idx = features.col_idx
    val value = features.value_r
    val isBinary = value == null
    val numFeatures = features.numRows
    var p = 0
    while (p < numFeatures) {
      val isBinary = value == null
      var i = ptr(p)
      var nume = 0.0
      var deno = 0.0
      while (i < ptr(p+1)) {
        val n = idx(i)
        if (isBinary) {
          nume += residual(n)+coefA(n)*weights(p)
          deno += coefA(n)
        }
        else {
          nume += value(i)*(residual(n)+coefA(n)*weights(p)*value(i))
          deno += coefA(n)*value(i)*value(i)
        }
        i += 1
      }
      //+ 1e-5 for numerical stability
      val newPrec = deno + regPara + 1e-5
      val newPara = (nume + regPara*prior(p))/newPrec
      if (emBayes || varType == Jaakkola) prec(p) = newPrec
      val diff = weights(p) - newPara
      if (math.abs(diff) > 1e-5) {
        var i = ptr(p)
        while (i < ptr(p+1)) {
          val n = idx(i)
          if (isBinary) residual(n) += coefA(n)*diff
          else residual(n) += coefA(n)*diff*value(i)
          i += 1
        }
      }
      weights(p) = newPara
      p += 1
    }
  }
  
  def updateGamma(weights: Array[Double], prior: Array[Double], 
      prec: Array[Double], rho: Double): Double = {
    val numFeatures = prior.length
    var se = 0.0
    var p = 0
    while (p < numFeatures) {
      val diff = prior(p) - weights(p)
      se += diff*diff + 1/prec(p)
      p += 1
    }
    (0.005f + numFeatures)/(0.005f + rho*se)
  }
  
  def dualAscent(localWeights: Array[Double], globalWeights: Array[Double], 
      lags: Array[Double]) = {
    assert(localWeights.length == globalWeights.length)
    val length = localWeights.length
    var p = 0
    while (p < length) {
      lags(p) += localWeights(p) - globalWeights(p)
      p += 1
    }
    lags
  }
  
  def sAVGMUpdate(weights1: Array[Double], weights2: Array[Double], 
      subsampleRate: Double): Array[Double] = {
    val length = weights1.length
    val weights = new Array[Double](length)
    var p = 0
    var de = 1-subsampleRate
    while (p < length) {
      weights(p) = (weights1(p)-subsampleRate*weights2(p))/de
      p += 1
    }
    weights
  }
  
  def l1Prox(nume: Double, deno: Double, regPara: Double): Double = {
    if (nume > regPara) (nume - regPara)/deno
    else if (nume < -regPara) (nume + regPara)/deno
    else 0f
  }
  
  def l1Prox(nume: Array[Double], deno: Array[Double], regPara: Double)
   : Array[Double] = {
    val numFeatures = nume.length
    val weights = new Array[Double](numFeatures)
    var p = 0
    while (p < numFeatures) {
      weights(p) = l1Prox(nume(p), deno(p), regPara)
      p += 1
    }
    weights
  }
  
  def l2Prox(nume: Double, deno: Double, regPara: Double): Double = {
    nume/(deno + regPara)
  }
  
  def l2Prox(nume: Array[Double], deno: Array[Double], regPara: Double)
      : Array[Double] = {
    val numFeatures = nume.length
    val weights = new Array[Double](numFeatures)
    var p = 0
    while (p < numFeatures) {
      weights(p) = l2Prox(nume(p), deno(p), regPara)
      p += 1
    }
    weights
  }
  
  def l2Prox(nume: Array[Double], deno: Array[Int], regPara: Double)
      : Array[Double] = {
    val numFeatures = nume.length
    val weights = new Array[Double](numFeatures)
    var p = 0
    while (p < numFeatures) {
      weights(p) = l2Prox(nume(p), deno(p), regPara)
      p += 1
    }
    weights
  }
}
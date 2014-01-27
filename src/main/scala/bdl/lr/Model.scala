package lr

import java.util.Random
import scala.collection.mutable.ArrayBuilder
import utilities._

class Model (para: Array[Float], lag: Array[Float], gamma: Array[Float])
  extends Serializable {
  
  def numFeatures = para.length
  def getPara = para
  def getLag = lag
  def getGammaMeanVariance(ard: Boolean) = {
    if (ard) {
      val mean = gamma.sum/gamma.length
      val variance = gamma.map(g => {
        val diff = g-mean
        diff*diff
      }).sum/gamma.length
      (mean, variance)
    }
    else (gamma(0), 0f)
  }
  
  def getGammaSpa(map: Array[Int]) = SparseVector(map, gamma)
  
  def getGamma(map: Array[Int], length: Int) = {
    val elements = new Array[Float](length)
    var p = 0
    while (p < numFeatures) {
      elements(map(p)) = gamma(p)
      p += 1
    }
    Vector(elements)
  }
  
  def getParaStatsSpa(map: Array[Int], admm: Boolean, emBayes: Boolean) = {
    val index = new ArrayBuilder.ofInt
    val value = new ArrayBuilder.ofFloat
    var p = 0
    while (p < numFeatures) {
      val ele = 
        if (admm && emBayes) (para(p) + lag(p))*gamma(p)
        else if (admm && !emBayes) para(p) + lag(p)
        else if (!admm && emBayes) para(p)*gamma(p)
        else para(p)
      if (math.abs(ele) > 1e-5f) {
        index += map(p)
        value += ele
      }
      p += 1
    }
    SparseVector(index.result, value.result)
  }
  
  def getParaStatsSpa(map: Array[Int]) = SparseVector(map, para)
  
  def getParaStats(map: Array[Int], length: Int, admm: Boolean, emBayes: Boolean) = {
    if (length == numFeatures) Vector(para)
    else {
      val elements = new Array[Float](length)
      var p = 0
      while (p < numFeatures) {
        elements(map(p)) = 
          if (admm && emBayes) (para(p) + lag(p))*gamma(p)
          else if (admm && !emBayes) para(p) + lag(p)
          else if (!admm && emBayes) para(p)*gamma(p)
          else para(p)
        p += 1
      }
      Vector(elements)
    }
  }
  
  def getParaStats(map: Array[Int], length: Int) = {
    val elements = new Array[Float](length)
    var p = 0
    while (p < numFeatures) {
      elements(map(p)) = para(p)
      p += 1
    }
    Vector(elements)
  }
  
  def getRegObj(prior: Array[Float], rho: Float, admm: Boolean) = {
    var obj = 0.0
    var p = 0
    while (p < numFeatures) {
      val diff = if (admm) para(p) - (prior(p) + lag(p))
      else para(p) - prior(p)
      obj += rho*gamma(p)*diff*diff/2
      if (admm) obj += rho*gamma(p)*lag(p)*diff
      p += 1
    }
    obj
  }
  
  def getPred(features: SparseMatrix) = {
    val numData = features.numCols
    val pred = new Array[Float](numData)
    Model.getWTX(features, para, pred)
    var n = 0
    while (n < numData) {
      pred(n) = Functions.sigmoid(pred(n))
      n += 1
    }
    pred
  }
  
  def getLLH(responses: Array[Boolean], features: SparseMatrix) = {
    val numData = features.numCols
    val wtx = new Array[Float](numData)
    Model.getWTX(features, para, wtx)
    var n = 0
    var llh = 0.0
    while (n < numData) {
      val ywtx = if (responses(n)) wtx(n) else -wtx(n)
      if (ywtx > -10) llh += -math.log(1 + math.exp(-ywtx))
      else llh += ywtx
      n += 1
    }
    llh
  }
  
  def updateADMM(globalPara: Array[Float]) = {
    //update the unscaled Lagrangian multipilers
    var p = 0
    while (p < numFeatures) {
      lag(p) += para(p) - globalPara(p)
      globalPara(p) -= lag(p)
      p += 1
    }
  }
  
  def jaakkola(responses: Array[Boolean], features: SparseMatrix, 
      prec: Array[Float], coefA: Array[Float], residual: Array[Float]) = {
    
    val ptr = features.row_ptr
    val idx = features.col_idx
    val value = features.value_r
    val numData = responses.length
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
          coefA(n) += para(p)
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
          coefA(n) += para(p)*value(i)
          residual(n) += value(i)*value(i)/prec(p)
          i += 1
        }
        p += 1
      }
    }
    n = 0
    var obj = 0.0
    while (n < numData) {
      val wtx = coefA(n)
      val y = if (responses(n)) 1 else 0
      val xi = math.sqrt(wtx*wtx + residual(n)).toFloat
      coefA(n) = if (xi < 1e-3f) 0.25f else (Functions.sigmoid(xi)-0.5f)/xi
      residual(n) =  (y-0.5f) - coefA(n)*wtx
      if (wtx < -10) obj += -y*wtx
      else if (wtx > 10) obj += (1-y)*wtx
      else obj += -y*wtx+math.log(1 + math.exp(wtx))
      n += 1
    }
    obj
  }
  
  def bohning(responses: Array[Boolean], features: SparseMatrix,
      coefA: Array[Float], residual: Array[Float]) = {
    val numData = responses.length
    var n = 0
    while (n < numData) {
      coefA(n) = 0.25f
      residual(n) = 0f
      n += 1
    }
    //calculate wTx and store it in residual
    Model.getWTX(features, para, residual)
    n = 0
    var obj = 0.0
    while (n < numData) {
      val wtx = residual(n)
      val exp = Functions.exp(-wtx).toFloat
      val sigmoid = 1/(1+exp)
      val y = if (responses(n)) 1 else 0
      residual(n) = y-sigmoid
      if (wtx < -10) obj += -y*wtx
      else if (wtx > 10) obj += (1-y)*wtx
      else obj += -y*wtx+math.log(1 + 1/exp)
      n += 1
    }
    obj
  }
  
  def taylor(responses: Array[Boolean], features: SparseMatrix,
      coefA: Array[Float], residual: Array[Float]) = {
    
    val numData = coefA.length
    var n = 0
    while (n < numData) {
      residual(n) = 0f
      n += 1
    }
    //calculate wTx and store it in residual
    Model.getWTX(features, para, residual)
    n = 0
    var obj = 0.0
    while (n < numData) {
      val wtx = residual(n)
      val exp = Functions.exp(-wtx).toFloat
      val sigmoid = 1/(1+exp)
      coefA(n) = sigmoid - sigmoid*sigmoid
      if (coefA(n) < 1e-5f) coefA(n) = 1e-5f
      val y = if (responses(n)) 1 else 0
      residual(n) = y-sigmoid
      if (wtx < -10) obj += -y*wtx
      else if (wtx > 10) obj += (1-y)*wtx
      else obj += -y*wtx+math.log(1 + 1/exp)
      n += 1
    }
    obj
  }
  
  def getUpdateStats(p: Int, ptr: Array[Int], idx: Array[Int], value: Array[Float],
      coefA: Array[Float], residual: Array[Float], stats: Array[Float]) = {
    val isBinary = value == null
    var i = ptr(p)
    stats(0) = 0f
    stats(1) = 0f
    while (i < ptr(p+1)) {
      val n = idx(i)
      if (isBinary) {
        stats(0) += residual(n)+coefA(n)*para(p)
        stats(1) += coefA(n)
      }
      else {
        stats(0) += value(i)*(residual(n)+coefA(n)*para(p)*value(i))
        stats(1) += coefA(n)*value(i)*value(i)
      }
      i += 1
    }
  }
  
  def updatePara(responses: Array[Boolean], features: SparseMatrix, prec: Array[Float],
      globalPara: Array[Float], coefA: Array[Float], residual: Array[Float],
      rho: Float, l1: Boolean, jaak: Boolean, emBayes: Boolean) = {
    
    val ptr = features.row_ptr
    val idx = features.col_idx
    val value = features.value_r
    val isBinary = value == null
    var p = 0
    val stats = new Array[Float](2)
    while (p < numFeatures) {
      getUpdateStats(p, ptr, idx, value, coefA, residual, stats)
      //+ 1e-5f for numerical stability
      val newPrec = stats(1) + rho*gamma(p) + 1e-5f
      val newPara = (stats(0) + rho*gamma(p)*globalPara(p))/newPrec
      if (emBayes || jaak) prec(p) = newPrec
      val diff = para(p) - newPara
      if (math.abs(diff) > 1e-5f) {
        var i = ptr(p)
        while (i < ptr(p+1)) {
          val n = idx(i)
          if (isBinary) residual(n) += coefA(n)*diff
          else residual(n) += coefA(n)*diff*value(i)
          i += 1
        }
      }
      para(p) = newPara
      p += 1
    }
  }
  
  def updateGamma(prior: Array[Float], prec: Array[Float], rho: Float, ard: Boolean) = {
    if (ard) {
      var p = 0
      while (p < numFeatures) {
        val diff = prior(p) - para(p)
        val se = diff*diff + 1/prec(p)
        gamma(p) = (0.005f + 1)/(0.005f + rho*se)
        if (gamma(p) < 1e-10f) gamma(p) = 1e-10f
        p += 1
      }
    }
    else {
      var se = 0f
      var p = 0
      while (p < numFeatures) {
        val diff = prior(p) - para(p)
        se += diff*diff + 1/prec(p)
        p += 1
      }
      val gammaNew = (0.005f + numFeatures)/(0.005f + rho*se)
      p = 0
      while (p < numFeatures) {
        gamma(p) = gammaNew
        p += 1
      }
    }
  }
  
  def runCD(responses: Array[Boolean], features: SparseMatrix, 
      maxIter: Int, thre: Float, rho: Float = 1f, 
      admm: Boolean = false, l1: Boolean = false, 
      bohn: Boolean = false, jaak: Boolean = false, 
      emBayes: Boolean = false, ard: Boolean = false,
      globalPara: Array[Float] = new Array[Float](numFeatures)) = {
    
    if (admm) updateADMM(globalPara)
    val numData = responses.length
    val coefA = new Array[Float](numData)
    val residual = new Array[Float](numData)
    val prec = 
      if (bohn || jaak || emBayes) Array.tabulate(numFeatures)(i => gamma(i))
      else null
    var iter = 0
    var obj = -Double.MaxValue
    var obj_old = Double.NegativeInfinity
    var llh = -Double.MaxValue
    var llh_old = Double.NegativeInfinity
    var sampleCount = 0
    while (iter < maxIter && math.abs(llh-llh_old) > thre) {
      obj_old = obj
      llh_old = llh
      // O(nnz + N exp)
      obj = 
        if (jaak) jaakkola(responses, features, prec, coefA, residual)
        else if (bohn) bohning(responses, features, coefA, residual)
        else taylor(responses, features, coefA, residual)
      // O(nnz)
      updatePara(responses, features, prec, globalPara, coefA, residual, rho, 
        l1, jaak, emBayes)
      if (emBayes) updateGamma(globalPara, prec, rho, ard)
      obj += getRegObj(globalPara, rho, admm)
      llh = getLLH(responses, features)
      iter += 1
    }
    val squaredDiff = 
      if (emBayes) {
        var p = 0
        var acc = 0f
        while (p < numFeatures) {
          val diff = para(p) - globalPara(p)
          acc += gamma(p)*(diff*diff + 1/prec(p))
          p += 1
        }
        acc
      }
      else 0f
    (this, iter, obj, squaredDiff)
  }
  
  def runCGQN(responses: Array[Boolean], features: SparseMatrix, 
      maxIter: Int, thre: Float, cg: Boolean = true, rho: Float = 1f, 
      admm: Boolean = false, globalPara: Array[Float] = new Array[Float](numFeatures)) 
    = {
    
    if (admm) updateADMM(globalPara)
    val gradient = new Array[Float](numFeatures)
    val gradient_old = new Array[Float](numFeatures)
    val direction = new Array[Float](numFeatures)
    val delta_para = if (cg) null else new Array[Float](numFeatures)
    var iter = 0
    var obj = -Double.MaxValue
    var obj_old = Double.NegativeInfinity
    var sampleCount = 0
    while (iter < maxIter && math.abs(obj-obj_old) > thre) {
      obj_old = obj
      //O(nnz + N exp)
      obj = Functions.getGrad(responses, features, para, gradient)
      obj += getRegObj(globalPara, rho, admm)
      var p = 1 //no shrinkage for the intercept
      while (p < numFeatures) {
        gradient(p) -= rho*gamma(p)*(para(p)-globalPara(p))
        p += 1
      }
      if (iter > 1) {
        //O(P)
        if (cg) Functions.getCGDirection(gradient, gradient_old, direction)
        else Functions.getLBFGSDirection(delta_para, gradient, gradient_old, direction)
      }
      else Functions.copy(gradient, direction)
      var gu = gradient(0)*direction(0)
      var hessian = 0f
      p = 1
      while (p < numFeatures) {
        hessian += direction(p)*direction(p)*gamma(p)*rho
        gu += gradient(p)*direction(p)
        p += 1
      }
      //O(nnz + N exp)
      val h = Functions.getHessian(features, para, direction)
      hessian += h
      p = 0
      while (p < numFeatures) {
        gradient_old(p) = gradient(p)
        val delta = gu/h*direction(p)
        if (!cg) delta_para(p) = delta
        para(p) += delta
        p += 1
      }
      iter += 1
    }
    (this, iter, obj, 0f)
  }
}

object Model {
  def apply (numFeatures: Int, gamma_init: Float, admm: Boolean) = {
    val para = new Array[Float](numFeatures)
    val lag = if (admm) new Array[Float](numFeatures) else null
    val gamma = Array.fill(numFeatures)(gamma_init)
    new Model(para, lag, gamma)
  }
  
  def apply (para: Array[Float]) = {
    val numFeatures = para.length
    new Model(para, null, null)
  }
  
  def getYWTX(responses: Array[Boolean], features: SparseMatrix, para: Array[Float],
      ywtx: Array[Float]) = {
    val ptr = features.row_ptr
    val idx = features.col_idx
    val value = features.value_r
    val isBinary = value == null
    val numFeatures = ptr.length - 1
    var p = 0 
    while (p < numFeatures) {
      var i = ptr(p)
      while (i < ptr(p+1)) {
        val n = idx(i)
        if (isBinary && responses(n)) ywtx(n) += para(p)
        else if (isBinary && !responses(n)) ywtx(n) -= para(p)
        else if (!isBinary && responses(n)) ywtx(n) += value(i)*para(p)
        else ywtx(n) -= value(i)*para(p)
        i += 1
      }
      p += 1
    }
  }
  
  def getWTX(features: SparseMatrix, para: Array[Float], wtx: Array[Float]) = {
    val ptr = features.row_ptr
    val idx = features.col_idx
    val value = features.value_r
    val numFeatures = ptr.length - 1
    if (value == null) {
      var p = 0 
      while (p < numFeatures) {
        var i = ptr(p)
        while (i < ptr(p+1)) {
          val n = idx(i)
          wtx(n) += para(p)
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
          wtx(n) += value(i)*para(p)
          i += 1
        }
        p += 1
      }
    }
  }
}
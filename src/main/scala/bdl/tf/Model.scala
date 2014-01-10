package tf

import utilities._
import java.util.Random

class Model (
    val factorMats: Array[Array[Array[Float]]], 
    precisionMats: Array[Array[Array[Float]]],
    lagMulMats: Array[Array[Array[Float]]],
    val gammas: Array[Array[Float]],
    val idxMaps: Array[Array[Int]]) extends Serializable {
  
  val numDims = factorMats.length
  val dimSizes = factorMats.map(mat => mat(0).length)
  val numFactors = factorMats(0).length
  
  def setGamma(gamma_init: Array[Float]) = {
    assert(gamma_init.length == gammas.length, "number of dims in gamma mismatch")
    var d = 0
    while (d < numDims) {
      var k = 0
      while (k < numFactors) {
        gammas(d)(k) = gamma_init(d)
        k += 1
      }
      d += 1
    }
  }
  
  def getFactorStats(dimIdx: Int, admm: Boolean) : Array[Array[Float]] = {
    val factors = factorMats(dimIdx)
    val lagMuls = lagMulMats(dimIdx)
    val size = dimSizes(dimIdx)
    if (admm) {
      var k = 0
      val stats = Array.ofDim[Float](numFactors, size)
      while (k < numFactors) {
        var n = 0
        while (n < dimSizes(dimIdx)) {
          stats(k)(n) = factors(k)(n) + lagMuls(k)(n)
          n+=1
        }
        k += 1
      }
      stats
    }
    else factors
  }
  
  def getPrecisionMat(dimIdx: Int) = precisionMats(dimIdx)
  
  //priors are N_i*K matrices, while factors are K*N_i
  def ccd(
      data: SparseCube, maxOuterIter: Int, maxInnerIter: Int, thre: Float, 
      updateDim3: Boolean, multicore: Boolean, vb: Boolean, 
      admms: Array[Boolean]  = Array.fill(3)(false),
      updateGamma: Array[Boolean] = Array.fill(3)(false),
      priorMat1: Array[Array[Float]] = Array.ofDim[Float](dimSizes(0), numFactors),
      priorMat2: Array[Array[Float]] = Array.ofDim[Float](dimSizes(1), numFactors),
      priorMat3: Array[Array[Float]] = Array.ofDim[Float](dimSizes(2), numFactors)
      ): (Model, Int, Float, Float, Float, Float) = {
    
    val ptr1 = data.ptr1; val ids1 = data.ids1
    val ptr2 = data.ptr2; val ids2 = data.ids2
    val ptr3 = data.ptr3; val ids3 = data.ids3
    val value = data.value
    val admm1 = admms(0); val updateGamma1 = updateGamma(0)
    val admm2 = admms(1); val updateGamma2 = updateGamma(1)
    val admm3 = admms(2); val updateGamma3 = updateGamma(2)
    val factorMat1 = factorMats(0); val precMat1 = precisionMats(0)
    val factorMat2 = factorMats(1); val precMat2 = precisionMats(1)
    val factorMat3 = factorMats(2); val precMat3 = precisionMats(2)
    val gamma1 = gammas(0); val gamma2 = gammas(1); val gamma3 = gammas(2)
    val size1 = factorMat1(0).length; val size2 = factorMat2(0).length
    val size3 = factorMat3(0).length
    
    //here we reuse value_c and value_r for res_r and res_c respectively
    val res = Model.getResidual(ptr1, ids2, ids3, value, factorMats, multicore)
    
    if (admm1) ADMM.updateLag(factorMats(0), multicore, lagMulMats(0), priorMat1)
    if (admm2) ADMM.updateLag(factorMats(1), multicore, lagMulMats(1), priorMat2)
    if (admm3&&updateDim3)
      ADMM.updateLag(factorMats(2), multicore, lagMulMats(2), priorMat3)
    val priorMat1T = if (multicore) null 
    else Array.tabulate(numFactors, size1)((k,n) => priorMat1(n)(k))
    val priorMat2T = if (multicore) null 
    else Array.tabulate(numFactors, size2)((k,n) => priorMat2(n)(k))
    val priorMat3T = if (multicore && !updateDim3) null 
    else Array.tabulate(numFactors, size3)((k,n) => priorMat3(n)(k))
    var iter = 0; var rmse_old = Float.MinValue; var rmse = 0f
    var sDe1 = 0f; var sDe2 = 0f; var sDe3 = 0f
    val dim1_par = Array.tabulate(size1)(i => i).par
    val dim2_par = Array.tabulate(size2)(i => i).par
    val dim3_par = Array.tabulate(size3)(i => i).par
    
    while (iter < maxOuterIter && math.abs(rmse_old-rmse) > thre) {
      iter += 1; var k = 0; rmse_old = rmse; sDe1 = 0f; sDe2 = 0f; sDe3 = 0f
      while (k<numFactors) {
        val factor1 = factorMat1(k); val prec1 = if (vb) precMat1(k) else null
        val factor2 = factorMat2(k); val prec2 = if (vb) precMat2(k) else null
        val factor3 = factorMat3(k); val prec3 = if (vb) precMat3(k) else null
        Model.updateResidual(ptr1, ids2, ids3, factor1, factor2, factor3,
            true, multicore, res)
        var i = 0
        while (i < maxInnerIter) {
          i += 1
          //update factors for the 2nd dimension
          val de2 = if (multicore)
            dim2_par.map(d => {
              if (vb)
                VB.update(d, ptr2(d), ids1, ids3, res, factor1, prec1, factor3, prec3,
                  priorMat2(d)(k), gamma2(k), factor2, prec2)
              else
                MAP.update(d, ptr2(d), ids1, ids3, res, factor1, factor3, 
                  priorMat2(d)(k), gamma2(k), factor2)
              val diff = factor2(d) - priorMat2(d)(k)
              val de = if (vb) diff*diff + 1/prec2(d) else diff*diff
              de
            }
          ).fold(0f)(_+_)
          else {
            if (vb)
              VB.update(ids2, ids1, ids3, res, factor1, prec1, factor3, prec3,
                priorMat2T(k), gamma2(k), factor2, prec2)
            else
              MAP.update(ids2, ids1, ids3, res, factor1, factor3,
                priorMat2T(k), gamma2(k), factor2)
          }
//          val de2 = 0
          //update factors for the 1st dimension
          val de1 = if (multicore) 
            dim1_par.map(d => {
              if (vb)
                VB.update(d, ptr1(d), ptr1(d+1), ids2, ids3, res, factor2, prec2, 
                  factor3, prec3, priorMat1(d)(k), gamma1(k), factor1, prec1)
              else
                MAP.update(d, ptr1(d), ptr1(d+1), ids2, ids3, res, factor2, factor3, 
                  priorMat1(d)(k), gamma1(k), factor1)
              val diff = factor1(d) - priorMat1(d)(k)
              val de = if (vb) diff*diff + 1/prec1(d) else diff*diff
              de
            }).fold(0f)(_+_)
          else 
            if (vb)
              VB.update(ids1, ids2, ids3, res, factor2, prec2, factor3, prec3,
                priorMat1T(k), gamma1(k), factor1, prec1)
            else
              MAP.update(ids1, ids2, ids3, res, factor2, factor3,
                priorMat1T(k), gamma1(k), factor1)
//          val de1 = 0
          //update factors for the 3rd dimension
          val de3 = if (multicore && updateDim3) 
            dim3_par.map(d => {
              if (vb)
                VB.update(d, ptr3(d), ids1, ids2, res, factor1, prec1, 
                  factor2, prec2, priorMat3(d)(k), gamma3(k), factor3, prec3)
              else
                MAP.update(d, ptr3(d), ids1, ids2, res, factor1, factor2, 
                  priorMat3(d)(k), gamma3(k), factor3)
              val diff = factor3(d) - priorMat3(d)(k)
              val de = if (vb) diff*diff + 1/prec3(d) else diff*diff
              de
            }
          ).fold(0f)(_+_)
          else if (updateDim3) 
            if (vb)
              VB.update(ids3, ids1, ids2, res, factor1, prec1, factor2, prec2,
                priorMat3T(k), gamma3(k), factor3, prec3)
            else
              MAP.update(ids3, ids1, ids2, res, factor1, factor2,
                priorMat3T(k), gamma3(k), factor3)
          else 0f
          if (iter == maxOuterIter && i == maxInnerIter) {
            if (updateGamma1) gamma1(k) = (size1-1)/(de1+0.01f)
            if (updateGamma2) gamma2(k) = (size2-1)/(de2+0.01f)
            if (updateGamma3 && updateDim3) gamma3(k) = (size3-1)/(de3+0.01f)
          }
          if (i == maxInnerIter) {
            sDe1 += de1
            sDe2 += de2
            sDe3 += de3
          }
        }
        Model.updateResidual(ptr1, ids2, ids3, factor1, factor2, factor3,
            false, multicore, res)
        k += 1
      }
      rmse = Model.getRMSE(res)
//      println("training rmse: " + rmse)
    }
    if (iter < maxOuterIter && math.abs(rmse_old-rmse) <= thre) {
      if (updateGamma1) {
        if (vb) VB.updateGamma(factorMat1, precMat1, priorMat1, gamma1)
        else MAP.updateGamma(factorMat1, precMat1, gamma1)
      }
      if (updateGamma2) {
        if (vb) VB.updateGamma(factorMat2, precMat2, priorMat2, gamma2)
        else MAP.updateGamma(factorMat2, precMat2, gamma2)
      }
      if (updateGamma3) {
        if (vb) VB.updateGamma(factorMat3, precMat3, priorMat3, gamma3)
        else MAP.updateGamma(factorMat3, precMat3, gamma3)
      }
    }
    sDe1 = math.sqrt(sDe1/(size1*numFactors)).toFloat
    sDe2 = math.sqrt(sDe2/(size2*numFactors)).toFloat
    sDe3 = math.sqrt(sDe3/(size3*numFactors)).toFloat
    (this, iter, rmse, sDe1, sDe2, sDe3)
  }
}

object Model {
  
  //initialization for 3-dim tensor factorization
  def apply (
      data: SparseCube, numFactors: Int, gamma_init: Array[Float], 
      admms: Array[Boolean], vb: Boolean) : Model = {
        
    val maps = new Array[Array[Int]](3)
    maps(0) = data.map1
    maps(1) = data.map2
    maps(2) = data.map3
    
    val sizes = new Array[Int](3)
    sizes(0) = data.map1.length
    sizes(1) = data.map2.length
    sizes(2) = data.map3.length
    
    val factorMats = sizes.map(size => Array.ofDim[Float](numFactors, size))
    
    randInit(sizes(0), numFactors, maps(0), 0, factorMats(0))
//    randInit(sizes(2), numFactors, maps(2), 2, factorMats(2))
    var n = 0
    while (n < sizes(2)) {
      var k = 0
      while (k < numFactors) {
        factorMats(2)(k)(n) = 1f
        k += 1
      }
      n += 1
    }
    
    val precisionMats = if (vb) gamma_init.view.zip(sizes).map{
        case(gamma, size) => Array.fill(numFactors, size)(Float.PositiveInfinity)
      }.toArray
    else Array.ofDim[Array[Array[Float]]](3)
    
    val lagMulMats = admms.view.zip(sizes).map{
      case (flag, size) => if (flag) Array.ofDim[Float](numFactors, size)
        else null
      }.toArray
    
    val gammas = gamma_init.map(Array.fill(numFactors)(_))
    
    new Model(factorMats, precisionMats, lagMulMats, gammas, maps)
  }
  
  // Hash an integer to propagate random bits at all positions, 
  // similar to java.util.HashTable
  def hash(x: Int): Int = {
    val r = x ^ (x >>> 20) ^ (x >>> 12)
    r ^ (r >>> 7) ^ (r >>> 4)
  }
  
  // randomly initialize the local latent factors
  def randInit(size: Int, numFactors: Int, idxMap: Array[Int], seed: Int,
      factor: Array[Array[Float]]) = {
    var n = 0
    val rand = new Random()
    while (n < size) {
      var k = 0
      // so the local latent factors with the same global index are aligned
      rand.setSeed(hash(idxMap(n)<<seed))
      while (k < numFactors) {
        factor(k)(n) = 0.1f*(rand.nextFloat-0.5f)
        k += 1
      }
      n += 1
    }
  }
  
  def toLocal(idxMap: Array[Int], pairs: Array[(Int, Array[Float])]) 
    : Array[Array[Float]] = {
    val size = idxMap.length; val numFactors = pairs(0)._2.length
    var i = 0; var j = 0
    var localFactor = Array.ofDim[Array[Float]](size)
    val l = pairs.length
    while (i < size && j < l) {
      if (idxMap(i) == pairs(j)._1) { 
        localFactor(i) = pairs(j)._2
        i+=1
        j+=1 
      }
      else if (idxMap(i) < pairs(j)._1) {
        localFactor(i) = Array.ofDim[Float](numFactors)
        i+=1
      }
      else j+=1
    }
    while (i < size) { localFactor(i) = Array.ofDim[Float](numFactors); i+=1 }
    localFactor
  }
  
  def getRMSE(res: Array[Float], multicore: Boolean = false) : Float = {
    val nnz = res.length
    val se =
      if (multicore) res.par.map(r => r*r).fold(0f)(_+_)
      else { var i = 0; var sd = 0f; while (i < nnz) { sd += res(i)*res(i); i+=1}; sd }
    math.sqrt(se/nnz).toFloat
  }
  
  def updateResidual(ptr1: Array[Int], ids2: Array[Int], ids3: Array[Int],
      factor1: Array[Float], factor2: Array[Float], factor3: Array[Float],
      add: Boolean, multicore: Boolean, res: Array[Float]) = {
    
    val length = ptr1.length-1
    if (add) {
      if (multicore) {
        for (d <- (0 until length).par) {
          var n = ptr1(d)
          while (n < ptr1(d+1)) {
            res(n) += factor1(d)*factor2(ids2(n))*factor3(ids3(n))
            n += 1
          }
        }
      }
      else {
        var d = 0
        while(d < length) {
          var n = ptr1(d)
          while (n < ptr1(d+1)) {
            res(n) += factor1(d)*factor2(ids2(n))*factor3(ids3(n))
            n += 1
          }
          d += 1
        }
      }
    }
    else {
      if (multicore) {
        for (d <- (0 until length).par) {
          var n = ptr1(d)
          while (n < ptr1(d+1)) {
            res(n) -= factor1(d)*factor2(ids2(n))*factor3(ids3(n))
            n += 1
          }
        }
      }
      else {
        var d = 0
        while (d < length) {
          var n = ptr1(d)
          while (n < ptr1(d+1)) {
            res(n) -= factor1(d)*factor2(ids2(n))*factor3(ids3(n))
            n += 1
          }
          d += 1
        }
      }
    }
  } // end of update residual
  
  def getResidual(ptr1: Array[Int], ids2: Array[Int], ids3: Array[Int],
      value: Array[Float], factorMats: Array[Array[Array[Float]]], multicore: Boolean) 
  : Array[Float] = {
    val length = ptr1.length-1
    val numFactors = factorMats(0).length
    val factor1 = factorMats(0)
    val factor2 = factorMats(1)
    val factor3 = factorMats(2)
    if (multicore) {
      for (l <- (0 until length).par) {
        var n = ptr1(l)
        while (n < ptr1(l+1)) {
          var k = 0
          while (k < numFactors) {
            value(n) -= factor1(k)(l)*factor2(k)(ids2(n))*factor3(k)(ids3(n))
            k += 1
          }
          n += 1
        }
      }
    }
    else {
      for (l <- 0 until length) {
        var n = ptr1(l)
        while (n < ptr1(l+1)) {
          var k = 0
          while (k < numFactors) {
            value(n) -= factor1(k)(l)*factor2(k)(ids2(n))*factor3(k)(ids3(n))
            k += 1
          }
          n += 1
        }
      }
    }
    value
  }//end of getResidual
  
}

package lr

import utilities._

object Functions {
  
  def copy(from: Array[Float], to: Array[Float]) = {
    assert(from.length == to.length)
    var i = 0
    while (i < from.length) {
      to(i) = from(i)
      i += 1
    }
  }
  
  def exp(value: Float) = {
    if (value < -10) 4.54e-5
    else if (value > 10) 22026
    else math.exp(value)
  }
  
  def sigmoid(value: Float) = {
    if (value < -10) 4.5398e-05f
    else if (value > 10) 1-1e-05f
    else 1/(1+math.exp(-value).toFloat)
  }
  
  def sigmoid(x : Double) : Double = {
    1 / (1 + math.exp(-x))
  }
  
  def getBinPred(features : SparseVector, w : Array[Float], numBins: Int)
    : Vector = {
    val prob = sigmoid(features.dot(Vector(w)))
    val inc = 1.0/numBins
    Vector(Array.tabulate(numBins)(i => if (prob > (i-1)*inc) 1.0f else 0.0f ))
  }
  
  def l1Prox(nume: Float, deno: Float, lambda: Float) = {
    if (nume > lambda) (nume - lambda)/deno
    else if (nume < -lambda) (nume + lambda)/deno
    else 0f
  }
  
  def l2Prox(nume: Float, deno: Float, lambda: Float) = {
    nume/(deno + lambda)
  }
  
  def getGrad(responses: Array[Boolean], features: SparseMatrix, w: Array[Float], 
      gradient: Array[Float])= {
    val numData = responses.length
    val numFeatures = w.length
    val ywtx = new Array[Float](numData)
    Model.getYWTX(responses, features, w, ywtx)
    val ptr = features.row_ptr
    val idx = features.col_idx
    val value = features.value_r
    val isBinary = value == null
    var n = 0; var obj = 0.0
    while (n < numData) {
      val exp = Functions.exp(-ywtx(n)).toFloat
      if (ywtx(n) < -10) obj -= ywtx(n)
      else if (ywtx(n) > -10 && ywtx(n) < 10) obj += math.log(1 + exp)
      ywtx(n) = 1/(1+exp)
      n += 1
    }
    var p = 0
    while (p < numFeatures) {
      var i = ptr(p)
      gradient(p) = 0
      while (i < ptr(p+1)) {
        val n = idx(i)
        if (responses(n) && isBinary) gradient(p) += (1 - ywtx(n))
        else if (responses(n) && !isBinary) gradient(p) += (1 - ywtx(n))*value(i)
        else if (!responses(n) && isBinary) gradient(p) -= (1 - ywtx(n))
        else gradient(p) -= (1 - ywtx(n))*value(i)
        i += 1
      }
      p += 1
    }
    obj
  }
  
  def getHessian(features: SparseMatrix, w: Array[Float], u: Array[Float]) = {
    val numData = features.numCols
    val numFeatures = features.numRows
    val ptr = features.row_ptr
    val idx = features.col_idx
    val value = features.value_r
    val isBinary = value == null
    val wtx = new Array[Float](numData)
    val utx = new Array[Float](numData)
    var p = 0
    while (p < numFeatures) {
      var i = ptr(p)
      while (i < ptr(p+1)) {
        val n = idx(i)
        if (isBinary) {
          wtx(n) += w(p)
          utx(n) += u(p)
        }
        else {
          wtx(n) += w(p)*value(i)
          utx(n) += u(p)*value(i)
        }
        i += 1
      }
      p += 1
    }
    var n = 0
    var hessian = 0f
    while (n < numData) {
      val sigmoid = Functions.sigmoid(wtx(n))
      val alpha = sigmoid*(1-sigmoid)
      if (alpha > 1e-5f) hessian += alpha*utx(n)*utx(n)
      else hessian += 1e-5f*utx(n)*utx(n)
      n += 1
    }
    hessian
  }
  
  def getAUC(tp : Array[Float], fp : Array[Float], numPos : Int, numNeg: Int) : Float = {
    assert(tp.length == fp.length)
    var tpr_prev = 0.0f
    var fpr_prev = 0.0f
    var auc = 0.0f
    for (i <- tp.length-1 to 0 by -1) {
      val tpr = tp(i)/numPos
      val fpr = fp(i)/numNeg
      auc += 0.5f*(tpr+tpr_prev)*(fpr-fpr_prev)
      tpr_prev = tpr
      fpr_prev = fpr
    }
    auc
  }
  
  def getLLH(data : Pair[Boolean, SparseVector], w : SparseVector) : Double = {
    val features = data._2
    val response = 
      if (data._1) 1
      else -1
    val yxw = response*features.dot(w)
    if (yxw > -10) -math.log(1 + math.exp(-yxw))
    else yxw
  }
  
  def getLLH(data : (Boolean, SparseVector), w : Vector) : Double = {
    val features = data._2
    val response = 
      if (data._1) 1
      else -1
    val yxw = response*features.dot(w)
    if (yxw > -10) -math.log(1 + math.exp(-yxw))
    else yxw
  }
  
  def getLLH(data : (Boolean, SparseVector), w : Array[Float]) : Double = {
    val features = data._2
    val response = 
      if (data._1) 1
      else -1
    val yxw = response*features.dot(Vector(w))
    if (yxw > -10) -math.log(1 + math.exp(-yxw))
    else yxw
  }
  
  def getGradient(data : Pair[Boolean, SparseVector], w : Vector) : SparseVector = {
    val features = data._2
    val response = 
      if (data._1) 1
      else -1
    features*response*(1-sigmoid(response * features.dot(w)))
  }
  
  def getGradient(data : Pair[Boolean, SparseVector], w : SparseVector)
    : SparseVector = {
    val features = data._2
    val response = 
      if (data._1) 1
      else -1
    features*response*(1-sigmoid(response * features.dot(w)))
  }
  
  def getHessian(data : Pair[Boolean, SparseVector], w : Vector, u : Vector) : Float = {
    val features = data._2
    val sigma = sigmoid(features.dot(w))
    val ux = features.dot(u)
    return (sigma*(1 - sigma))*ux*ux
  }
  
  def getHessian(data : Pair[Boolean, SparseVector], w : SparseVector, u : SparseVector) : Float = {
    val features = data._2
    val sigma = sigmoid(features.dot(w))
    val ux = features.dot(u)
    return (sigma*(1 - sigma))*ux*ux
  }
  
  def getCGDirection(gradient: Array[Float], gradient_old: Array[Float], 
      direction: Array[Float]) = {
    val numFeatures = gradient.length
    var p = 0
    var deno = 0f
    var nume = 0f
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
  
  def getLBFGSDirection(delta_para: Array[Float], gradient: Array[Float], 
      gradient_old: Array[Float], direction: Array[Float]) = {
    val numFeatures = gradient.length
    var p = 0; var dwdg = 0f; var dgdg = 0f; var dwg = 0f; var dgg = 0f
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
  
  def dist_cg(data : Array[(Boolean, SparseVector)], w: Vector, gamma: Float,
      max_iter: Int, bayes: Boolean) : (Array[Float], Float) = {
    
    var wi = w
    val gamma_old = gamma
    val P = w.length
    var delta_w = Vector.ones(P)
    var u = Vector.ones(P)
    var g_old = Vector.ones(P)
    var iter = 0
    while (iter < max_iter) {
      iter += 1
      val g = data.par.map(pair => getGradient(pair, wi)).reduce(_+_) - (wi - w)*gamma_old
      if (iter > 1) {
        val delta_g = g - g_old
        val beta = g.dot(delta_g)/u.dot(delta_g)
        u = g - u*beta
      }
      else u = g
      g_old = g
      val h = data.par.map(pair => getHessian(pair, wi, u)).reduce(_+_)
      delta_w = g.dot(u)/(gamma_old*u.dot(u) + h)*u
      wi = wi + delta_w
    }
    val gamma_updated = if (bayes)
      (0.5f + 0.5f*P)/(0.5f + 0.5f*wi.squaredDist(w))
      else
        gamma
    ((wi*gamma_updated).elements, gamma_updated)
  }
  
  def ADMM_CG(data : Array[(Boolean, SparseVector)], 
      wyr_dist_i: Tuple3[Array[Float], Array[Float], Float], w0: Array[Float], 
      gamma0: Float, max_iter: Int, bayes: Boolean, warmStart : Boolean, obj_th: Double) 
  : Tuple5[Array[Float], Array[Float], Float, Float, Int] = {
    
    // stable version for the non-sparse features
    
    var wi = Vector(wyr_dist_i._1)
    val ui_old = Vector(wyr_dist_i._2)
    val gamma_old = if (bayes) wyr_dist_i._3 else gamma0
    
    val w_global = Vector(w0)
    val ui_new = if (!bayes && warmStart) ui_old + wi - w_global else ui_old
    val P = w_global.length
    var delta_w = Vector.ones(P)
    var u = Vector.ones(P)
    var g_old = Vector.ones(P)
    val prior = w_global-ui_new
    var iter = 0
    var obj = -Double.MaxValue
    var obj_old = Double.NegativeInfinity
    var l2diff = 0f
    while (iter < max_iter && obj-obj_old > obj_th && g_old.squaredL2Norm > 1e-3) {
      iter += 1
      obj_old = obj
      val g = data.par.map(pair => getGradient(pair, wi)).reduce(_+_) - (wi - prior)*gamma_old
      if (iter > 1) {
        val delta_g = g - g_old
        var beta = g.dot(delta_g)/u.dot(delta_g)
        beta = if (beta>10) 10 else if (beta < -10) -10 else beta
        u = g - u*beta
      }
      else u = g
      g_old = g
      val h = data.par.map(pair => getHessian(pair, wi, u)).reduce(_+_) + gamma_old*u.dot(u) + 1e-5f
      delta_w = g.dot(u)/h*u
      wi = wi + delta_w
      l2diff = wi.squaredDist(prior)
      obj = data.par.map(pair => getLLH(pair, wi)).reduce(_+_) - gamma_old/2*l2diff
    }
    obj += gamma_old/2*l2diff
    val gamma_new = 
      if (bayes) {
        val gamma = (0.5f + 0.5f*P)/(0.5f + 0.5f*(l2diff+gamma0))
        math.max(gamma, gamma_old)
      }
      else gamma_old
      
    if (bayes) 
      obj -= gamma_new/2*l2diff
    else 
      obj -= ui_new.dot(wi-w_global)*gamma_new
    
    (wi.elements, ui_new.elements, gamma_new, obj.toFloat, iter)
  }
  
  def ADMM_CG(data : Array[(Boolean, SparseVector)], 
      wyr_dist_i: Tuple4[Array[Int], Array[Float], Array[Float], Float], w0: Array[Float], gamma: Float,
      alpha: Float, beta: Float, max_iter: Int, bayes: Boolean, warmStart : Boolean, obj_th: Double) 
      : Tuple6[Array[Int], Array[Float], Array[Float], Float, Float, Int] = {
    
    //the stable version, exploits the sparse structure in each partition
    
    val keyArray = wyr_dist_i._1
    var wi = SparseVector(keyArray, wyr_dist_i._2)
    val ui = SparseVector(keyArray, wyr_dist_i._3)
    val w_global = SparseVector(keyArray, Vector(w0))
    val gamma_old = if (bayes) wyr_dist_i._4 else gamma
    val ui_new = if (!bayes && warmStart) ui + wi - w_global else ui
    val prior = w_global-ui_new
    
    var delta_w = SparseVector(keyArray)
    var u = SparseVector(keyArray)
    var g_old = SparseVector(keyArray)
    var iter = 0
    var obj = -Double.MaxValue
    var obj_old = Double.NegativeInfinity
    var l2diff = 0f
    while (iter < max_iter && math.abs(obj-obj_old) > obj_th && g_old.squaredL2Norm > 1e-3) {
      iter += 1
      obj_old = obj
      val g = data.par.map(pair => getGradient(pair, wi)).reduce(_+_) - (wi - prior)*gamma_old
      if (iter > 1) {
        val delta_g = g - g_old
        val beta = (g.dot(delta_g) + 1e-5f)/(u.dot(delta_g) + 1e-5f)
        u = g - u*beta
      }
      else u = g
      g_old = g
      val h = data.par.map(pair => getHessian(pair, wi, u)).reduce(_+_) + gamma_old*u.dot(u) + 1e-5f
      delta_w = g.dot(u)/h*u
      wi = wi + delta_w
      l2diff = wi.squaredL2Dist(prior)
      obj = data.par.map(pair => getLLH(pair, wi)).reduce(_+_) - gamma_old/2*l2diff
    }
    obj += gamma_old/2*l2diff
    val gamma_new =
      if (bayes)
       (alpha + 0.5f*wi.size)/(beta + 0.5f*(l2diff+gamma))
      else
        gamma
    
    if (bayes) 
      obj -= gamma_new/2*l2diff
    else 
      obj -= ui_new.dot(wi-w_global)*gamma_new
    
    (keyArray, wi.getValues, ui_new.getValues, gamma_new, obj.toFloat, iter)
  }
  
  def dist_cg_t(data : Array[(Boolean, SparseVector)], 
      wy_dist_i: Tuple3[Array[Int], Array[Float], Array[Float]], w0: Vector, gamma: Float,
      alpha: Float, beta: Float, max_iter: Int, obj_th: Double, bayes: Boolean, filter_th : Double) 
      : Tuple6[Array[Int], Array[Float], Array[Float], Float, Float, Int] = {
    
    // prior on w_i is the heavy-tail t-distribution
    
    val keyArray = wy_dist_i._1
    var wi = SparseVector(keyArray, wy_dist_i._2)
    val yi = SparseVector(keyArray, wy_dist_i._3)
    val w = SparseVector(keyArray, w0)
    val yi_new = if (!bayes) yi + wi - w else yi
//    val prior = if (bayes) w else w-yi_new
    val prior = w 
    val P = keyArray.size
    var residual = (wi-prior).squaredL2Norm
    
    if (bayes && gamma < filter_th) 
      return (keyArray, wy_dist_i._2, wy_dist_i._3, filter_th.toFloat, residual, 0)
    
    var u = SparseVector(keyArray)
    var g_old = SparseVector(keyArray)
    var iter = 0
    var obj = -Double.MaxValue
    var obj_old = Double.NegativeInfinity
    
    while (iter < max_iter && math.abs(obj-obj_old) > obj_th) {
      iter += 1
      obj_old = obj
      val diff = wi - w
      val weight = if (bayes) (alpha+P*0.5f)/(beta + 0.5f*residual) else gamma
      val g = data.par.map(pair => getGradient(pair, wi)).reduce(_+_) - weight*diff
      if (iter > 1) {
        val delta_g = g - g_old
        val beta = g.dot(delta_g)/u.dot(delta_g)
        u = g - u*beta
      }
      else u = g
      g_old = g
      val h_reg = if (bayes) {
        val diff_u = u.dot(diff)
        - weight*(u.dot(u)-diff_u*diff_u/(beta + 0.5f*residual))
      }
      else
        - gamma*u.dot(u)
      val h = - data.par.map(pair => getHessian(pair, wi, u)).reduce(_+_) + h_reg
      val delta_w = g.dot(u)/h*u
      wi = wi - delta_w
      residual = (wi-prior).squaredL2Norm
      val obj_reg = if (bayes) (alpha+P*0.5f)*math.log(1+residual/(2*beta)) else gamma/2*residual
      obj = data.par.map(pair => getLLH(pair, wi)).reduce(_+_) - obj_reg
    }
    
    val gamma_updated = if (bayes) (alpha + 0.5f*P)/(beta + 0.5f*residual) else gamma
    (keyArray, wi.getValues, yi_new.getValues, gamma_updated, residual, iter)
  }
  
  def dist_cg(data : (Array[Int], Array[(Boolean, SparseVector)]), w0: Vector, gamma: Float, 
      max_iter: Int, th: Double, bayes: Boolean): Tuple3[Array[Int], Array[Float], Float] = {
    // initialization of each partition-specific w_i is based on the global weight w
    val keyArray = data._1
    var wi = SparseVector(keyArray, w0)
    val w = SparseVector(keyArray, w0)
    
    var delta_w = SparseVector(keyArray)
    var u = SparseVector(keyArray)
    var g_old = SparseVector(keyArray)
    var iter = 0
    var obj = 0.0
    var obj_old = Double.NegativeInfinity
    while (iter < max_iter && math.abs(obj-obj_old) > th) {
      iter += 1
      obj_old = obj
      val g = data._2.par.map(pair => getGradient(pair, wi)).reduce(_+_) - (wi - w)*gamma
      if (iter > 1) {
        val delta_g = g - g_old
        val beta = g.dot(delta_g)/u.dot(delta_g)
        u = g - u*beta
      }
      else u = g
      g_old = g
      val h = data._2.par.map(pair => getHessian(pair, wi, u)).reduce(_+_)
      delta_w = g.dot(u)/(gamma*u.dot(u) + h)*u
      wi = wi + delta_w
      obj = data._2.par.map(pair => getLLH(pair, wi)).reduce(_+_) + gamma/2*(wi - w).squaredL2Norm
    }
    val gamma_updated = if (bayes)
      (0.5f + 0.5f*wi.size)/(0.5f + 0.5f*wi.squaredL2Dist(w))
      else
        gamma
    (keyArray, wi.getValues, gamma_updated)
  }
  
  def EN_CD(data_colView : Pair[Array[Boolean], Array[(Int, SparseVector)]], 
      w_dist: (Array[Int], Array[Float]), w0: Array[Float], eta: Array[Float], gamma: Float, 
      max_iter: Int, obj_th: Double, l1: Boolean, l2: Boolean, ard: Boolean, warmStart: Boolean)
  : Tuple4[Array[Int], Array[Float], Float, Int] = {
    
    //coordinate descent for elastic net
    val y = data_colView._1
    val x_colView = data_colView._2
    val numData = y.length
    val w_indices = w_dist._1
    val w_values = w_dist._2
    val w_values_old = Array.tabulate(w_values.length)(i => w_values(i))
    val numLocalFeatures = w_indices.length
    assert(numLocalFeatures == x_colView.length)
    val sigma_wx = new Array[Float](numData)
    val residual = new Array[Float](numData)
    val w_updated = SparseVector(w_indices, w_values)
    val w_global = SparseVector(w_indices, Vector(w0))
    var iter = 0
    var obj = -Double.MaxValue
    var obj_old = Double.NegativeInfinity
    
    if (warmStart) {
      // calculating the weight using previous w
      var i = 0
      while (i < numLocalFeatures) {
        // p is the global feature index, i is the local feature index
        val p = x_colView(i)._1
        val isBinary =  x_colView(i)._2.isBinary
        assert(p==w_indices(i), "feature index mismatch")
        val x_p_indices = x_colView(i)._2.getIndices
        var j = 0
        while (j < x_p_indices.length) {
          val n = x_p_indices(j)
          val x_pn = if (isBinary) 1 else x_colView(i)._2.getValues(j)
          sigma_wx(n) += x_pn*w_values(i)
          j += 1
        }
        i += 1
      }
    }
    
    var n = 0
    while (n < numData) {
      sigma_wx(n) = if (warmStart) sigmoid(sigma_wx(n)) else 0.5f
      residual(n) = 0
      n += 1
    }
    while (iter < max_iter && math.abs(obj-obj_old) > obj_th) {
      var innerIter = 0
      var i = 0
      while (i < numLocalFeatures) {
        val p = x_colView(i)._1
        val isBinary =  x_colView(i)._2.isBinary
        assert(p==w_indices(i), "feature index mismatch")
        val x_p_indices = x_colView(i)._2.getIndices
        var nume = 0f
        var deno = 0f
        var j = 0
        while (j < x_p_indices.length) {
          val n = x_p_indices(j)
          val x_np = if (isBinary) 1 else x_colView(i)._2.getValues(j)
          residual(n) += w_values(i)*x_np
          if (y(n)) nume += (1-sigma_wx(n))*x_np*(1+sigma_wx(n)*residual(n))
          else nume += sigma_wx(n)*x_np*(-1+(1-sigma_wx(n))*residual(n))
          if (sigma_wx(n) < 1e-5 || sigma_wx(n) > 1-1e-5) deno += 1e-5f*x_np*x_np
          else deno += sigma_wx(n)*(1-sigma_wx(n))*x_np*x_np
          residual(n) -= w_values(i)*x_np
          j += 1
        }
        if (l2) {
          nume += gamma*w0(p)
          deno += gamma
        }
        else deno += gamma
        w_values(i) = 
          if (l1) {
            if ((nume - eta(p))/deno > w0(p)) 
              (nume - eta(p))/deno
            else if ((nume + eta(p))/deno < w0(p))
              (nume + eta(p))/deno
            else 
              w0(p)
          }
          else nume/deno
          
//        nume += gamma*prior_values(i)
//        deno += gamma
//        
//        val w_values_old = w_values(i)
//        w_values(i) = nume/deno  
            
        if (math.abs(w_values_old(i)-w_values(i)) > 1e-3) {
          var j = 0
          while (j < x_p_indices.length) {
            val n = x_p_indices(j)
            val x_np = if (isBinary) 1 else x_colView(i)._2.getValues(j)
            residual(n) += (w_values_old(i)-w_values(i))*x_np
            j += 1
          }
        }
        i += 1
      }
      n = 0;  while (n<numData) {residual(n) = 0; n += 1}
      
      var reg = if (l2) gamma/2*(w_updated.squaredL2Dist(w_global)) else 0
      if (l1) {
        var i = 0
        while (i < numLocalFeatures) {
          val p = x_colView(i)._1
          reg += eta(p)*math.abs(w_values(i) - w0(p))
          i += 1
       }
      }
      n = 0; while (n < numData) { sigma_wx(n) = 0; n += 1 }      
      i = 0
      while (i < numLocalFeatures) {
        // p is the global feature index, i is the local feature index
        val p = x_colView(i)._1
        assert(p==w_indices(i), "feature index mismatch")
        val x_p_indices = x_colView(i)._2.getIndices
        val isBinary =  x_colView(i)._2.isBinary
        var j = 0
        while (j < x_p_indices.length) {
          val n = x_p_indices(j)
          val x_pn = if (isBinary) 1 else x_colView(i)._2.getValues(j)
          sigma_wx(n) += x_pn*w_values(i)
          j += 1
        }
        w_values_old(i) = w_values(i)
        i += 1
      }
      obj = 0
      n = 0
      while (n < numData) {
        val y_n = if (y(n)) 1 else -1
        //calculate the objective function
        if (y_n*sigma_wx(n) > -5) obj += -math.log(1 + math.exp(-y_n*sigma_wx(n)))
        else obj += y_n*sigma_wx(n)
        sigma_wx(n) = sigmoid(sigma_wx(n))
        n += 1
      }
//      obj -= reg
      iter += 1
//      println("cd iter: " + iter + " new obj: " + obj)
    }
//    if (iter == max_iter)
//      println("this one didn't converge! " + "obj: " + obj + " \t old obj: " + obj_old + "\n")
//    else println("this one converged! Iter: " + iter +  "\n")
    
    val gamma_new = 
      if (ard)
       (1 + 0.5f*numLocalFeatures)/(1 + 0.5f*w_updated.squaredL2Dist(w_global))
      else
        gamma
    (w_indices, w_values, gamma_new, iter)
  }
  
  def updateW_BS(w_p: Seq[(Int, Float)], eta_p: Float, gamma: Array[Float], lambda: Float, 
      l1: Boolean, l2: Boolean) : Float = {
    //a binary search based solution to update global weight parameter w
    val numPartitions = w_p.length
    
    def getObj(w: Float) : Float = {
      // calculate the objective function
      var sum = 0f
      var i = 0
      while (i < numPartitions) {
        val k = w_p(i)._1
        val w_pk = w_p(i)._2
        if (l1) sum += eta_p*math.abs(w-w_pk)
        if (l2) sum += gamma(k)/2*(w-w_pk)*(w-w_pk)
        i += 1
      }
      sum + lambda/2*w*w
    }
    
    var nume = 0f
    var deno = 0f
    var i = 0
    if (l2) {
      while (i < numPartitions) {
        val k = w_p(i)._1
        val w_pk = w_p(i)._2
        nume += w_pk*gamma(k)
        deno += gamma(k)
        i += 1
      }
    }
    deno += lambda
    if (l1) {
      //binary search to find the optimal w
      var d_max = numPartitions
      var d_min = -numPartitions
      var i = 0
      while (d_max > d_min) {
        val d_mid = if (d_min+1==d_max) d_min else (d_max+d_min)/2
        val obj_mid = getObj((nume+eta_p*d_mid)/deno)
        val obj_right = getObj((nume+eta_p*(d_mid+1))/deno)
        if(obj_right >= obj_mid) d_max = d_mid
        else d_min = d_mid+1
        i += 1
//        assert(i <= math.log(numPartitions*2)/math.log(2)+2, 
//            "binary search reached " + i + " iterations! d_min: " + d_min + " d_max: " + d_max + 
//            " numPartitions: " + numPartitions)
      }
      assert(d_min == d_max)
      (nume+eta_p*d_min)/deno
    }
    else {
      nume/deno
    }
  }
  
  //todo: re-write the logistic regression code, using the column sparse format
  def ADMM_CD(data_colView : Pair[Array[Boolean], Array[(Int, SparseVector)]],
      wug_dist: (Array[Int], Array[Float], Array[Float], Float), w0: Array[Float],  gamma: Float,
      alpha: Float, beta: Float, max_iter: Int, bayes: Boolean, warmStart : Boolean, obj_th: Double)
  : Tuple6[Array[Int], Array[Float], Array[Float], Float, Float, Int] = {
   
    //coordinate descent for (B)-ADMM, sparse version
    val y = data_colView._1
    val x_colView = data_colView._2
    val numData = y.length
    val key_indices = wug_dist._1
    val w_values = wug_dist._2
    val w_updated = SparseVector(key_indices, w_values)
    val u_old = SparseVector(key_indices, wug_dist._3)
    val w_global = SparseVector(key_indices, Vector(w0))
    val gamma_old = if (bayes) wug_dist._4 else gamma
    val u_new = if (!bayes && warmStart) u_old + w_updated - w_global else u_old
    val prior = w_global-u_new
    val prior_values = prior.getValues
    val numLocalFeatures = key_indices.length
    assert(numLocalFeatures == x_colView.length)
    val sigma_wx = new Array[Float](numData)
    val residual = new Array[Float](numData)
    var iter = 0
    var obj = -Double.MaxValue
    var obj_old = Double.NegativeInfinity
    var reg = 0f
    if (warmStart) {
      // calculating the weight using previous w
      var i = 0
      while (i < numLocalFeatures) {
        // p is the global feature index, i is the local feature index
        val p = x_colView(i)._1
        assert(p==key_indices(i), "feature index mismatch")
        val x_p_indices = x_colView(i)._2.getIndices
        val isBinary = x_colView(i)._2.isBinary
        var j = 0
        while (j < x_p_indices.length) {
          val n = x_p_indices(j)
          val x_pn = if (isBinary) 1 else x_colView(i)._2.getValues(j)
          sigma_wx(n) += x_pn*w_values(i)
          j += 1
        }
        i += 1
      }
    }
    var n = 0
    while (n < numData) {
      sigma_wx(n) = if (warmStart) sigmoid(sigma_wx(n)) else 0.5f
      //why set the residual to 0 at the beginning?
      residual(n) = 0
      n += 1
    }
    while (iter < max_iter && math.abs(obj-obj_old) > obj_th) {
      obj_old = obj
      var i = 0
      while (i < numLocalFeatures) {
        val p = x_colView(i)._1
        assert(p==key_indices(i), "feature index mismatch")
        val x_p_indices = x_colView(i)._2.getIndices
        val isBinary = x_colView(i)._2.isBinary
        var nume = 0f
        var deno = 0f
        var j = 0
        while (j < x_p_indices.length) {
          val n = x_p_indices(j)
          val x_np = if (isBinary) 1 else x_colView(i)._2.getValues(j)
          residual(n) += w_values(i)*x_np
          
          if (y(n)) nume += (1-sigma_wx(n))*x_np*(1+sigma_wx(n)*residual(n))
          else nume += sigma_wx(n)*x_np*(-1+(1-sigma_wx(n))*residual(n))
          
//          if (y(n)) nume += (1-sigma_wx(n))*sigma_wx(n)*x_np*(1/sigma_wx(n)+residual(n))
//          else nume += (1-sigma_wx(n))*sigma_wx(n)*x_np*(-1/(1-sigma_wx(n))+residual(n))
          
          if (sigma_wx(n) < 1e-5 || sigma_wx(n) > 1-1e-5) deno += 1e-5f*x_np*x_np
          else deno += sigma_wx(n)*(1-sigma_wx(n))*x_np*x_np
          residual(n) -= w_values(i)*x_np
          j += 1
        }
        nume += gamma_old*prior_values(i)
        deno += gamma_old
        
        val w_values_old = w_values(i)
        w_values(i) = nume/deno
        
        if (math.abs(w_values_old-w_values(i)) > 1e-5) {
          var j = 0
          while (j < x_p_indices.length) {
            val n = x_p_indices(j)
            val x_np = if (isBinary) 1 else x_colView(i)._2.getValues(j)
            residual(n) += (w_values_old-w_values(i))*x_np
            j += 1
          }
        }
        i += 1
      }
      
      reg = gamma_old/2*(w_updated.squaredL2Dist(prior))
      //why set the residual to 0 before each iteration starts?
      n = 0; while (n < numData) { sigma_wx(n) = 0; residual(n) = 0; n += 1 }     
      i = 0
      while (i < numLocalFeatures) {
        // p is the global feature index, i is the local feature index
        val p = x_colView(i)._1
        val x_p_indices = x_colView(i)._2.getIndices
        val isBinary = x_colView(i)._2.isBinary
        var j = 0
        while (j < x_p_indices.length) {
          val n = x_p_indices(j)
          val x_pn = if (isBinary) 1 else x_colView(i)._2.getValues(j)
          sigma_wx(n) += x_pn*w_values(i)
          j += 1
        }
        i += 1
      }
      obj = 0
      n = 0
      while (n < numData) {
        val y_n = if (y(n)) 1 else -1
        //calculate the objective function
        if (y_n*sigma_wx(n) > -5) obj += -math.log(1 + math.exp(-y_n*sigma_wx(n)))
        else obj += y_n*sigma_wx(n)
        sigma_wx(n) = sigmoid(sigma_wx(n))
        n += 1
      }
      obj -= reg
      iter += 1
    }
    
    val gamma_new =
      if (bayes)
       (alpha + 0.5f*numLocalFeatures)/(beta + 0.5f*(w_updated.squaredL2Dist(prior)+gamma))
      else
        gamma
    
    obj += reg
    if (bayes) obj -= gamma_new/2*(w_updated.squaredL2Dist(w_global))
    else obj -= u_new.dot(w_updated - w_global)*gamma_new
    
    (key_indices, w_values, u_new.getValues, gamma_new, obj.toFloat, iter)
  }
  
  def ADMM_CD(data_colView : Pair[Array[Boolean], Array[(Int, SparseVector)]],
      wug_dist: (Array[Float], Array[Float], Float), w0: Array[Float], gamma: Float,
      alpha: Float, beta: Float, max_iter: Int, bayes: Boolean, warmStart : Boolean, obj_th: Double)
  : Tuple5[Array[Float], Array[Float], Float, Float, Int] = {
   
    //coordinate descent for (B)-ADMM, non-sparse version
    val y = data_colView._1
    val x_colView = data_colView._2
    val numData = y.length
    val w_values = wug_dist._1
    val w_updated = Vector(w_values)
    val u_old = Vector(wug_dist._2)
    val w_global = Vector(w0)
    val gamma_old = if (bayes) wug_dist._3 else gamma
    val u_new = if (!bayes && warmStart) u_old + w_updated - w_global else u_old
    val prior = w_global-u_new
    val prior_values = prior.elements
    
    val numLocalFeatures = x_colView.length
    val sigma_wx = new Array[Float](numData)
    val residual = new Array[Float](numData)
    var iter = 0
    var obj = -Double.MaxValue
    var obj_old = Double.NegativeInfinity
    var reg = 0f
    if (warmStart) {
      // calculating the weight using previous w
      var i = 0
      while (i < numLocalFeatures) {
        // p is the global feature index, i is the local feature index
        val p = x_colView(i)._1
        val x_p_indices = x_colView(i)._2.getIndices
        var j = 0
        val isBinary = x_colView(i)._2.isBinary
        while (j < x_p_indices.length) {
          val n = x_p_indices(j)
          val x_np = if (isBinary) 1 else x_colView(i)._2.getValues(j)
          sigma_wx(n) += x_np*w_values(p)
          j += 1
        }
        i += 1
      }
    }
    
    var n = 0
    while (n < numData) {
      sigma_wx(n) = if (warmStart) sigmoid(sigma_wx(n)) else 0.5f
      residual(n) = 0
      n += 1
    }
    
    while (iter < max_iter && math.abs(obj-obj_old) > obj_th) {
      obj_old = obj
      var i = 0
      while (i < numLocalFeatures) {
        val p = x_colView(i)._1
        val x_p_indices = x_colView(i)._2.getIndices
        val isBinary = x_colView(i)._2.isBinary
        var nume = 0f
        var deno = 0f
        var j = 0
        while (j < x_p_indices.length) {
          val n = x_p_indices(j)
          val x_np = if (isBinary) 1 else x_colView(i)._2.getValues(j)
          residual(n) += w_values(p)*x_np
          if (y(n)) nume += (1-sigma_wx(n))*x_np*(1+sigma_wx(n)*residual(n))
          else nume += sigma_wx(n)*x_np*(-1+(1-sigma_wx(n))*residual(n))
          if (sigma_wx(n) < 1e-5 || sigma_wx(n) > 1-1e-5) deno += 1e-5f*x_np*x_np
          else deno += sigma_wx(n)*(1-sigma_wx(n))*x_np*x_np
          residual(n) -= w_values(p)*x_np
          j += 1
        }
        nume += gamma_old*prior_values(p)
        deno += gamma_old
       
        val w_values_old = w_values(p)
        w_values(p) = nume/deno
        if (math.abs(w_values_old-w_values(p)) > 1e-5) {
          var j = 0
          while (j < x_p_indices.length) {
            val n = x_p_indices(j)
            val x_np = if (isBinary) 1 else x_colView(i)._2.getValues(j)
            residual(n) += (w_values_old-w_values(p))*x_np
            j += 1
          }
        }
        i += 1
      }
      
      reg = gamma_old/2*(w_updated.squaredDist(prior))
      n = 0; while (n < numData) { sigma_wx(n) = 0; residual(n) = 0; n += 1 }     
      i = 0
      while (i < numLocalFeatures) {
        // p is the global feature index, i is the local feature index
        val p = x_colView(i)._1
        val x_p_indices = x_colView(i)._2.getIndices
        val isBinary = x_colView(i)._2.isBinary
        var j = 0
        while (j < x_p_indices.length) {
          val n = x_p_indices(j)
          val x_np = if (isBinary) 1 else x_colView(i)._2.getValues(j)
          sigma_wx(n) += x_np*w_values(p)
          j += 1
        }
        i += 1
      }
      obj = 0
      n = 0
      while (n < numData) {
        val y_n = if (y(n)) 1 else -1
        //calculate the objective function
        if (y_n*sigma_wx(n) > -5) obj += -math.log(1 + math.exp(-y_n*sigma_wx(n)))
        else obj += y_n*sigma_wx(n)
        sigma_wx(n) = sigmoid(sigma_wx(n))
        n += 1
      }
      obj -= reg
      assert(!obj.isInfinity && !obj.isNaN(), "obj is inf or nan")
      iter += 1
    }
    
    val gamma_new =
      if (bayes)
       (alpha + 0.5f*numLocalFeatures)/(beta + 0.5f*(w_updated.squaredDist(prior)+gamma))
      else
        gamma
        
    obj += reg
    if (bayes) obj -= gamma_new/2*(w_updated.squaredDist(w_global))
    else obj -= u_new.dot(w_updated - w_global)*gamma_new
    (w_values, u_new.elements, gamma_new, obj.toFloat, iter)
  }
  
  def BDL_CD(data_colView : Pair[Array[Boolean], Array[(Int, SparseVector)]],
      wg_dist: (Array[Int], Array[Float], Array[Float]), w0: Array[Float], variance: Array[Float],
      alpha: Float, beta: Float, max_iter: Int, gamma0: Float, 
      bayes: Boolean, warmStart : Boolean, obj_th: Double)
  : Tuple5[Array[Int], Array[Float], Array[Float], Float, Int] = {
   
    //coordinate descent for BDL, sparse version
    val y = data_colView._1
    val x_colView = data_colView._2
    val numData = y.length
    val key_indices = wg_dist._1
    val numLocalFeatures = key_indices.length
    assert(numLocalFeatures == x_colView.length)
    val w_values = wg_dist._2
    val gamma = if (bayes && warmStart) wg_dist._3 else Array.tabulate(numLocalFeatures)(_=>gamma0)
    val w_updated = SparseVector(key_indices, w_values)
    
    val sigma_wx = new Array[Float](numData)
    val residual = new Array[Float](numData)
    var iter = 0
    var obj = -Double.MaxValue
    var obj_old = Double.NegativeInfinity
   
    if (warmStart) {
      // calculating the weight using previous w
      var i = 0
      while (i < numLocalFeatures) {
        // p is the global feature index, i is the local feature index
        val p = x_colView(i)._1
        assert(p==key_indices(i), "feature index mismatch")
        val x_p_indices = x_colView(i)._2.getIndices
        val x_p_values = x_colView(i)._2.getValues
        var j = 0
        while (j < x_p_indices.length) {
          val n = x_p_indices(j)
          val x_pn = x_p_values(j)
          sigma_wx(n) += x_pn*w_values(i)
          j += 1
        }
        i += 1
      }
    }
   
    var n = 0
    while (n < numData) {
      sigma_wx(n) = if (warmStart) sigmoid(sigma_wx(n)) else 0.5f
      residual(n) = 0
      n += 1
    }
    
    while (iter < max_iter && math.abs(obj-obj_old) > obj_th) {
      obj_old = obj
      var i = 0
      while (i < numLocalFeatures) {
        val p = x_colView(i)._1
        assert(p==key_indices(i), "feature index mismatch")
        val x_p_indices = x_colView(i)._2.getIndices
        val x_p_values = x_colView(i)._2.getValues
        var nume = 0f
        var deno = 0f
        var j = 0
        while (j < x_p_indices.length) {
          val n = x_p_indices(j)
          val x_np = x_p_values(j)
          residual(n) += w_values(i)*x_np
          if (y(n)) nume += (1-sigma_wx(n))*x_np*(1+sigma_wx(n)*residual(n))
          else nume += sigma_wx(n)*x_np*(-1+(1-sigma_wx(n))*residual(n))
          if (sigma_wx(n) < 1e-5 || sigma_wx(n) > 1-1e-5) deno += 1e-5f*x_np*x_np
          else deno += sigma_wx(n)*(1-sigma_wx(n))*x_np*x_np
          residual(n) -= w_values(i)*x_np
          j += 1
        }
        nume += gamma(i)*w0(key_indices(i))
        deno += gamma(i)
        
        val w_values_old = w_values(i)
        w_values(i) = nume/deno
       
        if (math.abs(w_values_old-w_values(i)) > 1e-3) {
          var j = 0
          while (j < x_p_indices.length) {
            val n = x_p_indices(j)
            val x_np = x_p_values(j)
            residual(n) += (w_values_old-w_values(i))*x_np
            j += 1
          }
        }
        i += 1
      }

      obj = 0
      n = 0; while (n < numData) { sigma_wx(n) = 0; residual(n) = 0; n += 1 }     
      i = 0
      while (i < numLocalFeatures) {
        // p is the global feature index, i is the local feature index
        val p = x_colView(i)._1
        val x_p_indices = x_colView(i)._2.getIndices
        val x_p_values = x_colView(i)._2.getValues
        var j = 0
        while (j < x_p_indices.length) {
          val n = x_p_indices(j)
          val x_pn = x_p_values(j)
          sigma_wx(n) += x_pn*w_values(i)
          j += 1
        }
        val res = w_values(i) - w0(key_indices(i))
        obj -= gamma(i)/2*(res*res)
        i += 1
      }
      
      n = 0
      while (n < numData) {
        val y_n = if (y(n)) 1 else -1
        //calculate the objective function
        if (y_n*sigma_wx(n) > -5) obj += -math.log(1 + math.exp(-y_n*sigma_wx(n)))
        else obj += y_n*sigma_wx(n)
        sigma_wx(n) = sigmoid(sigma_wx(n))
        n += 1
      }
      iter += 1
    }
    
    if (bayes) {
      var i = 0
      var tmp = 0f
      while(i < numLocalFeatures) {
        val res = w_values(i)-w0(key_indices(i))
//        tmp += res*res+variance(key_indices(i))
        gamma(i) = (alpha + 0.5f)/(beta + 0.5f*(res*res+variance(key_indices(i))))
        i += 1
      }
//      i = 0
//      while(i < numLocalFeatures) {
//        gamma(i) = (alpha + 0.5f*numLocalFeatures)/(beta + 0.5f*tmp)
//        i += 1 
//      }
    }
    (key_indices, w_values, gamma, obj.toFloat, iter)
  }
  
//  def BDL_CG(data : Array[(Boolean, SparseVector)],
//      wg_dist: (Array[Int], Array[Float], Array[Float]), w0: Array[Float], variance: Array[Float],
//      alpha: Float, beta: Float, max_iter: Int, gamma0: Float, 
//      bayes: Boolean, warmStart : Boolean, obj_th: Double)
//  : Tuple5[Array[Int], Array[Float], Array[Float], Float, Int] = {
//   
//    //conjugate gradient descent for BDL
//    val keyArray = wg_dist._1
//    val numLocalFeatures = keyArray.length
//    var wi = SparseVector(keyArray, wg_dist._2)
//    val w = SparseVector(keyArray, Vector(w0))
//    val gamma = 
//      if (bayes && warmStart) 
//        SparseVector(keyArray, Vector(wg_dist._3)) 
//      else 
//        SparseVector(keyArray, gamma0)
//    
//    var delta_w = SparseVector(keyArray)
//    var u = SparseVector(keyArray)
//    var g_old = SparseVector(keyArray)
//    var iter = 0
//    var obj = -Double.MaxValue
//    var obj_old = Double.NegativeInfinity
//    while (iter < max_iter && math.abs(obj-obj_old) > obj_th) {
//      iter += 1
//      obj_old = obj
//      val g = data.par.map(pair => getGradient(pair, wi)).reduce(_+_) - (wi - w)*gamma
//      if (iter > 1) {
//        val delta_g = g - g_old
//        val beta = g.dot(delta_g)/u.dot(delta_g)
//        u = g - u*beta
//      }
//      else u = g
//      g_old = g
//      val h = data.par.map(pair => getHessian(pair, wi, u)).reduce(_+_) + gamma*u.dot(u)
//      delta_w = g.dot(u)/h*u
//      wi = wi + delta_w
//      obj = data.par.map(pair => getLLH(pair, wi)).reduce(_+_) - gamma/2*(wi - prior).squaredL2Norm
//    }
//    
//    val dist = if (bayes) wi.squaredL2Dist(w) else 0
//    val gamma_new =
//      if (bayes)
//       (alpha + 0.5f*wi.size)/(beta + 0.5f*(dist+gamma0))
//      else
//        gamma
//    (keyArray, wi.getValues, gamma, obj.toFloat, iter)
//  }
  
  def BDL_CD(data_colView : Pair[Array[Boolean], Array[(Int, SparseVector)]],
      wg_dist: (Array[Float], Array[Float]), w0: Array[Float], variance: Array[Float],
      alpha: Float, beta: Float, max_iter: Int, gamma0: Float, 
      bayes: Boolean, warmStart : Boolean, obj_th: Double)
  : Tuple4[Array[Float], Array[Float], Float, Int] = {
   
    //coordinate descent for BDL
    val y = data_colView._1
    val x_colView = data_colView._2
    val numData = y.length
    val w_values = wg_dist._1
    val numFeatures = w_values.length
    val numLocalFeatures = x_colView.length
    val gamma = if (bayes && warmStart) wg_dist._2 else Array.tabulate(numFeatures)(_=>gamma0)
    val w_updated = Vector(w_values)
    
    val sigma_wx = new Array[Float](numData)
    val residual = new Array[Float](numData)
    var iter = 0
    var obj = -Double.MaxValue
    var obj_old = Double.NegativeInfinity
   
    if (warmStart) {
      // calculating the weight using previous w
      var i = 0
      while (i < numLocalFeatures) {
        // p is the global feature index, i is the local feature index
        val p = x_colView(i)._1
        val x_p_indices = x_colView(i)._2.getIndices
        val isBinary = x_colView(i)._2.isBinary
        var j = 0
        while (j < x_p_indices.length) {
          val n = x_p_indices(j)
          val x_pn = if (isBinary) 1 else x_colView(i)._2.getValues(j)
          sigma_wx(n) += x_pn*w_values(p)
          j += 1
        }
        i += 1
      }
    }
   
    var n = 0
    while (n < numData) {
      sigma_wx(n) = if (warmStart) sigmoid(sigma_wx(n)) else 0.5f
      residual(n) = 0
      n += 1
    }
   
    while (iter < max_iter && math.abs(obj-obj_old) > obj_th) {
      obj_old = obj
      var i = 0
      while (i < numLocalFeatures) {
        val p = x_colView(i)._1
        val x_p_indices = x_colView(i)._2.getIndices
        val isBinary = x_colView(i)._2.isBinary
        var nume = 0f
        var deno = 0f
        var j = 0
        while (j < x_p_indices.length) {
          val n = x_p_indices(j)
          val x_np = if (isBinary) 1 else x_colView(i)._2.getValues(j)
          residual(n) += w_values(p)*x_np
          if (y(n)) nume += (1-sigma_wx(n))*x_np*(1+sigma_wx(n)*residual(n))
          else nume += sigma_wx(n)*x_np*(-1+(1-sigma_wx(n))*residual(n))
          if (sigma_wx(n) < 1e-5 || sigma_wx(n) > 1-1e-5) deno += 1e-5f*x_np*x_np
          else deno += sigma_wx(n)*(1-sigma_wx(n))*x_np*x_np
          residual(n) -= w_values(p)*x_np
          j += 1
        }
        nume += gamma(p)*w0(p)
        deno += gamma(p)
       
        val w_values_old = w_values(p)
        w_values(p) = nume/deno
       
        if (math.abs(w_values_old-w_values(p)) > 1e-3) {
          var j = 0
          while (j < x_p_indices.length) {
            val n = x_p_indices(j)
            val x_np = if (isBinary) 1 else x_colView(i)._2.getValues(j)
            residual(n) += (w_values_old-w_values(p))*x_np
            j += 1
          }
        }
        i += 1
      }
      
      obj = 0
      n = 0; while (n < numData) { sigma_wx(n) = 0; residual(n) = 0; n += 1 }     
      i = 0
      while (i < numLocalFeatures) {
        // p is the global feature index, i is the local feature index
        val p = x_colView(i)._1
        val x_p_indices = x_colView(i)._2.getIndices
        val isBinary = x_colView(i)._2.isBinary
        var j = 0
        while (j < x_p_indices.length) {
          val n = x_p_indices(j)
          val x_pn = if (isBinary) 1 else x_colView(i)._2.getValues(j)
          sigma_wx(n) += x_pn*w_values(p)
          j += 1
        }
        val res = w_values(p) - w0(p)
        obj -= gamma(p)/2*(res*res)
        i += 1
      }
      
      n = 0
      while (n < numData) {
        val y_n = if (y(n)) 1 else -1
        //calculate the objective function
        if (y_n*sigma_wx(n) > -5) obj += -math.log(1 + math.exp(-y_n*sigma_wx(n)))
        else obj += y_n*sigma_wx(n)
        sigma_wx(n) = sigmoid(sigma_wx(n))
        n += 1
      }
      iter += 1
    }
    
    if (bayes) {
      var p = 0
      var tmp = 0f
      while(p < numFeatures) {
        val res = w_values(p)-w0(p)
        gamma(p) = (alpha + 0.5f)/(beta + 0.5f*(res*res+variance(p)))
        p += 1
      }
    }
    (w_values, gamma, obj.toFloat, iter)
  }
}
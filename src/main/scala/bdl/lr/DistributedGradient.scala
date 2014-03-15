package lr

import org.apache.spark.{HashPartitioner, Partitioner, SparkContext}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.rdd._
import org.apache.spark.serializer.KryoRegistrator
import org.apache.spark.SparkContext._
import utilities.SparseVector

class DistributedGradient {
  
}


object DistributedGradient {
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
}
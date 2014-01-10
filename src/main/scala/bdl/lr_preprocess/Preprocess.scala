package lr_preprocess

import org.apache.spark.{SparkContext, rdd}
import org.apache.spark.SparkContext._
import scala.collection.mutable.HashMap
import utilities._

object Preprocess {
  
  def parseLRLine(line : String, featureMap : Map[Int, Int], binary : Boolean)
    : (Boolean, SparseVector) = {
    
    var keyList : List[Int] = List()
    var valueList : List[Float] = List()
    var response = true
    val featureSet = featureMap.keySet
    line.split(" ").foreach(token => { 
      token.split(":") match {
        case Array(id, value) => {
          val featureID = id.toInt
          if (featureSet.contains(featureID)) {
            keyList = (featureMap(featureID)+1) :: keyList
            valueList = value.toFloat :: valueList
          }
        }
        case Array(value) => {
          response = if (value.startsWith("+")) true
          else if (value.startsWith("-")) false
          else if (value.startsWith("0")) false
          else true
        }
      }
    })
    
    // adding the offset
    keyList = 0 :: keyList
    if (!binary) valueList = 1 :: valueList
    if (!binary) (response, SparseVector(keyList.toArray, valueList.toArray, false))
    else (response, SparseVector(keyList.toArray, false))
  }
  
  def wordCountKDD2010(data : rdd.RDD[String]) = {
    data.flatMap(line => line.split(" ")).filter(token => token.length>2)
    .map(token => (token.split(":")(0).toInt, 1)).reduceByKey(_+_)
  }
  
  def parseKDD2010Line(line : String, featureMap : Map[Int, Int], binary : Boolean = false) 
    : (Boolean, SparseVector) = {
    
    val parts = line.split(" ")
    val response = if (parts(0).startsWith("+")) true
          else if (parts(0).startsWith("-")) false
          else if (parts(0).startsWith("0")) false
          else true
    val featureSet = featureMap.keySet
    var keyList : List[Int] = List()
    var valueList : List[Float] = List()
    var i = 1
    while (i < parts.length) {
      val token = parts(i).split(":")
      val featureID = token(0).toInt
      if (featureSet.contains(featureID)) {
        keyList = (featureMap(featureID)+1) :: keyList
        if (!binary && token.length>1) valueList = token(1).toFloat :: valueList
      }
      i+=1
    }
    // adding the offset
    keyList = 0 :: keyList
    if (!binary && valueList.size>0) valueList = 1 :: valueList
    if (!binary && valueList.size>0) (response, SparseVector(keyList.toArray, valueList.toArray, false))
    else (response, SparseVector(keyList.toArray, false))
  }
  
  def wordCountAds(data : rdd.RDD[String]) = {
    data.map(line => {
      val tokens = line.split("\t")
      if (tokens.length != 4) {System.err.println(line); ""}
      else tokens(3)}
    ).flatMap(line => line.split(" ")).map((_, 1)).reduceByKey(_+_)
  }
  
  def parseAdsLine(line : String, featureMap : Map[String, Int]) : Pair[Boolean, SparseVector] = {
    var keyList : List[Int] = List()
    val parts = line.split("\t")
    val featureSet = featureMap.keySet
    val response = parts(1).toInt > 0
    if (parts.length != 4) {
      System.err.println("unexpected record: " + line)
      return null
    }
    val tokens = parts(3).split(" ")
    val feature = Array.newBuilder[Int]
    var i = 0
    while (i < tokens.length) {
      if (featureSet.contains(tokens(i))) {
        keyList = (featureMap(tokens(i))+1) :: keyList
      }
      i+=1
    }
    (response, SparseVector(keyList.toArray, false))
  }
  
  def toColView(data : Array[(Boolean, SparseVector)]) 
    : Pair[Array[Boolean], Array[(Int, SparseVector)]] = {
    val colViewMap = new HashMap[Int, (List[Int], List[Float])]
    val numData = data.length
    val responsesArray = new Array[Boolean](numData)
    var n = 0
    while(n < numData) {
      val indices = data(n)._2.getIndices
      val values = data(n)._2.getValues
      var i = 0
      while (i<indices.length) {
        val p = indices(i)
        val value = values(i)
        val colViewPairs = colViewMap.getOrElse(p, (scala.Nil, scala.Nil))
        colViewMap.put(p, (n::colViewPairs._1, value::colViewPairs._2))
        i += 1
      }
      responsesArray(n) = data(n)._1
      n += 1
    }
    val colViewArray = colViewMap.map(pair => 
      (pair._1, SparseVector(pair._2._1.toArray, pair._2._2.toArray))).toArray
    
    def quickSort(pairs : Array[(Int, SparseVector)]) 
      : Array[(Int, SparseVector)] = {
      if (pairs.length <= 1) pairs
      else {
        val pivot = pairs(pairs.length/2)
        Array.concat(quickSort(pairs.filter(pair => pair._1 < pivot._1)), 
            pairs.filter(pair => pair._1 == pivot._1), 
            quickSort(pairs.filter(pair => pair._1 > pivot._1)))
      }
    }
    (responsesArray, quickSort(colViewArray))
  }
  
  
  def toColViewBinary(data : Array[(Boolean, SparseVector)]) 
    : Pair[Array[Boolean], Array[(Int, SparseVector)]] = {
    val colViewMap = new HashMap[Int, List[Int]]
    val numData = data.length
    val responsesArray = new Array[Boolean](numData)
    var n = 0
    while(n < numData) {
      val indices = data(n)._2.getIndices
      val values = data(n)._2.getValues
      var i = 0
      while (i<indices.length) {
        val p = indices(i)
        val value = values(i)
        val colViewPairs = colViewMap.getOrElse(p, (scala.Nil))
        colViewMap.put(p, n::colViewPairs)
        i += 1
      }
      responsesArray(n) = data(n)._1
      n += 1
    }
    val colViewArray = colViewMap.map(pair => (pair._1, SparseVector(pair._2.toArray))).toArray
    
    def quickSort(pairs : Array[(Int, SparseVector)]) 
      : Array[(Int, SparseVector)] = {
      if (pairs.length <= 1) pairs
      else {
        val pivot = pairs(pairs.length/2)
        Array.concat(quickSort(pairs.filter(pair => pair._1 < pivot._1)), 
            pairs.filter(pair => pair._1 == pivot._1), 
            quickSort(pairs.filter(pair => pair._1 > pivot._1)))
      }
    }
    (responsesArray, quickSort(colViewArray))
  }
}
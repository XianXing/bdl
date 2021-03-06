package preprocess

import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.{Partitioner, SparkContext}
import org.apache.spark.broadcast.Broadcast

import utilities.SparseVector
import utilities.SparseMatrix

//preprocess for the logistic regression model
object LR2 {
  
  def toSparseVector(inputDir: String, sc: SparkContext, 
      featureMap: Broadcast[Map[Int,Int]], numSlices: Int, 
      featureThre: Int, isBinary: Boolean, isSeq: Boolean, fraction: Float = 1,
      seed: Int = 1): RDD[(Int, (Byte, SparseVector))]= {
    isSeq match {
      case true => {
        sc.objectFile[Array[Int]](inputDir).sample(false, fraction, seed).map(arr => {
          val bid = (math.random*numSlices).toInt
          if (featureThre > 0) (bid, parseArr(arr, featureMap.value))
          else (bid, parseArr(arr))
        })
      }
      case false => {
        sc.textFile(inputDir).sample(false, fraction, seed).map(line => {
          val bid = (math.random*numSlices).toInt
          if (featureThre > 0) (bid, parseLine(line, featureMap.value, isBinary))
          else (bid, parseLine(line, isBinary))
        })
      }
    }
  }
  
  def toSparseMatrix(inputDir: String, sc: SparkContext, 
      featureMap: Broadcast[Map[Int,Int]], part: Partitioner, numSlices: Int, 
      featureThre: Int, isBinary: Boolean, isSeq: Boolean, fraction: Float = 1,
      seed: Int = 1): RDD[(Int, (Array[Byte], SparseMatrix))]= {
    
    toSparseVector(inputDir, sc, featureMap, numSlices, featureThre, isBinary, isSeq,
      fraction, seed)
    .groupByKey(part)
    .mapValues(seq => (seq.map(_._1).toArray, SparseMatrix(seq.map(_._2).toArray)))
  }
  
  def parseResponse(string: String): Byte = {
    if (string.startsWith("+")) 1
    else if (string.startsWith("-")) -1
    else if (string.startsWith("0")) -1
    else 1
  }
  
  def parseLine(line: String, featureMap: Map[Int, Int], binary: Boolean)
    : (Byte, SparseVector) = {
    
    val parts = line.split(" ")
    val response = parseResponse(parts(0))
    var key : List[Int] = Nil
    var value : List[Float] = Nil
    // adding the offset
    key = 0 :: key
    if (!binary) value = 1 :: value
    var i = 1
    while (i < parts.length) {
      val token = parts(i).split(":")
      val featureID = token(0).toInt
      if (featureMap.contains(featureID)) {
        key = (featureMap(featureID)+1) :: key
        if (!binary && token.length>1) value = token(1).toFloat :: value
      }
      i+=1
    }
    if (binary) (response, SparseVector(key.toArray, false))
    else (response, SparseVector(key.toArray, value.toArray, false))
  }
  
  def parseLine(line: String, binary: Boolean): (Byte, SparseVector) = {
    
    val parts = line.split(" ")
    val response = parseResponse(parts(0))
    val key = new Array[Int](parts.length)
    val value = if (binary) null else new Array[Float](parts.length)
    //adding the offset
    key(0) = 0
    if (!binary) value(0) = 1
    var i = 1
    while (i < parts.length) {
      val token = parts(i).split(":")
      key(i) = token(0).toInt+1 //+1 because of the offset (bias)
      if (!binary && token.length>1) value(i) = token(1).toFloat
      i+=1
    }
    if (binary) (response, SparseVector(key, false))
    else (response, SparseVector(key, value, false))
  }
  
  def countLine(line : String, weight: Int = 1) = {
    val parts = line.split(" ")
    val response = parseResponse(parts(0))
    val length = parts.length
    val results = new Array[(Int, Int)](length - 1)
    var i = 1
    while (i < length) {
      val token = parts(i).split(":")
      if (response == 1) results(i-1) = (token(0).toInt, weight)
      else results(i-1) = (token(0).toInt, 1)
      i+=1
    }
    results
  }
  
  def countArr(arr : Array[Int], weight: Int = 1) = {
    val length = arr.length
    val results = new Array[(Int, Int)](length - 1)
    val response = arr.last == 1
    var i = 0
    while (i < length - 1) {
      if (response) results(i) = (arr(i), weight)
      else results(i) = (arr(i), 1)
      i+=1
    }
    results
  }
  
  def parseArr(arr: Array[Int], featureMap : Map[Int, Int]) 
    : (Byte, SparseVector) = {
    val response = arr.last.toByte
    val length = arr.length
    var key : List[Int] = Nil
    var i = 0
    while (i < length - 1) {
      if (featureMap.contains(arr(i))) key = featureMap(arr(i)) :: key
      i+=1
    }
    (response, SparseVector(key.toArray))
  }
  
  def parseArr(arr: Array[Int]) : (Byte, SparseVector) = {
    val response = arr.last.toByte
    (response, SparseVector(arr.dropRight(1)))
  }
}
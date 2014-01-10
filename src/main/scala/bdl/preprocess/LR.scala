package preprocess

import utilities._

import org.apache.spark.{storage, Partitioner, SparkContext, rdd, broadcast}
import org.apache.spark.SparkContext._
import org.apache.hadoop.io.NullWritable

import java.io._
import java.text.DecimalFormat
import scala.collection.mutable.{HashMap, HashSet}
import scala.collection.JavaConversions._
import scala.io._
import scala.util.Random

//preprocess for the logistic regression model
object LR {
  
  def parseResponse(string: String) = {
    if (string.startsWith("+")) true
    else if (string.startsWith("-")) false
    else if (string.startsWith("0")) false
    else true
  }
    
  def parseLine(line : String, featureMap : Map[Int, Int], binary : Boolean = false)
    : (Boolean, SparseVector) = {
    
    val parts = line.split(" ")
    val response = parseResponse(parts(0))
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
    if (!binary && valueList.size>0) 
      (response, SparseVector(keyList.toArray, valueList.toArray, false))
    else (response, SparseVector(keyList.toArray, false))
  }
  
  def count(line : String, weight: Int = 1) = {
    val parts = line.split(" ")
    val response = parseResponse(parts(0))
    val length = parts.length
    val results = new Array[(Int, Int)](length - 1)
    var i = 1
    while (i < length) {
      val token = parts(i).split(":")
      if (response) results(i-1) = (token(0).toInt, 1)
      else results(i-1) = (token(0).toInt, weight)
      i+=1
    }
    results
  }
  
  def main(args: Array[String]) {
    
    val master = "local[2]"
    val jar = Seq("sparkproject.jar")
    val jobName = "preprocess_TF"
//    val inputDir = "../datasets/MovieLens/ml-1m/ra.train"
    val inputDir = "../datasets/MovieLens/ml-1m/ra.test"
    val outputDir = "input/ml-1m/mf_test"
    val sc = new SparkContext(master, jobName, System.getenv("SPARK_HOME"), jar)
  }
}
package preprocess

import java.io._
import java.text.DecimalFormat

import scala.collection.mutable.{HashMap, ArrayBuilder}
import scala.io._
import scala.util.Random

import org.apache.spark.{storage, Partitioner, SparkContext, rdd, broadcast}
import org.apache.spark.SparkContext._
import org.apache.hadoop.io.NullWritable

import utilities._


//preprocess for the logistic regression model
object LR {
  
  def parseResponse(string: String) = {
    if (string.startsWith("+")) true
    else if (string.startsWith("-")) false
    else if (string.startsWith("0")) false
    else true
  }
    
  def parseLine(line : String, featureMap : Map[Int, Int], binary : Boolean)
    : (Boolean, SparseVector) = {
    
    val parts = line.split(" ")
    val response = parseResponse(parts(0))
    val key = new ArrayBuilder.ofInt
    key.sizeHint(parts.length)
    val value = new ArrayBuilder.ofFloat
    if (!binary) value.sizeHint(parts.length)
    // adding the offset
    key += 0
    if (!binary) value += 1
    var i = 1
    while (i < parts.length) {
      val token = parts(i).split(":")
      val featureID = token(0).toInt
      if (featureMap.contains(featureID)) {
        key += (featureMap(featureID)+1)
        if (!binary && token.length>1) value += token(1).toFloat
      }
      i+=1
    }
    if (binary) (response, SparseVector(key.result, false))
    else (response, SparseVector(key.result, value.result, false))
  }
  
  def parseLine(line : String, binary : Boolean) : (Boolean, SparseVector) = {
    
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
      key(i) = token(0).toInt
      if (!binary && token.length>1) value(i) = token(1).toFloat
      i+=1
    }
    if (binary) (response, SparseVector(key, false))
    else (response, SparseVector(key, value, false))
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
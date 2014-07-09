package preprocess

import java.io.File
import java.util.Random

import scala.io.Source
import scala.collection.mutable.ArrayBuffer

import breeze.linalg._
import breeze.collection.mutable.SparseArray

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, Partitioner, SparkConf}
import org.apache.spark.broadcast.Broadcast

import org.apache.commons.math3.distribution.GammaDistribution

import tm.Document

object TM {
  
  def hash(x: Int): Int = {
    val r = x ^ (x >>> 20) ^ (x >>> 12)
    r ^ (r >>> 7) ^ (r >>> 4)
  }
  
  def toSparseVector(line: String): SparseVector[Int] = {
    val tokens = line.split(" ")
    val length = tokens.length
    val indexes = new Array[Int](length)
    val data = new Array[Int](length)
    var l = 0
    while (l < length) {
      val pair = tokens(l).split(":")
      indexes(l) = pair(0).toInt
      data(l) = pair(1).toInt
      l += 1
    }
    new SparseVector(new SparseArray[Int](indexes, data, length, length, 1))
  }
  
  def toDocument(line: String, numTopics: Int): Document = {
    val content = toSparseVector(line)
    new Document(content, DenseVector.ones[Double](numTopics))
  }
  
  def toCorpus(inputDir: String, numTopics: Int, seed: Int): Array[Document] = {
    val gd = new GammaDistribution(100, 1./100)
    gd.reseedRandomGenerator(seed)
    val builder = new ArrayBuffer[Document]
    val files = new File(inputDir).listFiles
      .filter(file => !file.getName().endsWith(".crc") && file.length > 0)
    for (file <- files) {
      val length = file.length()
      val lines = Source.fromFile(file).getLines.toArray
      for (line <- lines) {
        val content = toSparseVector(line)
        val gamma = DenseVector.fill(numTopics)(gd.sample)
        builder += new Document(content, gamma)
      }
    }
    builder.toArray
  }
  
  def toCorpus(contents: Array[SparseVector[Int]], numTopics: Int)
    : Array[Document] = {
    contents.map(content => new Document(content, DenseVector.ones[Double](numTopics)))
  }
  
  def toCSCMatrix(inputDocsPath: String): CSCMatrix[Int] = {
    val lines = Source.fromFile(inputDocsPath).getLines
    val numDocs = lines.next.toInt
    val numWords = lines.next.toInt
    val nnz = lines.next.toInt
    val docsBuilder = new CSCMatrix.Builder[Int](rows=numWords, cols=numDocs)
    for (line <- lines) {
      val tokens = line.split(" ")
      docsBuilder.add(tokens(1).toInt - 1, tokens(0).toInt - 1, tokens(2).toInt)
    }
    docsBuilder.result
  }
  
  def preprocessBagOfWords(inputPath: String, outputDir: String, jarPath: String,
      numCores: Int, trainRatio: Double) {
    val conf = new SparkConf()
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//      .set("spark.kryo.registrator",  classOf[utilities.Registrator].getName)
      .set("spark.kryo.referenceTracking", "false")
      .set("spark.kryoserializer.buffer.mb", "8")
      .set("spark.locality.wait", "10000")
      .set("spark.akka.frameSize", "64")
      .setJars(Seq(jarPath))
    val mode = "local[" + numCores + "]"
    val jobName = "preprocessing bag-of-words dataset"
    val sc = new SparkContext(mode, jobName, conf)
    val all = sc.textFile(inputPath, numCores).map(_.split(" ")).filter(_.length == 3)
      .map(triple => (triple(0).toInt-1, (triple(1).toInt-1, triple(2))))
      .groupByKey(numCores*2).map(_._2)
      .map(_.map{case(w, n) => w + ":" + n + " "}.reduce(_+_))
      .mapPartitionsWithIndex{case (pid, docs) =>
        val rand = new Random()
        rand.setSeed(pid)
        docs.map((rand.nextDouble() < trainRatio, _))
      }.persist(StorageLevel.MEMORY_AND_DISK)
    
    val trainingPath = outputDir + "train"
    all.filter(_._1).map(_._2).saveAsTextFile(trainingPath)
    val validatingPath = outputDir + "validate"
    all.filter(!_._1).map(_._2).saveAsTextFile(validatingPath)
  }
  
  def main(args : Array[String]) = {
    val inputPath = args(0)
    val outputDir = args(1)
    val jarPath = args(2)
    val numCores = args(3).toInt
    val trainingRatio = args(4).toDouble
//    val prefix = "/Users/xianxingzhang/Documents/workspace/datasets/Bags_Of_Words/"
//    val inputPath = prefix + "docword.nips.txt"
//    val outputDir = prefix + "nips_processed/"
//    val jarPath = "sparkproject.jar"
//    val numCores = 2
//    val trainingRatio = 0.8
    preprocessBagOfWords(inputPath, outputDir, jarPath, numCores, trainingRatio)
  }
}
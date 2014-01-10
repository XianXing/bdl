package lr_preprocess

import utilities._
import org.apache.spark.{SparkContext, storage, HashPartitioner}
import org.apache.spark.SparkContext._
import java.io._

object PreprocessRawAdsData {
  val JOB_NAME = "preprocess Ads data"
  val JARS = Seq("sparkproject.jar")
  def main (args : Array[String]) {
    val currentTime = System.currentTimeMillis();
    
    val mode = args(0)
    val numCores = args(1).toInt
    val trainingInputPath = args(2)
    val testingInputPath = args(3)
    val trainingOutputPath = args(4)
    val testingOutputPath = args(5)
    val freqPath = args(6)
    val histPath = args(7)
    System.setProperty("spark.ui.port", "44717")
    
//    val numCores = 8
//    val mode = "local[" + numCores + "]"
//    val trainingInputPath = "input/Ads/trainingData"
//    val testingInputPath = "input/Ads/testingData"
//    val trainingOutputPath = "input/Ads/train"
//    val testingOutputPath = "input/Ads/test"
//    val freqLogPath = "output/AdsStats_Freq"
//    val histLogPath = "output/AdsStats_Hist"
    
	System.setProperty("spark.default.parallelism", numCores.toString)
    System.setProperty("spark.serializer", "spark.KryoSerializer")
    System.setProperty("spark.kryo.registrator", "util.MyRegistrator")
    System.setProperty("spark.storage.blockManagerSlaveTimeoutMs", "600000")
    val storageLevel = storage.StorageLevel.MEMORY_AND_DISK_SER
    
    val environment : Map[String, String] = null
    val sc = new SparkContext(mode, JOB_NAME, System.getenv("SPARK_HOME"), JARS, 
        environment)
    
    val bwFreq = new BufferedWriter(new FileWriter(new File(freqPath)))
    val bwHist = new BufferedWriter(new FileWriter(new File(histPath)))
    
    val rawTrainingData = sc.textFile(trainingInputPath, numCores).persist(storageLevel)
    val rawTestingData = sc.textFile(testingInputPath, numCores)
    val stats = Preprocess.wordCountAds(rawTrainingData).persist(storageLevel)
    
    val freq = stats.map(pair => pair._2).collect
    scala.util.Sorting.quickSort(freq)
    var n = 0 
    while (n < freq.length) {
//      println("freq: " + freq(n))
      bwFreq.write(freq(n) + "\n")
      n += 1
    }
    bwFreq.close()
    val hist = stats.map(pair => (pair._2, 1)).reduceByKey(_+_).collect
    
    def quickSort(pairs : Array[(Int, Int)]) : Array[(Int, Int)] = {
      if (pairs.length <= 1) pairs
      else {
        val pivot = pairs(pairs.length/2)
        Array.concat(quickSort(pairs.filter(pair => pair._2 < pivot._2)), 
            pairs.filter(pair => pair._2 == pivot._2), 
            quickSort(pairs.filter(pair => pair._2 > pivot._2)))
      }
    }
    
    val sortedHist = quickSort(hist)
    n = 0 
    while (n < sortedHist.length) {
      bwHist.write(sortedHist(n)._1 + ":" + sortedHist(n)._2 + "\n")
      n += 1
    }
    bwHist.close()
    
    val featureSet = stats.map(pair => pair._1).collect
    val P = featureSet.size + 1
    val featureMap = sc.broadcast(featureSet.zipWithIndex.toMap)
    
    def makeString(pair : Pair[Boolean, SparseVector]) : String = {
      val response = pair._1
      val features = pair._2.getIndices
      val sb = new StringBuilder
      if (response) sb += '1'
      else sb += '0'
      sb += ' '
      features.foreach(id => sb ++= id.toString+" ")
      sb.toString
    }
    
    rawTrainingData.map(line => {
      val record = Preprocess.parseAdsLine(line, featureMap.value)
      if (record!=null) makeString(record)
    }).saveAsTextFile(trainingOutputPath)
    
    rawTestingData.map(line => {
      val record = Preprocess.parseAdsLine(line, featureMap.value)
      if (record!=null) makeString(record)
    }).saveAsTextFile(testingOutputPath)
    println("P original: " + featureSet.length)
  }
}
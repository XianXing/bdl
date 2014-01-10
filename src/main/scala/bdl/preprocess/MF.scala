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

//preprocess for the matrix factorization problems
object MF {

  def parseLine(line: String, sep: String) : Record = {
    val tokens = line.split(sep)
    assert(tokens.length == 3, "unexpected record: " + line)
    new Record(tokens(0).toInt, tokens(1).toInt, tokens(2).toFloat)
  }
  
  def parseLine(line: String, sep: String, mean: Float, scale: Float) : Record = {
    val tokens = line.split(sep)
    assert(tokens.length == 3, "unexpected record: " + line)
    new Record(tokens(0).toInt, tokens(1).toInt, (tokens(2).toFloat-mean)/scale)
  }
  
  def parseLine(line: String) : Array[Record] = {
    var list = Nil
    var pair = line.split('\t')
    val rowInx = pair(0).toInt
    val ratings = pair(1).split(' ')
    for (record <- ratings) 
      yield {
        pair = record.split(":")
        new Record(rowInx, pair(0).toInt, pair(1).toFloat)
      }
  }
  
  def parseLine(line: String, mean: Float, scale: Float) : Array[Record] = {
    var list = Nil
    var pair = line.split('\t')
    val rowInx = pair(0).toInt
    val ratings = pair(1).split(' ')
    for (record <- ratings) 
      yield {
        pair = record.split(":")
        new Record(rowInx, pair(0).toInt, (pair(1).toFloat-mean)/scale)
      }
  }
  
  def getPartitionID(record: Record, numRowBlocks: Int, numColBlocks: Int) 
    : Int = (record.rowIdx%numRowBlocks)*numColBlocks + record.colIdx%numColBlocks
    
  def getPartitionMap(numRows: Int, numRowBlocks: Int, bw: BufferedWriter) 
    : Array[Byte] = {
    val partitionMap = Array.ofDim[Byte](numRows)
    val random = new Random
    val stats = Array.ofDim[Int](numRowBlocks)
    var r = 0
    while (r < numRows) {
      val id = (random.nextInt(numRowBlocks)).toByte
      stats(id) += 1
      partitionMap(r) = id
      r += 1
    }
    var i = 0; while (i<stats.length) { 
      println("block " + i + " has " + stats(i) + " elements")
      bw.write("block " + i + " has " + stats(i) + " elements\n")
      i += 1
    }
    partitionMap
  }
  
  def getPartitionedData(records: rdd.RDD[Record], numRowBlocks: Int, 
      numColBlocks: Int, partitioner: Partitioner)
    : rdd.RDD[(Int, SparseMatrix)] = {
    records.map{ record => (
        (record.rowIdx%numRowBlocks)*numColBlocks + record.colIdx%numColBlocks, record)
    }.groupByKey(partitioner).mapValues(seq => SparseMatrix(seq.toArray))
  }
  
  def getPartitionedData(records: rdd.RDD[Record], numColBlocks: Int,
      rowPartitionMap: broadcast.Broadcast[Array[Byte]], 
      colPartitionMap: broadcast.Broadcast[Array[Byte]],
      partitioner: Partitioner) 
   : rdd.RDD[(Int, SparseMatrix)] = {
    records.mapPartitions(_.map(record => {
      val rowPID = rowPartitionMap.value(record.rowIdx)
      val colPID = colPartitionMap.value(record.colIdx)
      (rowPID*numColBlocks + colPID, record)
    })).groupByKey(partitioner).mapValues(seq => SparseMatrix(seq.toArray))
  }
  
  def preprocessNetflix(
      inputDir: String, probePath: String, trainPath: String, testPath: String)= {
    
    def readProbe(probePath: String) = {
      val dict = new HashMap[String, HashSet[String]]
      var set: HashSet[String] = null
      for (line <- Source.fromFile(probePath).getLines) {
        if (line.contains(':')) {
          set = new HashSet[String]
          dict.put(line.substring(0, line.length-1), set)
        }
        else
          set.add(line)
      }
      dict
    }
    
    val dict = readProbe(probePath)
    val trainBW = new BufferedWriter(new FileWriter(new File(trainPath)))
    val testBW = new BufferedWriter(new FileWriter(new File(testPath)))
    val sbTrain: StringBuilder = new StringBuilder()
    val sbTest: StringBuilder = new StringBuilder()
    var numtrain = 0; var numtest = 0
    for (file <- new java.io.File(inputDir).listFiles) {
      val lines = Source.fromFile(file).getLines.toList
      val rowIdx = lines(0).substring(0, lines(0).length-1)
      val set = dict.getOrElse(rowIdx, null)
      var firstTrain = true; var firstTest = true
      for (line <- lines.tail) {
        val tokens = line.split(",")
        assert(tokens.length == 3, "unexpected line: " + line)
        if (set!=null && set.contains(tokens(0))) {
          numtest += 1
          if (firstTest) {
            sbTest.append(lines(0).substring(0, lines(0).length-1))
            sbTest.append('\t')
            firstTest = false
          }
          sbTest.append(tokens(0)+':'+tokens(1)+' ')
        }
        else {
          numtrain += 1
          if (firstTrain) {
            sbTrain.append(lines(0).substring(0, lines(0).length-1))
            sbTrain.append('\t')
            firstTrain = false
          }
          sbTrain.append(tokens(0)+':'+tokens(1)+' ')
        }
      }
      if (sbTrain.length > 0) { trainBW.write(sbTrain.toString + '\n'); sbTrain.clear }
      if (sbTest.length > 0) { testBW.write(sbTest.toString + '\n'); sbTest.clear }
    }
    println("num of train: " + numtrain + "\nnum of test: " + numtest)
    trainBW.close()
    testBW.close()
  }
  
  def generateDict(inputPath: String, thre: Int) 
    : scala.collection.Map[String, Int]= {
    val dict = new HashMap[String, Int]
    for (line <- Source.fromFile(inputPath).getLines) {
      val tokens = line.split(' ')
      for (token <- tokens) dict.put(token, dict.getOrElse(token, 0)+1)
    }
    dict.filter(pair => pair._2>thre).keySet.zipWithIndex.toMap
  }
  
  def preprocessNYTimes(
      inputPath: String, trainPath: String, testPath: String, ratio: Double) = {
    
    val doc = new HashMap[Int, Int]
    val trainBW = new BufferedWriter(new FileWriter(new File(trainPath)))
    val testBW = new BufferedWriter(new FileWriter(new File(testPath)))
    val sbTrain: StringBuilder = new StringBuilder()
    val sbTest: StringBuilder = new StringBuilder()
    val numberFormat = new DecimalFormat("0.00");
    val dict = generateDict(inputPath, 5)
    val numWords = dict.size
    println("dict size: " + numWords)
    var numDocs = 0
    var numTrain = 0L
    var numTest = 0
    for (line <- Source.fromFile(inputPath).getLines) {
      val tokens = line.split(' ')
      for (token <- tokens) {
        dict.get(token) match {
          case Some(id) => doc.put(id, doc.getOrElse(id, 0)+1)
          case None => None
        }
      }
      doc.map(pair => {
        if (Random.nextDouble < ratio) 
          sbTrain.append(pair._1 + ":" + numberFormat.format(math.log(pair._2+1))+" ") 
        else 
          sbTest.append(pair._1 + ":" + numberFormat.format(math.log(pair._2+1))+" ")
      })
      doc.clear
      if (sbTrain.length > 0) {
        trainBW.write(numDocs + "\t" + sbTrain.toString)
        trainBW.newLine()
        numTrain += sbTrain.length
        sbTrain.clear
      }
      if (sbTest.length > 0) {
        testBW.write(numDocs + "\t" + sbTest.toString)
        testBW.newLine()
        numTest += sbTest.length
        sbTest.clear
      }
      numDocs += 1
    }
    println("num of docs: " + numDocs + "\nnum of words: " + numWords)
    println("num of train: " + numTrain + "\nnum of test: " + numTest)
    trainBW.close()
    testBW.close()
  }
  
  def preprocessWiki(
      inputPath: String, trainDir: String, testPath: String, ratio: Double) {
    
    val sbTrain: StringBuilder = new StringBuilder()
    val sbTest: StringBuilder = new StringBuilder()
    var nnz = 0
    var fileCount = 0 
    var trainBW = new BufferedWriter(new FileWriter(
        new File(trainDir + "/" + fileCount)))
    val testBW = new BufferedWriter(new FileWriter(new File(testPath)))
    val random = new Random
    for (line <- Source.fromFile(inputPath).getLines) {
      if (nnz%5138830 == 0)  {
        trainBW.close()
        fileCount += 1
        trainBW = new BufferedWriter(new FileWriter(
            new File(trainDir + "/" + fileCount)))
      }
      if (nnz > 1) {  
        //skip the first two lines
        if (random.nextFloat < ratio) {
          trainBW.write(line)
          trainBW.newLine()
        }
        else {
          testBW.write(line)
          testBW.newLine()
        }
      }
      nnz += 1
    }
    trainBW.close
    testBW.close
  }
  
  def preprocessMovieLens(sc: SparkContext, inputDir: String, outputDir: String) = {
    
    val STORAGE_LEVEL = storage.StorageLevel.MEMORY_AND_DISK_SER
    val data = sc.textFile(inputDir).map(line => {
      val tokens = line.split("::")
      assert(tokens.length == 4)
      val rowIdx = tokens(0).toInt
      val colIdx = tokens(1).toInt
      val value = tokens(2).toFloat
      new Record(rowIdx, colIdx, value)
    }).persist(STORAGE_LEVEL)
    val (numRows, numCols, nnz, sum) = data.map(record => 
      (record.rowIdx, record.colIdx, 1, record.value)
    ).reduce((tuple1, tuple2) => (math.max(tuple1._1, tuple2._1), 
        math.max(tuple1._2, tuple2._2), tuple1._3+tuple2._3, tuple1._4+tuple2._4))
    val mean = sum/nnz
    println("numRows: "+(numRows+1) + "\tnumCols: " + (numCols+1) + "\tmean: " + mean)
//    data.map(record => (NullWritable.get, 
//        new Record(record.rowIdx, record.colIdx, record.value-mean)))
    data.map(record => (NullWritable.get, record)).saveAsSequenceFile(outputDir)
  }
  
  def main(args: Array[String]) {
    
    val master = "local[2]"
    val jar = Seq("sparkproject.jar")
    val jobName = "preprocess_TF"
//    val inputDir = "../datasets/MovieLens/ml-1m/ra.train"
    val inputDir = "../datasets/MovieLens/ml-1m/ra.test"
    val outputDir = "input/ml-1m/mf_test"
    val sc = new SparkContext(master, jobName, System.getenv("SPARK_HOME"), jar)
    preprocessMovieLens(sc, inputDir, outputDir)
    
//    val inputDir = "input/EachMovie-GL"
//    val trainDir = "output/train"
//    val testDir = "output/test"
//    val ratio = 0.98
//    val sep = " "
//    val (master, jar, inputDir, trainDir, testDir, ratio)
//      = (args(0), Seq(args(1)), args(2), args(3), args(4), args(5).toDouble)
//    val sep = if (args.length == 7) args(6) else "\t"
//    
//    val STORAGE_LEVEL = storage.StorageLevel.MEMORY_AND_DISK_SER
//    val sc = new SparkContext(master, "prep wiki", System.getenv("SPARK_HOME"), jar)
//    val all = sc.textFile(inputDir).map(line => {
//      val tokens = line.trim.split(sep)
//      assert(tokens.length == 3)
//      (Random.nextDouble < ratio, 
//      (tokens(0).toInt, tokens(1).toInt, tokens(2).toFloat))
//    }).persist(STORAGE_LEVEL)
//    val (sum, count) = all.map(pair => (pair._2._3.toDouble, 1L))
//      .reduce((p1, p2) => (p1._1+p2._1, p1._2+p2._2))
//    println("mean is: " + sum/count)
//    all.filter(_._1).map(_._2).saveAsObjectFile(trainDir)
//    println("preprocessed training data stored at " + trainDir)
//    all.filter(!_._1).map(_._2).saveAsObjectFile(testDir)
//    println("preprocessed testing data stored at " + testDir)
  }
}
package preprocess

import scala.collection.mutable.{HashMap, HashSet, ArrayBuffer}
import scala.io._
import scala.util.Random
import scala.collection.JavaConversions._

import java.io._
import java.text.DecimalFormat
import java.util.TreeMap

import org.apache.spark.{storage, Partitioner, SparkContext, rdd, broadcast}
import org.apache.spark.SparkContext._
import org.apache.spark.SparkContext

import org.apache.hadoop.io.NullWritable

import utilities._

//preprocessing for the Tensor factorization problems
object TF {

  def parseLine(line: String, sep: String) : Triplet = {
    val tokens = line.split(sep)
    assert(tokens.length == 4, "unexpected record: " + line)
    new Triplet(tokens(0).toInt, tokens(1).toInt, tokens(2).toInt, tokens(3).toFloat)
  }
  
  def parseLine(line: String, sep: String, mean: Float, scale: Float) : Triplet = {
    val tokens = line.split(sep)
    assert(tokens.length == 4, "unexpected record: " + line)
    new Triplet(tokens(0).toInt, tokens(1).toInt, tokens(2).toInt, 
        (tokens(3).toFloat-mean)/scale)
  }
  
  def parseLine(line: String) : Array[Triplet] = {
    var list = Nil
    var pair = line.split('\t')
    val rowInx = pair(0).toInt
    val ratings = pair(1).split(' ')
    for (record <- ratings) 
      yield {
        pair = record.split(":")
        new Triplet(rowInx, pair(0).toInt, pair(1).toInt, pair(2).toFloat)
      }
  }
  
  def parseLine(line: String, mean: Float, scale: Float) : Array[Triplet] = {
    var list = Nil
    var pair = line.split('\t')
    val rowInx = pair(0).toInt
    val ratings = pair(1).split(' ')
    for (record <- ratings) 
      yield {
        pair = record.split(":")
        new Triplet(rowInx, pair(0).toInt, pair(1).toInt, (pair(2).toFloat-mean)/scale)
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
  
  def getPartitionedData(records: rdd.RDD[Triplet], numBlocks2: Int, numBlocks3: Int,
      map1: broadcast.Broadcast[Array[Byte]], 
      map2: broadcast.Broadcast[Array[Byte]],
      map3: broadcast.Broadcast[Array[Byte]],
      partitioner: Partitioner,
      multicore: Boolean = false) 
   : rdd.RDD[(Int, SparseCube)] = {
    records.map(triplet => {
      val bid1 = map1.value(triplet._1)
      val bid2 = map2.value(triplet._2)
      val bid3 = map3.value(triplet._3)
      (bid1*numBlocks2*numBlocks3 + bid2*numBlocks3 + bid3, triplet)
    }).groupByKey(partitioner).mapValues(seq => SparseCube(seq.toArray, multicore))
  }
  
  def preprocessMovieLens(inputDir: String, outputDir: String, interval: Int) = {
    
    def analyzeTimeStamp(inputDir: String, interval: Int) : Int = {
      val timeStampArray = new TreeMap[Int, Int]
      for (line <- Source.fromFile(inputDir).getLines.toList) {
        val tokens = line.split("::")
        assert(tokens.length == 4)
        val time = tokens(3).toInt/interval
        val count = timeStampArray.getOrElse(time, 0)
        timeStampArray.put(time, count+1)
      }
      println("number months: " + timeStampArray.size)
      timeStampArray.map(_.toString).foreach(println)
      timeStampArray.firstKey
    }
    
    val begin = analyzeTimeStamp(inputDir, interval)
    val bw = new BufferedWriter(new FileWriter(new File(outputDir)))
    for (line <- Source.fromFile(inputDir).getLines.toList) {
      val tokens = line.split("::")
      assert(tokens.length == 4)
      val dim1 = tokens(0)
      val dim2 = tokens(1)
      val value = tokens(2)
      val dim3 = tokens(3)
      val time = dim3.toInt/interval-begin
      bw.write(dim1 + ' ' + dim2 + ' ' + time + ' ' + value + '\n')
    }
    bw.close
  }
  
  def analyzeMovieLensTimeStamps(sc: SparkContext, inputDir: String, interval: Int) 
    : Int = {
      val timeStampArray = sc.textFile(inputDir).map(line => {
        val tokens = line.split("::")
        assert(tokens.length == 4)
        (tokens(3).toInt/interval, 1)
      }).reduceByKey(_+_).sortByKey(true).collect
      println("number months: " + timeStampArray.size)
      timeStampArray.map(_.toString).foreach(println)
      timeStampArray(0)._1
    }
  
  def preprocessMovieLens(
      sc: SparkContext, inputDir: String, outputDir: String, interval: Int) = {
    
    val STORAGE_LEVEL = storage.StorageLevel.MEMORY_AND_DISK_SER
    val begin = analyzeMovieLensTimeStamps(sc, inputDir, interval)
    println("begin: " + begin)
    val data = sc.textFile(inputDir).map(line => {
      val tokens = line.split("::")
      assert(tokens.length == 4)
      val dim1 = tokens(0).toInt
      val dim2 = tokens(1).toInt
      val value = tokens(2).toFloat
      val dim3 = tokens(3).toInt/interval-begin
      new Triplet(dim1, dim2, dim3, value)
    }).persist(STORAGE_LEVEL)
    val (size1, size2, size3, nnz, sum) = data.map(triplet => 
      (triplet._1, triplet._2, triplet._3, 1, triplet.value)
    ).reduce((tuple1, tuple2) => (math.max(tuple1._1, tuple2._1), 
        math.max(tuple1._2, tuple2._2), math.max(tuple1._3, tuple2._3), 
        tuple1._4+tuple2._4, tuple1._5+tuple2._5))
    val mean = sum/nnz
    println("d1: " + (size1+1) + "\td2: " + (size2+1) +
      "\td3: " + (size3+1) + "\tmean: " + mean)
    data.map(triplet => (NullWritable.get, triplet)).saveAsSequenceFile(outputDir)
//    data.map(triplet => (NullWritable.get, 
//      new Triplet(triplet._1, triplet._2, triplet._3, triplet.value-mean)))
//    .saveAsSequenceFile(outputDir)
  }
  
  def getTimeNetflix(string: String, interval: Int) = {
    val ymd = string.split('-')
    ((ymd(0).toInt-1999)*365 + ymd(1).toInt*30 + ymd(2).toInt)/interval
  }
  
  def analyzeNetflixTimeStamps(inputDir: String, interval:Int, sliceSize: Int) 
    : HashMap[Int, Int] = {
      val histogram = new TreeMap[Short, Int]
      val map = new HashMap[Int, Int]
      var numSamples = 0
      val files = new java.io.File(inputDir).listFiles
      for (file <- files) {
        val lines = Source.fromFile(file).getLines
        lines.next
        for (line <- lines) {
          val tokens = line.split(',')
          assert(tokens.length == 3, "unexpected line: " + line)
          val time = getTimeNetflix(tokens(2), interval).toShort
          val count = histogram.getOrElse(time, 0)
          histogram.put(time, count+1)
          numSamples += 1
        }
      }
      var acc = 0
      var frameID = 0
      histogram.foreach{case (time, count) =>
        acc += count
        if (sliceSize - acc < sliceSize/5) {
          frameID += 1
          acc = 0
        }
        map.put(time, frameID)
        println(time + ", count: " + count + " ID: " + frameID)
      }
      println("num of time slices: " + map.size)
      map
    }
  
  def preprocessNetflix(
      inputDir: String, probeFilePath: String, interval: Int, sliceSize: Int,
      trainPath: String, testPath: String)= {
    
    //read the probe dataset, which is a subset of the training dataset. so we need to
    //filter out all ratings appeared in the probe dataset from the training dataset
    def readProbe(probeFilePath: String) : HashMap[String, HashSet[String]] = {
      val dict = new HashMap[String, HashSet[String]]
      var set: HashSet[String] = null
      for (line <- Source.fromFile(probeFilePath).getLines) {
        if (line.contains(':')) {
          set = new HashSet[String]
          dict.put(line.substring(0, line.length-1), set)
        }
        else
          set.add(line)
      }
      dict
    }
    
    val dict = readProbe(probeFilePath)
    val map = analyzeNetflixTimeStamps(inputDir, interval, sliceSize)
    val trainBW = new BufferedWriter(new FileWriter(new File(trainPath+map.size)))
    val testBW = new BufferedWriter(new FileWriter(new File(testPath+map.size)))
    val sbTrain: StringBuilder = new StringBuilder()
    val sbTest: StringBuilder = new StringBuilder()
    var numtrain = 0; var numtest = 0
    val files = new java.io.File(inputDir).listFiles
    for (file <- files) {
      val lines = Source.fromFile(file).getLines
      val first = lines.next
      val rowIdx = first.substring(0, first.length-1)
      val set = dict.getOrElse(rowIdx, null)
      var firstTrain = true; var firstTest = true
      for (line <- lines) {
        val tokens = line.split(",")
        assert(tokens.length == 3, "unexpected line: " + line)
        val time = map.getOrElse(getTimeNetflix(tokens(2), interval), 0)
        if (set!=null && set.contains(tokens(0))) {
          numtest += 1
          if (firstTest) {
            sbTest.append(rowIdx)
            sbTest.append('\t')
            firstTest = false
          }
          sbTest.append(tokens(0)+':'+ time +':'+tokens(1)+' ')
        }
        else {
          numtrain += 1
          if (firstTrain) {
            sbTrain.append(rowIdx)
            sbTrain.append('\t')
            firstTrain = false
          }
          sbTrain.append(tokens(0)+':'+time +':'+tokens(1)+' ')
        }
      }
      if (sbTrain.length > 0) { trainBW.write(sbTrain.toString + '\n'); sbTrain.clear }
      if (sbTest.length > 0) { testBW.write(sbTest.toString + '\n'); sbTest.clear }
    }
    println("num of train: " + numtrain + "\nnum of test: " + numtest)
    trainBW.close()
    testBW.close()
  }
  
  def getTimeYahooMusic(string: String, interval: Int = 1) 
    = (string.toInt/interval).toShort
  
  def analyzeYahooMusicTimeStamps(inputPath: String, interval: Int, sliceSize: Int) = {
    val timeStampArray = new TreeMap[Short, Int]
    val map = new HashMap[Int, Int]
    var totalCount = 0
    for (line <- Source.fromFile(inputPath).getLines) {
      val tokens = line.split('\t')
      if (tokens.length==4) {
        val day = getTimeYahooMusic(tokens(2), interval)
        val count = timeStampArray.getOrElse(day, 0)
        timeStampArray.put(day, count+1)
        totalCount += 1
      }
    }
    var acc = 0
    var frameID = 0
    timeStampArray.foreach{case (day, count) =>
      acc += count
      if (sliceSize - acc < sliceSize/5) {
        frameID += 1
        acc = 0
      }
      map.put(day, frameID)
      println(day + ", count: " + count + " ID: " + frameID)
    }
    println("num of time slices: " + map.size)
    println("total number of observations: " + totalCount)
    map
  }
  
  def preprocessYahooMusic(inputPath: String, outputDir: String, splitThres: Int,
      interval: Int, map: HashMap[Int, Int]) = {
    
    var numOfUsers = 0
    var numFiles = 0
    var numOfObs = 0
    var bw : BufferedWriter = null
    var userID = ""
    var ignore = false
    for (line <- Source.fromFile(inputPath).getLines) {
      if (line.contains('|')) {
        if (numOfUsers % splitThres == 0) {
          if (bw != null) bw.close
          bw = new BufferedWriter(new FileWriter(
              new File(outputDir+'/'+numFiles)))
          numFiles += 1
        }
        else bw.newLine
        val tokens = line.split('|')
        if (tokens(1).toInt < 10000) {
          userID = tokens(0)
          bw.write(userID+"\t")
          numOfUsers += 1
          ignore = false
        }
        else ignore = true
      }
      else if (!ignore){
        val tokens = line.split('\t')
        val time = map(getTimeYahooMusic(tokens(2), interval))
        bw.write(tokens(0)+':'+time+':'+tokens(1)+" ")
        numOfObs += 1
      }
    }
    bw.close
    println("total number of users: " + numOfUsers)
    println("total number of observations: " + numOfObs)
  }
  
  def main(args: Array[String]) {
    
//    val master = "local[2]"
//    val jar = Seq("sparkproject.jar")
//    val jobName = "preprocess_TF"
//    val inputDir = "../datasets/MovieLens/ml-1m/ra.train"
//    val outputDir = "input/ml-1m/tf_train"
////    val inputDir = "../datasets/MovieLens/ml-1m/ra.test"
////    val outputDir = "input/ml-1m/tf_test"    
//    val sc = new SparkContext(master, jobName, System.getenv("SPARK_HOME"), jar)
//    val interval = 3600*24*30
//    preprocessMovieLens(sc, inputDir, outputDir, interval)
    
//    val sliceSize = 5000
//    val interval = 1
//    val inputDir = "../datasets/Netflix/training_set"
//    val probeFilePath = "../datasets/Netflix/probe.txt"
//    val trainFilePath = "../datasets/Netflix/tf_train_ss_" + 
//      sliceSize + "_i_" + interval + "_ns_"
//    val testFilePath = "../datasets/Netflix/tf_test_ss_" + 
//      sliceSize + "_i_" + interval + "_ns_"
//    preprocessNetflix(inputDir, probeFilePath, interval, sliceSize, 
//        trainFilePath, testFilePath)
    
    val interval = 15
    val sliceSize = 10000
    val splitThre = 50000
    val trainInputPath = "../datasets/YahooMusic/Webscope_C15/" +
    		"ydata-ymusic-kddcup-2011-track1/trainIdx1.txt"
    val validateInputPath = "../datasets/YahooMusic/Webscope_C15/" +
          "ydata-ymusic-kddcup-2011-track1/validationIdx1.txt"
    val map =  analyzeYahooMusicTimeStamps(trainInputPath, interval, sliceSize)
    
    val trainOutputDir = "../datasets/YahooMusic/tf_train_ss_" + 
      sliceSize + "_i_" + interval + "_ns_" + map.size
    val validateOutputDir = "../datasets/YahooMusic/tf_validate_ss_" + 
      sliceSize + "_i_" + interval + "_ns_" + map.size
    
    val trainDir = new File(trainOutputDir)
    if (trainDir.exists || trainDir.mkdir) {
      preprocessYahooMusic(trainInputPath, trainOutputDir, splitThre, interval, map)
    }
    val validateDir = new File(validateOutputDir)
    if (validateDir.exists || validateDir.mkdir) {
      preprocessYahooMusic(validateInputPath, validateOutputDir, splitThre, interval, 
          map)
    }
  }
}
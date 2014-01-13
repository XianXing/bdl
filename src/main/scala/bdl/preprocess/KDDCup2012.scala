package preprocess

import scala.collection.mutable.{HashMap, ArrayBuilder}

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, HashPartitioner, storage}
import org.apache.spark.SparkContext._
import org.apache.spark.serializer.KryoRegistrator


object KDDCup2012 {
  
  val dataDir = "/home/xz60/data/KDDCup2012/"
  val trainFile = dataDir + "training.txt"
  val despTokenFile = dataDir + "descriptionid_tokensid.txt"
  val keywordTokenFile = dataDir + "purchasedkeywordid_tokensid.txt"
  val queryTokenFile = dataDir + "queryid_tokensid.txt" 
  val titleTokenFile = dataDir + "titleid_tokensid.txt"
  val userProfileFile = dataDir + "userid_profile.txt"
  
  val testFeatureFile = dataDir + "test_features"
  val testLabelFile = dataDir + "test_labels"
  
  val outputDir = "output/KDDCup2012/"
  val tmpDir = "/tmp/spark"
  val numCores = 16
  val numReducers = 16*numCores
  val mode = "local[16]"
  val jars = Seq("examples/target/scala-2.9.3/" +
  		"spark-examples-assembly-0.8.1-incubating.jar")
    
  def main(args : Array[String]) {
    
    System.setProperty("spark.local.dir", tmpDir)
    System.setProperty("spark.default.parallelism", numReducers.toString)
    System.setProperty("spark.storage.memoryFraction", "0.3")
    System.setProperty("spark.akka.frameSize", "1110") //for large .collect() objects
//    System.setProperty("spark.serializer", 
//        "org.apache.spark.serializer.KryoSerializer")
//    System.setProperty("spark.kryo.registrator", "utilities.Registrator")
//    System.setProperty("spark.kryoserializer.buffer.mb", "16")
//    System.setProperty("spark.kryo.referenceTracking", "false")
    
    val jobName = "Preprocess_KDDCup2012"
    val logPath = outputDir + jobName + ".txt"
    
    val storageLevel = storage.StorageLevel.MEMORY_AND_DISK_SER
    val sc = new SparkContext(mode, jobName, System.getenv("SPARK_HOME"), jars)
      
//    Read in the descriptionid_tokensid.txt file
    //num unique desriptions = 3171830, max desriptionID = 3171829
    //if despTokenSize = 20000, each token appears at least 81 times 
    val numDesp = 3171829+1
    val despTokenSize = 20000
    val despToken = new Array[Array[Int]](numDesp)
    val tokensMap = new HashMap[Int, Int]
    val despTokenRDD = sc.textFile(despTokenFile).map(line => {
      val array = line.split("\t")
      assert(array.length == 2)
      (array(0).toInt, array(1).split("\\|").map(_.toInt))
    }).persist(storageLevel)
    despTokenRDD.flatMap(_._2).map((_, 1)).reduceByKey(_+_)
      .map(pair => (pair._2, pair._1)).sortByKey(false).map(_._2)
      .take(despTokenSize).zipWithIndex.foreach(pair => tokensMap(pair._1) = pair._2)
    despTokenRDD.collect.par.foreach(pair => {
      despToken(pair._1) = pair._2.map(tokensMap.getOrElse(_, -1)).filter(_>=0)
    })
    despTokenRDD.unpersist(false)
    
//    Read in the purchasedkeywordid_tokensid.txt file
    //num unique keywords = 1249785, max keywordID = 1249784
    //if keywordTokenSize = 20000, each token appears at least 11 times
    val numKeywords = 1249784 + 1
    val keywordTokenSize = 20000
    val keywordToken = new Array[Array[Int]](numKeywords)
    tokensMap.clear
    val keywordTokenRDD = sc.textFile(keywordTokenFile).map(line => {
      val array = line.split("\t")
      assert(array.length == 2)
      (array(0).toInt, array(1).split("\\|").map(_.toInt))
    }).persist(storageLevel)
    keywordTokenRDD.flatMap(_._2).map((_, 1)).reduceByKey(_+_)
      .map(pair => (pair._2, pair._1)).sortByKey(false).map(_._2)
      .take(keywordTokenSize).zipWithIndex.foreach(pair => tokensMap(pair._1) = pair._2)
    keywordTokenRDD.collect.par.foreach(pair => {
      keywordToken(pair._1) = pair._2.map(tokensMap.getOrElse(_, -1)).filter(_>=0)
    })
    keywordTokenRDD.unpersist(false)
    
//    Read in the queryid_tokensid.txt file
    //num unique queries: 26243606, max queryID = 26243605
    //if queryTokenSize = 20000, each token appears at least 407 times
    val numQueries = 26243605 + 1
    val queryTokenSize = 20000
    val queryToken = new Array[Array[Int]](numQueries)
    tokensMap.clear
    val queryTokenRDD = sc.textFile(queryTokenFile).map(line => {
      val array = line.split("\t")
      assert(array.length == 2)
      (array(0).toInt, array(1).split("\\|").map(_.toInt))
    }).persist(storageLevel)
    queryTokenRDD.flatMap(_._2).map((_, 1)).reduceByKey(_+_)
      .map(pair => (pair._2, pair._1)).sortByKey(false).map(_._2)
      .take(queryTokenSize).zipWithIndex.foreach(pair => tokensMap(pair._1) = pair._2)
    queryTokenRDD.collect.par.foreach(pair => {
      queryToken(pair._1) = pair._2.map(tokensMap.getOrElse(_, -1)).filter(_>=0)
    })
    queryTokenRDD.unpersist(false)
    
//    Read in the titleid_tokensid.txt file
    //num of unique titiles = 4051441, max titleID = 4051440 (246.6 MB)
    //if titleTokenSize = 20000, each token appears at least 49 times
    val numTitles = 4051440 + 1
    val titleTokenSize = 20000
    val titleToken = new Array[Array[Int]](numTitles)
    tokensMap.clear
    val titleTokenRDD = sc.textFile(titleTokenFile).map(line => {
      val array = line.split("\t")
      assert(array.length == 2)
      (array(0).toInt, array(1).split("\\|").map(_.toInt))
    }).persist(storageLevel)
    titleTokenRDD.flatMap(_._2).map((_, 1)).reduceByKey(_+_)
      .map(pair => (pair._2, pair._1)).sortByKey(false).map(_._2)
      .take(titleTokenSize).zipWithIndex.foreach(pair => tokensMap(pair._1) = pair._2)
    titleTokenRDD.collect.par.foreach(pair => {
      titleToken(pair._1) = pair._2.map(tokensMap.getOrElse(_, -1)).filter(_>=0)
    })
    titleTokenRDD.unpersist(false)
    
//    Read in the userid_profile.txt file
    //num of unique users = 23669283, max userID = 23907634
    val numUsers = 23907634 + 1
    val userProfile = new Array[Array[Int]](numUsers)
    sc.textFile(userProfileFile).map(line => {
      val array = line.split("\t")
      assert(array.length == 3)
      (array(0).toInt, Array(array(1).toInt, array(2).toInt))
    }).collect.par.foreach(pair => userProfile(pair._1) = pair._2)
    
 //    Read in the training.txt file
    val train = sc.textFile(trainFile).mapPartitions(_.map(_.split("\t")).map(arr =>
      Array(arr(0).toInt, arr(1).toInt, arr(3).toInt, arr(4).toInt, arr(5).toInt, 
          arr(6).toInt, arr(7).toInt, arr(8).toInt, arr(9).toInt, arr(10).toInt, 
          arr(11).toInt))
      ).persist(storageLevel)
    
//  after preprocessing, trainFile line format:    
//  0. Click, 1. Impression, 2. AdID, 3. AdvertiserID, 4. Depth, 5. Position, 
//  6. QueryID, 7. KeywordID, 8. TitleID, 9. DescriptionID, 10. UserID 
      
//    Extracting some features from the raw data:

//    Number of appearances of the same user
    val userFreq = new Array[Int](numUsers)
    train.map(tokens => (tokens(10), tokens(1))).reduceByKey(_+_)
      .collect.foreach(pair => userFreq(pair._1) = pair._2)
    
//    Number of occurrences of the same query
    val queryFreq = new Array[Int](numQueries)
    train.map(tokens => (tokens(6), tokens(1))).reduceByKey(_+_)
      .collect.par.foreach(pair => queryFreq(pair._1) = pair._2)
    
//    Number of occurrences of the same ad
    //num of unique Ads 641707, max AdsID 22238277
    val adFreqPair = train.map(tokens => (tokens(2), tokens(1)))
      .reduceByKey(_+_).collect
    val adIDMap = new HashMap[Int, Int]
    adFreqPair.map(_._1).zipWithIndex.foreach(pair => adIDMap(pair._1) = pair._2)
    val numAds = adIDMap.size
    val adFreq = new Array[Int](numAds)
    adFreqPair.foreach(pair => adFreq(adIDMap(pair._1)) = pair._2)
    
//    Average click-through-rate for query
    val queryCtr = new Array[Float](numQueries)
    train.map(tokens => (tokens(6), (tokens(0), tokens(1))))
      .reduceByKey((p1, p2) => (p1._1+p2._1, p1._2+p2._2))
      .mapValues{case(click, impression) => 1.0f*click/impression}
      .collect.par.foreach(pair => queryCtr(pair._1) = pair._2)
    
//    Average click-through-rate for user
    val userCtr = new Array[Float](numUsers)
    train.map(tokens => (tokens(10), (tokens(0), tokens(1))))
      .reduceByKey((p1, p2) => (p1._1+p2._1, p1._2+p2._2))
      .mapValues{case(click, impression) => 1.0f*click/impression}
      .collect.par.foreach(pair => userCtr(pair._1) = pair._2)
      
//    Average click-through-rate for advertiser
    //num of unique advertisers 14847, max advertiserID 39191
    val numAdvrs = 39191+1
    val advrCtr = new Array[Float](numAdvrs)
    train.map(tokens => (tokens(3), (tokens(0), tokens(1))))
      .reduceByKey((p1, p2) => (p1._1+p2._1, p1._2+p2._2))
      .mapValues{case(click, impression) => 1.0f*click/impression}
      .collect.foreach(pair => advrCtr(pair._1) = pair._2)
    
//    Average click-through-rate for keyword advertised
    val keywordCtr = new Array[Float](numKeywords)
    train.map(tokens => (tokens(7), (tokens(0), tokens(1))))
      .reduceByKey((p1, p2) => (p1._1+p2._1, p1._2+p2._2))
      .mapValues{case(click, impression) => 1.0f*click/impression}
      .collect.foreach(pair => keywordCtr(pair._1) = pair._2)
      
    val queryTokenBC = sc.broadcast(queryToken)
    val queryFreqBC = sc.broadcast(queryFreq)
    val queryFreqDim = 25
    val queryFreqBinSize = math.max(queryFreq.reduce(math.max(_,_))/queryFreqDim, 1)
    val queryCtrBC = sc.broadcast(queryCtr)
    val queryCtrBinDim = 150
    val queryCtrBinSize = queryCtr.reduce(math.max(_,_))/queryCtrBinDim
    val despTokenBC = sc.broadcast(despToken)
    val keywordTokenBC = sc.broadcast(keywordToken)
    val keywordCtrBC = sc.broadcast(keywordCtr)
    val keywordCtrBinDim = 150
    val keywordCtrBinSize = keywordCtr.reduce(math.max(_,_))/keywordCtrBinDim
    val titleTokenBC = sc.broadcast(titleToken)
    val userProfileBC = sc.broadcast(userProfile)
    val userFreqBC = sc.broadcast(userFreq)
    val userFreqDim = 25
    val userFreqBinSize = math.max(userFreq.reduce(math.max(_,_))/userFreqDim, 1)
    val userCtrBC = sc.broadcast(userCtr)
    val userCtrBinDim = 150
    val userCtrBinSize = userCtr.reduce(math.max(_,_))/userCtrBinDim
    val adFreqBC = sc.broadcast(adFreq)
    val adFreqDim = 25
    val adFreqBinSize = math.max(adFreq.reduce(math.max(_,_))/adFreqDim, 1)
    val adIDMapBC = sc.broadcast(adIDMap)
    val advrCtrBC = sc.broadcast(advrCtr)
    val advrCtrBinDim = 150
    val advrCtrBinSize = advrCtr.reduce(math.max(_,_))/advrCtrBinDim
//  Form features from raw data:
    train.flatMap(arr => {
      val queryToken = queryTokenBC.value
      val queryFreq = queryFreqBC.value
      val queryCtr = queryCtrBC.value
      val despToken = despTokenBC.value
      val keywordToken = keywordTokenBC.value
      val keywordCtr = keywordCtrBC.value
      val titleToken = titleTokenBC.value
      val userProfile = userProfileBC.value
      val userFreq = userFreqBC.value
      val userCtr = userCtrBC.value
      val adFreq = adFreqBC.value
      val adIDMap = adIDMapBC.value
      val advrCtr = advrCtrBC.value
      val feature = new ArrayBuilder.ofInt
      feature.sizeHint(30)
      //intercept
      feature += 0
      var offset = 1
      //Query, D=queryTokenSize
      feature ++= queryToken(arr(6)).map(_+offset)
      offset += queryTokenSize
      //Gender, D=3
      if (userProfile(arr(10)) != null) feature += userProfile(arr(10))(0) + offset
      offset += 3
      //Keyword, D=keywordTokenSize
      feature ++= keywordToken(arr(7)).map(_+offset)
      offset += keywordTokenSize
      //Title, D=titleTokenSize
      feature ++= titleToken(arr(8)).map(_+offset)
      offset += titleTokenSize
      //Description, D=despTokenSize
      feature ++= despToken(arr(9)).map(_+offset)
      offset += despTokenSize
      //Advertiser, D=39192
      feature += arr(3) + offset
      offset += 39192
      //AdID, D=641707
      feature += adIDMap(arr(2)) + offset
      offset += 641707
      //age, D=6
      if (userProfile(arr(10)) != null) feature += userProfile(arr(10))(1)-1 + offset
      offset += 6
      //UserFreq, D=userFreqDim
      feature += math.min(userFreq(arr(10))/userFreqBinSize, userFreqDim-1) + offset
      offset += userFreqDim
      //Position, D=3
      feature += math.min(arr(5)-1, 2) + offset
      offset += 3
      //Depth, D=3
      feature += math.min(arr(4)-1, 2) + offset
      offset += 3
      //QueryFreq, D=queryFreqDim
      feature += math.min(queryFreq(arr(6))/queryFreqBinSize, queryFreqDim-1) + offset
      offset += queryFreqDim
      //AdFreq, D=adFreqDim
      feature += math.min(adFreq(adIDMap(arr(2)))/adFreqBinSize, adFreqDim-1) + offset
      offset += adFreqDim
      //QueryLength, D=20
      feature += math.min(queryToken(arr(6)).length, 19) + offset
      offset += 20
      //TitleLength, D=30
      feature += math.min(titleToken(arr(8)).length, 29) + offset
      offset += 30
      //DespLength, D=50
      feature += math.min(despToken(arr(9)).length, 49) + offset
      offset += 50
      //QueryCtr, D=queryCtrBinDim
      feature += 
        math.min((queryCtr(arr(6))/queryCtrBinSize).toInt, queryCtrBinDim-1) + offset
      offset += queryCtrBinDim
      //UserCtr, D=userCtrBinDim
      feature += 
        math.min((userCtr(arr(10))/userCtrBinSize).toInt, userCtrBinDim-1) + offset
      offset += userCtrBinDim
      //AdvrCtr, D=advrCtrBinDim
      feature += 
        math.min((advrCtr(arr(3))/advrCtrBinSize).toInt, advrCtrBinDim-1) + offset
      offset += advrCtrBinDim
      //WordCtr, D=keywordCtrBinDim
      feature += math.min((keywordCtr(arr(7))/keywordCtrBinSize).toInt, 
        keywordCtrBinDim-1) + offset
      offset += keywordCtrBinDim
      
      var click = arr(0)
      var impression = arr(1)
      val records = new Array[Array[Int]](impression)
      if (click >= 1) feature += 1
      else feature += -1
      records(0) = feature.result
      val length = records(0).length
      impression -= 1
      click -= 1
      var count = 1
      while (impression > 0) {
        val record = new Array[Int](length)
        Array.copy(records(0), 0, record, 0, length)
        record(length-1) = if (click >= 1) 1 else -1
        impression -= 1
        click -= 1
        records(count) = record
        count += 1
      }
      records
    }).saveAsObjectFile(outputDir + "train_obj")
//    .map(arr => arr(arr.length-1) + "\t" + arr.take(arr.length-1).mkString(" "))
//    .saveAsTextFile(outputDir)
    
    val test_feature = sc.textFile(testFeatureFile).map(_.split("\t"))
      .map(arr => (arr(0).trim.toInt, arr.drop(2).map(_.toInt)))
    val test_label = sc.textFile(testLabelFile).map(_.split("\t"))
      .map(arr => (arr(0).trim.toInt, arr(1).split(",").take(2).map(_.toInt)))
    val test = test_label.join(test_feature).map{
      case(id, (arr1, arr2)) =>
        val builder = new ArrayBuilder.ofInt
        val length = arr1.length+arr2.length
        assert(length == 11)
        builder.sizeHint(length)
        builder ++= arr1
        builder ++= arr2
        builder.result
    }.flatMap( arr => {
      val queryToken = queryTokenBC.value
      val queryFreq = queryFreqBC.value
      val queryCtr = queryCtrBC.value
      val despToken = despTokenBC.value
      val keywordToken = keywordTokenBC.value
      val keywordCtr = keywordCtrBC.value
      val titleToken = titleTokenBC.value
      val userProfile = userProfileBC.value
      val userFreq = userFreqBC.value
      val userCtr = userCtrBC.value
      val adFreq = adFreqBC.value
      val adIDMap = adIDMapBC.value
      val advrCtr = advrCtrBC.value
      val feature = new ArrayBuilder.ofInt
      feature.sizeHint(30)
      //intercept
      feature += 0
      var offset = 1
      //Query's token, D=queryTokenSize
      feature ++= queryToken(arr(6)).map(_+offset)
      offset += queryTokenSize
      //User's gender, D=3
      if (userProfile(arr(10)) != null) feature += userProfile(arr(10))(0) + offset
      offset += 3
      //Keyword's token, D=keywordTokenSize
      feature ++= keywordToken(arr(7)).map(_+offset)
      offset += keywordTokenSize
      //Title's token, D=titleTokenSize
      feature ++= titleToken(arr(8)).map(_+offset)
      offset += titleTokenSize
      //Description's token, D=despTokenSize
      feature ++= despToken(arr(9)).map(_+offset)
      offset += despTokenSize
      //Advertiser's ID, D=39192
      feature += arr(3) + offset
      offset += 39192
      //Ads' ID, D=641707
      if (adIDMap.contains(arr(2))) feature += adIDMap(arr(2)) + offset
      offset += 641707
      //User's age, D=6
      if (userProfile(arr(10)) != null) feature += userProfile(arr(10))(1)-1 + offset
      offset += 6
      //UserFreq, D=userFreqDim
      feature += math.min(userFreq(arr(10))/userFreqBinSize, userFreqDim-1) + offset
      offset += userFreqDim
      //Ads' position, D=3
      feature += math.min(arr(5)-1, 2) + offset
      offset += 3
      //Ads' depth, D=3
      feature += math.min(arr(4)-1, 2) + offset
      offset += 3
      //QueryFreq, D=queryFreqDim
      feature += math.min(queryFreq(arr(6))/queryFreqBinSize, queryFreqDim-1) + offset
      offset += queryFreqDim
      //AdFreq, D=adFreqDim
      if (adIDMap.contains(arr(2))) {
        feature += 
          math.min(adFreq(adIDMap(arr(2)))/adFreqBinSize, adFreqDim-1) + offset
      }
      offset += adFreqDim
      //Query token's length, D=20
      feature += math.min(queryToken(arr(6)).length, 19) + offset
      offset += 20
      //Title token's length, D=30
      feature += math.min(titleToken(arr(8)).length, 29) + offset
      offset += 30
      //Description's length, D=50
      feature += math.min(despToken(arr(9)).length, 49) + offset
      offset += 50
      //QueryCtr, D=queryCtrBinDim
      feature += 
        math.min((queryCtr(arr(6))/queryCtrBinSize).toInt, queryCtrBinDim-1) + offset
      offset += queryCtrBinDim
      //UserCtr, D=userCtrBinDim
      feature += 
        math.min((userCtr(arr(10))/userCtrBinSize).toInt, userCtrBinDim-1) + offset
      offset += userCtrBinDim
      //AdvrCtr, D=advrCtrBinDim
      if (arr(3) < advrCtr.length) {
        feature += 
          math.min((advrCtr(arr(3))/advrCtrBinSize).toInt, advrCtrBinDim-1) + offset
      }
      offset += advrCtrBinDim
      //WordCtr, D=keywordCtrBinDim
      feature += math.min((keywordCtr(arr(7))/keywordCtrBinSize).toInt, 
        keywordCtrBinDim-1) + offset
      offset += keywordCtrBinDim
      
      var click = arr(0)
      var impression = arr(1)
      val records = new Array[Array[Int]](impression)
      if (click >= 1) feature += 1
      else feature += -1
      records(0) = feature.result
      val length = records(0).length
      impression -= 1
      click -= 1
      var count = 1
      while (impression > 0) {
        val record = new Array[Int](length)
        Array.copy(records(0), 0, record, 0, length)
        record(length-1) = if (click >= 1) 1 else -1
        impression -= 1
        click -= 1
        records(count) = record
        count += 1
      }
      records
    }).saveAsObjectFile(outputDir + "test_obj")
//    .map(arr => arr(arr.length-1) + "\t" + arr.take(arr.length-1).mkString(" "))
//    .saveAsTextFile(outputDir + "test")
    System.exit(0)
  }
}
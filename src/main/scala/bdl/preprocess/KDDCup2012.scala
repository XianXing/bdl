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
    val numDesps = 3171829+1
    val despTokenSize = 20000
    val despToken = new Array[Array[Int]](numDesps)
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
    val keywordTokenFreqTh = 20
    val keywordToken = new Array[Array[Int]](numKeywords)
    tokensMap.clear
    val keywordTokenRDD = sc.textFile(keywordTokenFile).map(line => {
      val array = line.split("\t")
      assert(array.length == 2)
      (array(0).toInt, array(1).split("\\|").map(_.toInt))
    }).persist(storageLevel)
    keywordTokenRDD.flatMap(_._2).map((_, 1)).reduceByKey(_+_)
      .filter(_._2 > keywordTokenFreqTh).map(_._1)
      .collect.zipWithIndex.foreach(pair => tokensMap(pair._1) = pair._2)
    val keywordTokenSize = tokensMap.size
    keywordTokenRDD.collect.par.foreach(pair => {
      keywordToken(pair._1) = pair._2.map(tokensMap.getOrElse(_, -1)).filter(_>=0)
    })
    keywordTokenRDD.unpersist(false)
    
//    Read in the queryid_tokensid.txt file
    //num unique queries: 26243606, max queryID = 26243605
    val numQueries = 26243605 + 1
    val queryTokenFreqTh = 20
    val queryToken = new Array[Array[Int]](numQueries)
    tokensMap.clear
    val queryTokenRDD = sc.textFile(queryTokenFile).map(line => {
      val array = line.split("\t")
      assert(array.length == 2)
      (array(0).toInt, array(1).split("\\|").map(_.toInt))
    }).persist(storageLevel)
    queryTokenRDD.flatMap(_._2).map((_, 1)).reduceByKey(_+_)
      .filter(_._2>queryTokenFreqTh).map(_._1)
      .collect.zipWithIndex.foreach(pair => tokensMap(pair._1) = pair._2)
    val queryTokenSize = tokensMap.size
    queryTokenRDD.collect.par.foreach(pair => {
      queryToken(pair._1) = pair._2.map(tokensMap.getOrElse(_, -1)).filter(_>=0)
    })
    queryTokenRDD.unpersist(false)
    
//    Read in the titleid_tokensid.txt file
    //num of unique titiles = 4051441, max titleID = 4051440 (246.6 MB)
    //if titleTokenSize = 20000, each token appears at least 49 times
    val numTitles = 4051440 + 1
    val titleTokenFreqTh = 50
    val titleToken = new Array[Array[Int]](numTitles)
    tokensMap.clear
    val titleTokenRDD = sc.textFile(titleTokenFile).map(line => {
      val array = line.split("\t")
      assert(array.length == 2)
      (array(0).toInt, array(1).split("\\|").map(_.toInt))
    }).persist(storageLevel)
    titleTokenRDD.flatMap(_._2).map((_, 1)).reduceByKey(_+_)
      .filter(_._2 > titleTokenFreqTh).map(_._1)
      .collect.zipWithIndex.foreach(pair => tokensMap(pair._1) = pair._2)
    val titleTokenSize = tokensMap.size
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
      
//    Number of appearances of the same user
//    Average click-through-rate for user
//    num of unique users = 23669283, max userID = 23907634
    val userStats = train.map(tokens => (tokens(10), (tokens(0), tokens(1))))
      .reduceByKey((p1, p2) => (p1._1+p2._1, p1._2+p2._2)).collect.par
    val userFreq = new Array[Int](numUsers)
    userStats.map(pair => (pair._1, pair._2._1))
      .foreach(pair => userFreq(pair._1) = pair._2)
    val userWithClicks = new Array[Boolean](numUsers)
    userStats.filter(_._2._1>0).map(_._1).foreach(i => userWithClicks(i) = true)
    val userCtr = new Array[Float](numUsers)
    userStats.map{
      case(id, (click, impression)) => (id, (click + 0.05f*75)/(impression+75))
    }.foreach(pair => userCtr(pair._1) = pair._2)
      
//    Number of occurrences of the same query
//    Average click-through-rate for query
     //num unique queries: 26243606, max queryID = 26243605
    val queryStats = train.map(tokens => (tokens(6), (tokens(0), tokens(1))))
      .reduceByKey((p1, p2) => (p1._1+p2._1, p1._2+p2._2)).collect.par
    val queryFreq = new Array[Int](numQueries)
    queryStats.map(pair => (pair._1, pair._2._1))
      .foreach(pair => queryFreq(pair._1) = pair._2)
    val queryWithClicks = new Array[Boolean](numQueries)
    queryStats.filter(_._2._1>0).map(_._1).foreach(i => queryWithClicks(i) = true)
    val queryCtr = new Array[Float](numQueries)
    queryStats.map{
      case(id, (click, impression)) => (id, (click + 0.05f*75)/(impression+75))
    }.foreach(pair => queryCtr(pair._1) = pair._2)
    
//    Number of occurrences of the same ad
//    Average click-through-rate for ads' id
    //num of unique Ads 641707, max AdsID 22238277
    val adsStats = train.map(tokens => (tokens(2), (tokens(0), tokens(1))))
      .reduceByKey((p1, p2) => (p1._1+p2._1, p1._2+p2._2)).collect
    val adIDMap = new HashMap[Int, Int]
    adsStats.map(_._1).zipWithIndex.foreach(pair => adIDMap(pair._1) = pair._2)
    val numAds = adIDMap.size
    val adFreq = new Array[Int](numAds)
    adsStats.map(pair => (pair._1, pair._2._1))
      .foreach(pair => adFreq(adIDMap(pair._1)) = pair._2)
    val adWithClicks = new Array[Boolean](numAds)
    adsStats.filter(_._2._1 > 0).map(_._1).foreach(i => adWithClicks(adIDMap(i)) = true)
    val adCtr = new Array[Float](numAds)
    adsStats.map{
      case(id, (click, impression)) => (id, (click + 0.05f*75)/(impression+75))
    }.foreach(pair => adCtr(adIDMap(pair._1)) = pair._2)
    
//    Average click-through-rate for advertiser
    //num of unique advertisers 14847, max advertiserID 39191
    val numAdvrs = 39191+1
    val advrCtr = new Array[Float](numAdvrs)
    train.map(tokens => (tokens(3), (tokens(0), tokens(1))))
      .reduceByKey((p1, p2) => (p1._1+p2._1, p1._2+p2._2))
      .mapValues{case(click, impression) => (click + 0.05f*75)/(impression+75)}
      .collect.foreach(pair => advrCtr(pair._1) = pair._2)
    
//    Average click-through-rate for keyword advertised
    val keywordStats = train.map(tokens => (tokens(7), (tokens(0), tokens(1))))
      .reduceByKey((p1, p2) => (p1._1+p2._1, p1._2+p2._2)).collect.par
    val keywordWithClicks = new Array[Boolean](numKeywords)
    keywordStats.filter(_._2._1>0).map(_._1).foreach(i => keywordWithClicks(i) = true)
    val keywordCtr = new Array[Float](numKeywords)
    keywordStats.map{
      case(id, (click, impression)) => (id, (click + 0.05f*75)/(impression+75))
    }.foreach(pair => keywordCtr(pair._1) = pair._2)
    
//    Average click-through-rate for titile id
    val titleCtr = new Array[Float](numTitles)
    train.map(tokens => (tokens(8), (tokens(0), tokens(1))))
      .reduceByKey((p1, p2) => (p1._1+p2._1, p1._2+p2._2))
      .mapValues{case(click, impression) => (click + 0.05f*75)/(impression+75)}
      .collect.foreach(pair => titleCtr(pair._1) = pair._2)
      
//    Average click-through-rate for description id
    val despCtr = new Array[Float](numDesps)
    train.map(tokens => (tokens(9), (tokens(0), tokens(1))))
      .reduceByKey((p1, p2) => (p1._1+p2._1, p1._2+p2._2))
      .mapValues{case(click, impression) => (click + 0.05f*75)/(impression+75)}
      .collect.foreach(pair => despCtr(pair._1) = pair._2)
      
    val queryTokenBC = sc.broadcast(queryToken)
    val queryFreqBC = sc.broadcast(queryFreq)
    val queryFreqDim = 25
    val queryFreqBinSize = math.max(queryFreq.reduce(math.max(_,_))/queryFreqDim, 1)
    val queryCtrBC = sc.broadcast(queryCtr)
    val queryCtrBinDim = 100
    val queryCtrBinSize = queryCtr.reduce(math.max(_,_))/queryCtrBinDim
    val queryIDBinDim = 10000
    val queryIDBinSize = numQueries/queryIDBinDim
    val queryWithClicksBC = sc.broadcast(queryWithClicks)
    val despTokenBC = sc.broadcast(despToken)
    val despCtrBC = sc.broadcast(despCtr)
    val despCtrBinDim = 100
    val despCtrBinSize = despCtr.reduce(math.max(_,_))/despCtrBinDim
    val keywordTokenBC = sc.broadcast(keywordToken)
    val keywordCtrBC = sc.broadcast(keywordCtr)
    val keywordCtrBinDim = 100
    val keywordCtrBinSize = keywordCtr.reduce(math.max(_,_))/keywordCtrBinDim
    val keywordWithClicksBC = sc.broadcast(keywordWithClicks)
    val titleTokenBC = sc.broadcast(titleToken)
    val titleCtrBC = sc.broadcast(titleCtr)
    val titleCtrBinDim = 100
    val titleCtrBinSize =  titleCtr.reduce(math.max(_,_))/titleCtrBinDim
    val userProfileBC = sc.broadcast(userProfile)
    val userFreqBC = sc.broadcast(userFreq)
    val userFreqDim = 25
    val userFreqBinSize = math.max(userFreq.reduce(math.max(_,_))/userFreqDim, 1)
    val userCtrBC = sc.broadcast(userCtr)
    val userCtrBinDim = 100
    val userCtrBinSize = userCtr.reduce(math.max(_,_))/userCtrBinDim
    val userIDBinDim = 10000
    val userIDBinSize = numUsers/userIDBinDim
    val userWithClicksBC = sc.broadcast(userWithClicks)
    val adFreqBC = sc.broadcast(adFreq)
    val adFreqDim = 25
    val adFreqBinSize = math.max(adFreq.reduce(math.max(_,_))/adFreqDim, 1)
    val adCtrBC = sc.broadcast(adCtr)
    val adCtrBinDim = 100
    val adCtrBinSize = math.max(adCtr.reduce(math.max(_,_))/adCtrBinDim, 1)
    val adIDMapBC = sc.broadcast(adIDMap)
    val adIDBinDim = 10000
    val adIDBinSize = numUsers/adIDBinDim
    val adWithClicksBC = sc.broadcast(adWithClicks)
    val advrCtrBC = sc.broadcast(advrCtr)
    val advrCtrBinDim = 100
    val advrCtrBinSize = advrCtr.reduce(math.max(_,_))/advrCtrBinDim
//  Form features from raw data:
    train.flatMap(arr => {
      val queryToken = queryTokenBC.value
      val queryFreq = queryFreqBC.value
      val queryCtr = queryCtrBC.value
      val queryWithClicks = queryWithClicksBC.value
      val despToken = despTokenBC.value
      val despCtr= despCtrBC.value
      val keywordToken = keywordTokenBC.value
      val keywordCtr = keywordCtrBC.value
      val keywordWithClicks = keywordWithClicksBC.value
      val titleToken = titleTokenBC.value
      val titleCtr = titleCtrBC.value
      val userProfile = userProfileBC.value
      val userFreq = userFreqBC.value
      val userCtr = userCtrBC.value
      val userWithClicks = userWithClicksBC.value
      val adFreq = adFreqBC.value
      val adCtr = adCtrBC.value
      val adWithClicks = adWithClicksBC.value
      val adIDMap = adIDMapBC.value
      val advrCtr = advrCtrBC.value
      
      var click = arr(0)
      var impression = arr(1)
      val feature = new ArrayBuilder.ofInt
      feature.sizeHint(25)
      //intercept
      feature += 0
      var offset = 1
      //AdCtr, D=adCtrBinDim
      feature += 
        math.min((adCtr(adIDMap(arr(2)))/adCtrBinSize).toInt, adCtrBinDim-1) + offset
      offset += adCtrBinDim
      //AdvrCtr, D=advrCtrBinDim
      feature += 
        math.min((advrCtr(arr(3))/advrCtrBinSize).toInt, advrCtrBinDim-1) + offset
      offset += advrCtrBinDim
      //QueryCtr, D=queryCtrBinDim
      feature += 
        math.min((queryCtr(arr(6))/queryCtrBinSize).toInt, queryCtrBinDim-1) + offset
      offset += queryCtrBinDim
      //UserCtr, D=userCtrBinDim
      feature += 
        math.min((userCtr(arr(10))/userCtrBinSize).toInt, userCtrBinDim-1) + offset
      offset += userCtrBinDim
      //WordCtr, D=keywordCtrBinDim
      feature += math.min((keywordCtr(arr(7))/keywordCtrBinSize).toInt, 
        keywordCtrBinDim-1) + offset
      offset += keywordCtrBinDim
      //binary User ID, Ad ID, query ID for records with clicks
      if (userWithClicks(arr(10))) feature += arr(10) + offset
      offset += numUsers
      if (adWithClicks(adIDMap(arr(2)))) feature += adIDMap(arr(2)) + offset
      offset += numAds
      if (queryWithClicks(arr(6))) feature += arr(6) + offset
      offset += numQueries
      
      //value-User, value-Query
      feature += math.min(arr(10)/userIDBinSize, userIDBinDim-1) + offset
      offset += userIDBinDim
      feature += math.min(arr(6)/queryIDBinSize, queryIDBinDim-1) + offset
      offset += queryIDBinDim
      //Number of tokens in query/title/description/keyword
      //Query token's length, D=20
      feature += math.min(queryToken(arr(6)).length, 19) + offset
      offset += 20
      //Title token's length, D=30
      feature += math.min(titleToken(arr(8)).length, 29) + offset
      offset += 30
      //Desp token's length, D=50
      feature += math.min(despToken(arr(9)).length, 49) + offset
      offset += 50
      //Keyword token's length, D=10
      feature += math.min(keywordToken(arr(7)).length, 9) + offset
      offset += 10
      //binary-Gender, binary-Age, binary-PositionDepth, binary-QueryTokens
      //Gender, D=3
      if (userProfile(arr(10)) != null) feature += userProfile(arr(10))(0) + offset
      offset += 3
      //Age, D=6
      if (userProfile(arr(10)) != null) feature += userProfile(arr(10))(1)-1 + offset
      offset += 6
      //binary Position-Depth
      feature += 6*arr(5)/arr(4)
      offset += 6
      //binary query tokens, D=queryTokenSize
      if (queryWithClicks(arr(6))) feature ++= queryToken(arr(6)).map(_+offset)
      offset += queryTokenSize
      if (keywordWithClicks(arr(7))) feature ++= keywordToken(arr(7)).map(_+offset)
      offset += keywordTokenSize
      
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
      val queryWithClicks = queryWithClicksBC.value
      val despToken = despTokenBC.value
      val despCtr= despCtrBC.value
      val keywordToken = keywordTokenBC.value
      val keywordCtr = keywordCtrBC.value
      val keywordWithClicks = keywordWithClicksBC.value
      val titleToken = titleTokenBC.value
      val titleCtr = titleCtrBC.value
      val userProfile = userProfileBC.value
      val userFreq = userFreqBC.value
      val userCtr = userCtrBC.value
      val userWithClicks = userWithClicksBC.value
      val adFreq = adFreqBC.value
      val adCtr = adCtrBC.value
      val adWithClicks = adWithClicksBC.value
      val adIDMap = adIDMapBC.value
      val advrCtr = advrCtrBC.value
      
      var click = arr(0)
      var impression = arr(1)
      val feature = new ArrayBuilder.ofInt
      feature.sizeHint(25)
      //intercept
      feature += 0
      var offset = 1
      //AdCtr, D=adCtrBinDim
      if (adIDMap.contains(arr(2))) {
        feature += 
          math.min((adCtr(adIDMap(arr(2)))/adCtrBinSize).toInt, adCtrBinDim-1) + offset
      }
      offset += adCtrBinDim
      //AdvrCtr, D=advrCtrBinDim
      if (arr(3) < advrCtr.length) {
        feature += 
          math.min((advrCtr(arr(3))/advrCtrBinSize).toInt, advrCtrBinDim-1) + offset
      }
      offset += advrCtrBinDim
      //QueryCtr, D=queryCtrBinDim
      feature += 
        math.min((queryCtr(arr(6))/queryCtrBinSize).toInt, queryCtrBinDim-1) + offset
      offset += queryCtrBinDim
      //UserCtr, D=userCtrBinDim
      feature += 
        math.min((userCtr(arr(10))/userCtrBinSize).toInt, userCtrBinDim-1) + offset
      offset += userCtrBinDim
      //WordCtr, D=keywordCtrBinDim
      feature += math.min((keywordCtr(arr(7))/keywordCtrBinSize).toInt, 
        keywordCtrBinDim-1) + offset
      offset += keywordCtrBinDim
      //binary User ID, Ad ID, query ID for records with clicks
      if (userWithClicks(arr(10))) feature += arr(10) + offset
      offset += numUsers
      if (adIDMap.contains(arr(2)) && adWithClicks(adIDMap(arr(2)))) {
        feature += adIDMap(arr(2)) + offset
      }
      offset += numAds
      if (queryWithClicks(arr(6))) feature += arr(6) + offset
      offset += numQueries
      
      //value-User, value-Query
      feature += math.min(arr(10)/userIDBinSize, userIDBinDim-1) + offset
      offset += userIDBinDim
      feature += math.min(arr(6)/queryIDBinSize, queryIDBinDim-1) + offset
      offset += queryIDBinDim
      //Number of tokens in query/title/description/keyword
      //Query token's length, D=20
      feature += math.min(queryToken(arr(6)).length, 19) + offset
      offset += 20
      //Title token's length, D=30
      feature += math.min(titleToken(arr(8)).length, 29) + offset
      offset += 30
      //Desp token's length, D=50
      feature += math.min(despToken(arr(9)).length, 49) + offset
      offset += 50
      //Keyword token's length, D=10
      feature += math.min(keywordToken(arr(7)).length, 9) + offset
      offset += 10
      //binary-Gender, binary-Age, binary-PositionDepth, binary-QueryTokens
      //Gender, D=3
      if (userProfile(arr(10)) != null) feature += userProfile(arr(10))(0) + offset
      offset += 3
      //Age, D=6
      if (userProfile(arr(10)) != null) feature += userProfile(arr(10))(1)-1 + offset
      offset += 6
      //binary Position-Depth
      feature += 6*arr(5)/arr(4)
      offset += 6
      //binary query tokens, D=queryTokenSize
      if (queryWithClicks(arr(6))) feature ++= queryToken(arr(6)).map(_+offset)
      offset += queryTokenSize
      if (keywordWithClicks(arr(7))) feature ++= keywordToken(arr(7)).map(_+offset)
      offset += keywordTokenSize
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
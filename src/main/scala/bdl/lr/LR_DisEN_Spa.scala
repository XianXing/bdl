package lr

import utilities._
import lr_preprocess._

import java.io._

import scala.math._
import scala.util.Sorting._

import org.apache.spark.{SparkContext, storage, HashPartitioner}
import org.apache.spark.SparkContext._

import org.apache.commons.cli._
import org.apache.commons.math.special.Gamma
import Functions._

object LR_DisEN_Spa extends Settings {
//  val trainingInputPath = "input/KDDCUP2010/kdda"
//  val testingInputPath = "input/KDDCUP2010/kdda.t"
  val trainingInputPath = "input/UCI_Adult/a9a"
  val testingInputPath = "input/UCI_Adult/a9a.t"
//  val trainingInputPath = "input/Ads/part-m-00000"
//  val testingInputPath = "input/Ads/part-m-00000"
    
  val outputDir = "output/"
  val numCores = 8
  val mode = "local[" + numCores + "]"
  val MAX_ITER = 5
  val lambda_init = 1.01f
  val gamma_init = 1.0f
  val eta_init = 0.1f
  val supTh = 20
  val numSlices = 80
  val ard_eta = false
  val ard_gamma = true
  val updateLambda = false
  val interval = 1
  val targetAUC = 0.92
  val l1 = false
  val l2 = true
  val logPath = "output/log" + "_LR_DisEN_" + "L1_" + l1 + "_L2_" + l2
  val JARS = Seq("sparkproject.jar")
  val stopCriteria = 1e-6
  val JOB_NAME = if (l1&&l2) "DistEN" else if(l1) "DistL1" else "DistL2"
    
  def main(args : Array[String]) {
    
    val currentTime = System.currentTimeMillis();
    val options = new Options()
	options.addOption(HELP, true, "print the help message")
	options.addOption(RUNNING_MODE_OPTION, true, "running mode option")
	options.addOption(TRAINING_OPTION, true, "training data input path/directory")
	options.addOption(TESTING_OPTION, true, "testing data output path/directory")
	options.addOption(OUTPUT_OPTION, true, "output path/directory")
	options.addOption(ITERATION_OPTION, true, "maximum number of iterations")
	options.addOption(L1_REGULARIZATION, false, "L1 option")
	options.addOption(L2_REGULARIZATION, false, "L2 option")
	options.addOption(ARD_GAMMA_OPTION, false, "gamma is updated as a vector")
	options.addOption(ARD_ETA_OPTION, false, "eta is updated as a vector")
	options.addOption(GAMMA, true, "initial guess for DisEN parameter gamma")
	options.addOption(ETA, true, "initial guess for DisEN parameter eta")
    options.addOption(LAMBDA, true, "initial guess for DisEN parameter lambda")
	options.addOption(FEATURE_SUPPORT_THRESHOLD, true, "feature support threshold")
	options.addOption(NUM_PARTITIONS_OPTION, true, "number of min splits of the input")
	options.addOption(NUM_CORES_OPTION, true, "number of cores to use")
    options.addOption(TARGET_AUC_OPTION, true, "convergence AUC")
    
	val parser = new GnuParser();
	val formatter = new HelpFormatter();
	val line = parser.parse(options, args);
	if (line.hasOption(HELP) || args.length == 0) {
	  formatter.printHelp("Help", options);
	  System.exit(0);
	}
	assert(line.hasOption(RUNNING_MODE_OPTION), "running mode needs to be specified")
	assert(line.hasOption(TRAINING_OPTION), "training data path needs to be specified")
	assert(line.hasOption(TESTING_OPTION), "testing data path needs to be specified")
	
	val mode = line.getOptionValue(RUNNING_MODE_OPTION)
	
	val trainingInputPath = line.getOptionValue(TRAINING_OPTION)
	
	val testingInputPath = line.getOptionValue(TESTING_OPTION)
	
	val outputDir = 
	  if (line.hasOption(OUTPUT_OPTION))
	    if (!line.getOptionValue(OUTPUT_OPTION).endsWith(PATH_SEPERATOR))
	      line.getOptionValue(OUTPUT_OPTION) + PATH_SEPERATOR
	    else
	      line.getOptionValue(OUTPUT_OPTION)
	  else
	    "output" + PATH_SEPERATOR
	  
	val MAX_ITER =
	  if (line.hasOption(ITERATION_OPTION))
	    line.getOptionValue(ITERATION_OPTION).toInt
	  else 10
	  
	val lambda_init = 
	  if (line.hasOption(LAMBDA))
	    line.getOptionValue(LAMBDA).toFloat
	  else 1.1f
	  
	val gamma_init =
	  if (line.hasOption(GAMMA))
	    line.getOptionValue(GAMMA).toFloat
	  else 1.1f
	  
	val eta_init =
	  if (line.hasOption(ETA))
	    line.getOptionValue(ETA).toFloat
	  else 1.1f
	  
	val supTh = 
	  if (line.hasOption(FEATURE_SUPPORT_THRESHOLD))
	    line.getOptionValue(FEATURE_SUPPORT_THRESHOLD).toInt
	  else 0
	  
	val numSlices =
	  if (line.hasOption(NUM_PARTITIONS_OPTION))
	    line.getOptionValue(NUM_PARTITIONS_OPTION).toInt
	  else 120
	  
	val numCores =
	  if (line.hasOption(NUM_CORES_OPTION))
	    math.min(line.getOptionValue(NUM_CORES_OPTION).toInt, numSlices)
	  else 120
	  
    val interval = 
	  if (line.hasOption(INTERVAL_OPTION))
	    line.getOptionValue(INTERVAL_OPTION).toInt
	  else 1
	
    val targetAUC =
	  if (line.hasOption(TARGET_AUC_OPTION))
	    line.getOptionValue(TARGET_AUC_OPTION).toDouble
	  else 0.9
	
	val ard_gamma = line.hasOption(ARD_GAMMA_OPTION)
	val ard_eta = line.hasOption(ARD_GAMMA_OPTION)
	
	val l1 = line.hasOption(L1_REGULARIZATION)
	val l2 = line.hasOption(L2_REGULARIZATION)
	val JOB_NAME = if (l1&&l2) "DistEN" else if(l1) "DistL1" else "DistL2"
	  
    val logPath = (outputDir + JOB_NAME + "_th_" + supTh + "_part_" + numSlices +
        "_lambda_" + lambda_init + "_gamma_" + gamma_init + "_eta_" + eta_init + "_ardEta_" + ard_eta +
        "_ardGamma_" + ard_gamma)
    
    System.setProperty("spark.local.dir", "/export/home/spark/spark/tmp")
    System.setProperty("spark.ui.port", "44717")
    
//	System.setProperty("spark.kryoserializer.buffer.mb", "512")
	System.setProperty("spark.worker.timeout", "360")
	System.setProperty("spark.storage.blockManagerSlaveTimeoutMs", "800000")
	System.setProperty("spark.akka.timeout", "60")
	System.setProperty("spark.default.parallelism", numCores.toString)
//	System.setProperty("spark.mesos.coarse", "true")
//	System.setProperty("spark.cores.max", "12")
//    System.setProperty("spark.serializer", "spark.KryoSerializer")
//    System.setProperty("spark.kryo.registrator", "util.MyRegistrator")
    val storageLevel = storage.StorageLevel.MEMORY_AND_DISK_SER
    val bwLog = new BufferedWriter(new FileWriter(new File(logPath)))
    val sc = new SparkContext(mode, JOB_NAME, System.getenv("SPARK_HOME"), JARS)
    val rawTrainingData = sc.textFile(trainingInputPath, numCores)
    
    // to filter out infrequent features
    val featureSet = Preprocess.wordCountKDD2010(rawTrainingData).filter(pair => pair._2 >= supTh)
      .map(pair => pair._1).collect
    val P = featureSet.size + 1
    
    val featureMap = sc.broadcast(featureSet.zipWithIndex.toMap)
    
    val trainingData = rawTrainingData.map(line => {
      (math.floor(math.random*numSlices).toInt, Preprocess.parseKDD2010Line(line, featureMap.value, false))
    }).groupByKey(numCores).mapValues(seq => seq.toArray)
    .partitionBy(new HashPartitioner(numCores)).persist(storageLevel)
    
    val trainingData_ColView = trainingData.mapValues(pair => 
      if (pair(0)._2.isBinary) Preprocess.toColViewBinary(pair)
      else Preprocess.toColView(pair)
      ).persist(storageLevel)
    
    val testingData = sc.textFile(testingInputPath, numCores).map(line =>
      Preprocess.parseKDD2010Line(line, featureMap.value, false)).persist(storageLevel)
    
    val splitsStats = trainingData.mapValues(array => array.length).collect
    val numTrain = splitsStats.map(pair => pair._2).reduce(_+_)  
    val testingDataPos = testingData.filter(_._1).persist(storageLevel)
    val testingDataNeg = testingData.filter(!_._1).persist(storageLevel)
    val numTestPos = testingDataPos.count.toInt
    val numTestNeg = testingDataNeg.count.toInt
    val numTest = numTestPos + numTestNeg
    
    val trainingFeatureSets = trainingData.mapValues(
        array => array.map(pair => pair._2.getIndices.toSet).reduce((k1, k2)=>k1.union(k2)))
        .persist(storageLevel)
        
    splitsStats.foreach(pair => println("partition " + pair._1 + " has " + pair._2 + " samples"))
    trainingFeatureSets.mapValues(array => array.size).collect.foreach(
        pair => println("partition " + pair._1 + " has " + pair._2 + " features"))
    println("number of: features " + P + "; training data " + numTrain + "; testing data " + numTest)
    bwLog.write("number of: features " + P + "; training data " + numTrain 
        + "; testing data " + numTest + '\n')
    println("Time elapsed after preprocessing " + (System.currentTimeMillis()-currentTime)*0.001 + "(s)")
    bwLog.write("Time elapsed after preprocessing " 
        + (System.currentTimeMillis()-currentTime)*0.001 + "(s)\n")
        
    //initialize the parameters
    var auc = 0.0
    var obj = 0.0
    var wg_dist = trainingFeatureSets.mapValues(set => {val arr = set.toArray; quickSort(arr); arr})
      .mapValues(array => ((array, array.map(i=>((math.random-0.5)/math.sqrt(100)).toFloat)), gamma_init))
      .persist(storageLevel)
    val lambda = Array.tabulate(P)(i => lambda_init)
    val w_updated = new Array[Float](P)
    var iterTime = System.currentTimeMillis()
    val gamma_sum = new Array[Float](P)
    val gamma = Array.tabulate(numSlices)(i => gamma_init)
    val eta = Array.tabulate(P)(i => eta_init)
    var w_broadcast = sc.broadcast(w_updated)
    var eta_broadcast = sc.broadcast(eta)
    var iter = 0
    while (iter < MAX_ITER && auc < targetAUC && obj > Double.NegativeInfinity) {
      val obj_th = math.pow(iter+1, -3)
//      val obj_th = 0
      val warmStart = iter>0
      val wg_dist_updated = trainingData_ColView.join(wg_dist, numCores).mapValues(pair => 
        EN_CD(pair._1, pair._2._1, w_broadcast.value, eta_broadcast.value, pair._2._2, (iter+1)*20,
        obj_th, l1&&iter>0, l2, ard_gamma&&iter>0, warmStart))
        .persist(storageLevel)
      
      if (ard_gamma&&iter>0) {
        wg_dist_updated.mapValues(tuple => tuple._3).collect.foreach(pair=>gamma(pair._1)=pair._2)
      }
      else {
        var i = 0; while (i<gamma.length) {gamma(i) = gamma_init; i += 1}
      }
      
      wg_dist_updated.flatMap(pair => pair._2._1.zip(pair._2._2.map(v => (pair._1, v))))
      .groupByKey(numCores).map(pair => (pair._1, 
            updateW_BS(pair._2, eta(pair._1), gamma, lambda(pair._1), l1, l2)))
        .collect.foreach(pair => w_updated(pair._1) = pair._2)
      
      w_broadcast = sc.broadcast(w_updated)
      wg_dist =  wg_dist_updated.mapValues(tuple => ((tuple._1, tuple._2), tuple._3))
      
      val innerIter = wg_dist_updated.map(tuple => tuple._2._4).reduce(_+_)*1.0/numSlices
      //evaluation the algorithm using likelihood and prediction accuracy
      obj = trainingData.flatMap(list => list._2.map(data => getLLH(data, w_broadcast.value))).reduce(_+_)
      val tp = testingDataPos.map(data => getBinPred(data._2, w_broadcast.value, 50))
    			    .reduce(_+=_).elements
      val fp = testingDataNeg.map(data => getBinPred(data._2, w_broadcast.value, 50))
     			.reduce(_+=_).elements
      auc = getAUC(tp, fp, numTestPos, numTestNeg)
      if (true) {
        if (ard_gamma) {
          println("gamma: " + Vector(gamma))
          bwLog.write("gamma: " + Vector(gamma) + "\n")
        }
        else {
          println("gamma(avg): " + gamma.sum/gamma.length)
          bwLog.write("gamma(avg): " + gamma.sum/gamma.length + "\n")
        }
        if (ard_eta) {
          println("eta: " + Vector(eta))
          bwLog.write("eta: " + Vector(eta) + "\n")
        }
        else {
          println("eta(avg): " + eta.sum/eta.length)
          bwLog.write("eta(avg): " + eta.sum/eta.length + "\n")
        }
      }
      println("lambda(avg): " + lambda.sum/lambda.length)
      bwLog.write("lambda(avg): " + lambda.sum/lambda.length + "\n")
      val time = (System.currentTimeMillis() - iterTime)*0.001
      if (w_updated.length < 200) {
        println("w: " + Vector(w_updated))
        bwLog.write("w: " + Vector(w_updated) + "\n")
      }
      else {
        println("w_l2norm: " + Vector(w_updated).l2Norm)
        bwLog.write("w_l2norm: " + Vector(w_updated).l2Norm + "\n")
      }
      println("Average number of inner iterations: " + innerIter)
      println("Iter: " + iter + " time elapsed: " + time + " AUC: " + auc + " obj: " + obj)
      bwLog.write("Average number of inner iterations: " + innerIter + '\n')
      bwLog.write("Iter: " + iter + " time elapsed: " + time + " AUC: " + auc + " obj: " + obj + '\n')
      iter += 1
      iterTime = System.currentTimeMillis()
    }
    
//    bwLog.write("Final w " + w + '\n')
    bwLog.write("Total time elapsed " + (System.currentTimeMillis()-currentTime)*0.001 + "(s)")
    bwLog.close()
    println("Total time elapsed " + (System.currentTimeMillis()-currentTime)*0.001 + "(s)")
    System.exit(0)
  }
}
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

object LR_BDL_CD_Spa extends Settings {
//  val trainingInputPath = "input/KDDCUP2010/kdda"
//  val testingInputPath = "input/KDDCUP2010/kdda.t"
  val trainingInputPath = "input/UCI_Adult/a9a"
  val testingInputPath = "input/UCI_Adult/a9a.t"
//  val trainingInputPath = "input/Ads/part-m-00000"
//  val testingInputPath = "input/Ads/part-m-00000"
  
  val outputDir = "output/"
  val numCores = 8
  val mode = "local[" + numCores + "]"
  val MAX_ITER = 10
  val lambda_init = 1.01f
  val gamma_init = 1.0f
  val alpha_init = 0.01f
  val beta = 0.01f
  val supTh = 10
  val numSlices = 80
  val bayes = true
  val updateLambda = false
  val interval = 1
  val targetAUC = 0.92
  val l1 = false
  val l2 = true
  val JOB_NAME = "Logistic Regression (BDL)"
  val logPath = "output/log" + JOB_NAME
  val JARS = Seq("sparkproject.jar")
  val stopCriteria = 1e-6
    
  def main(args : Array[String]) {
    
    val currentTime = System.currentTimeMillis();
    val options = new Options()
	options.addOption(HELP, false, "print the help message")
	options.addOption(RUNNING_MODE_OPTION, true, "running mode option")
	options.addOption(TRAINING_OPTION, true, "training data input path/directory")
	options.addOption(TESTING_OPTION, true, "testing data output path/directory")
	options.addOption(OUTPUT_OPTION, true, "output path/directory")
	options.addOption(UPDATE_LAMBDA_OPTION, false, "update lambda option (only for l2)")
	options.addOption(ITERATION_OPTION, true, "maximum number of iterations")
	options.addOption(L2_REGULARIZATION, false, "L2 option")
	options.addOption(L1_REGULARIZATION, false, "L1 option")
	options.addOption(LAMBDA, true, "initial guess for lambda")
	options.addOption(GAMMA, true, "initial guess for gamma")
	options.addOption(FEATURE_SUPPORT_THRESHOLD, true, "feature support threshold")
	options.addOption(NUM_PARTITIONS_OPTION, true, "number of min splits of the input")
	options.addOption(NUM_CORES_OPTION, true, "number of cores to use")
    options.addOption(TARGET_AUC_OPTION, true, "convergence AUC")
    options.addOption(BAYESIAN_OPTION, false, "Bayesian update option")
    
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
	  else 0.1f
	  
	val gamma_init =
	  if (line.hasOption(GAMMA))
	    line.getOptionValue(GAMMA).toFloat
	  else 0.1f
	  
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
	    line.getOptionValue(NUM_CORES_OPTION).toInt
	  else 120
	  
    val interval = 
	  if (line.hasOption(INTERVAL_OPTION))
	    line.getOptionValue(INTERVAL_OPTION).toInt
	  else 1
	
    val targetAUC =
	  if (line.hasOption(TARGET_AUC_OPTION))
	    line.getOptionValue(TARGET_AUC_OPTION).toDouble
	  else 0.9
	
	val bayes = line.hasOption(BAYESIAN_OPTION)
	
	val l1 = line.hasOption(L1_REGULARIZATION)
	val l2 = line.hasOption(L2_REGULARIZATION) || !l1
	val updateLambda = l2&&line.hasOption(UPDATE_LAMBDA_OPTION)
	
    val JOB_NAME = "LR-BDL-Spa-ARD"
    val logPath = 
	  if (l1) {
	    (outputDir + JOB_NAME + "_th_" + supTh + "_part_" + numSlices  + "_L1" + "_Bayes_" + bayes
	        + "_lambda_" + lambda_init + "_gamma_" + gamma_init)
	  }
	  else {
	    (outputDir + JOB_NAME + "_th_" + supTh + "_part_" + numSlices + "_L2" + "_Bayes_" + bayes
	        + "_lambda_" + lambda_init + "_gamma_" + gamma_init + "_updateLambda_" + updateLambda)
	  }
	
	System.setProperty("spark.ui.port", "44717")
    System.setProperty("spark.local.dir", "/grid/a/mapred/tmp")
    
//	System.setProperty("spark.kryoserializer.buffer.mb", "128")
	System.setProperty("spark.worker.timeout", "360")
	System.setProperty("spark.storage.blockManagerSlaveTimeoutMs", "800000")
	System.setProperty("spark.akka.timeout", "60")
	System.setProperty("spark.default.parallelism", numCores.toString)
//	System.setProperty("spark.mesos.coarse", "true")
//	System.setProperty("spark.cores.max", "300")
//    System.setProperty("spark.serializer", "spark.KryoSerializer")
//    System.setProperty("spark.kryo.registrator", "util.MyRegistrator")
    val storageLevel = storage.StorageLevel.MEMORY_AND_DISK_SER
    val bwLog = new BufferedWriter(new FileWriter(new File(logPath)))
    val environment : Map[String, String] = null
    val sc = new SparkContext(mode, JOB_NAME, System.getenv("SPARK_HOME"), JARS, 
        environment)
    val rawTrainingData = sc.textFile(trainingInputPath, numCores)
    
    // to filter out infrequent features
    val stats = Preprocess.wordCountKDD2010(rawTrainingData).persist(storageLevel)
    println("P original: " + stats.count, " supTh: " + supTh)
    val featureSet = stats.filter(pair => pair._2 >= supTh).map(pair => pair._1).collect
    val P = featureSet.size + 1
    
    val featureMap = sc.broadcast(featureSet.zipWithIndex.toMap)
    
    val trainingData = rawTrainingData.map(line => {
      (math.floor(math.random*numSlices).toInt, 
          Preprocess.parseKDD2010Line(line, featureMap.value, false))
    }).groupByKey(numCores).mapValues(seq => seq.toArray)
    .partitionBy(new HashPartitioner(numCores)).persist(storageLevel)
    
    val trainingData_ColView = trainingData.mapValues(pair => Preprocess.toColView(pair))
      .persist(storageLevel)
    
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
    
    val groupedFeatures = Array.ofDim[Boolean](P, numSlices)
    trainingFeatureSets.flatMap(pair => 
      pair._2.map(feature => {val b = new Array[Boolean](numSlices); b(pair._1) = true; (feature->b)}))
      .reduceByKey((arr1, arr2) => 
        {var i = 0; val arr = new Array[Boolean](numSlices)
         while (i<numSlices) {arr(i) = arr1(i) | arr2(i); i += 1}; arr})
         .collect.foreach(pair => {groupedFeatures(pair._1) = pair._2})
    
    val featureCount = new Array[Int](P)
    var i = 0
      while (i < P) {
        var j = 0
        while (j < numSlices) {
          if (groupedFeatures(i)(j)) {
            featureCount(i) += 1
          }
          j += 1
        }
        i += 1
      }
         
    //initialize the parameters
    var wg_dist = trainingFeatureSets.mapValues(set => {val arr = set.toArray; quickSort(arr); arr})
      .mapValues(array => (array, new Array[Float](array.length), 
          array.map(i=>gamma_init))).persist(storageLevel)
    val lambda = Array.tabulate(P)(i => lambda_init)
    var alpha = alpha_init
    
    splitsStats.foreach(pair => println("partition " + pair._1 + " has " + pair._2 + " samples"))
    trainingFeatureSets.mapValues(array => array.size).collect.foreach(
        pair => println("partition " + pair._1 + " has " + pair._2 + " features"))
    println("number of: features " + P + "; training data " + numTrain + "; testing data " + numTest)
    bwLog.write("number of: features " + P + "; training data " + numTrain 
        + "; testing data " + numTest + '\n')
    println("Time elapsed after preprocessing " + (System.currentTimeMillis()-currentTime)*0.001 + "(s)")
    bwLog.write("Time elapsed after preprocessing " 
        + (System.currentTimeMillis()-currentTime)*0.001 + "(s)\n")
        
    var iterTime = System.currentTimeMillis()
    var iter = 0
    var auc = 0.0
    var obj = 0.0
    val w_updated = new Array[Float](P)
    val w_average = new Array[Float](P)
    val variance = new Array[Float](P)
    
    var w_broadcast = sc.broadcast(w_updated)
    var var_broadcast = sc.broadcast(variance)
    var w_ave_broadcast = sc.broadcast(w_average)
    
    while (iter < MAX_ITER && auc < targetAUC && obj > Double.NegativeInfinity) {
      val filter_th = 0.01*math.pow(iter, 3)
      val warmStart = iter > 0
      val obj_th = math.pow(iter+1, -2)
      
      val wg_dist_updated = trainingData_ColView.join(wg_dist, numCores).mapValues(tuple => 
            BDL_CD(tuple._1, tuple._2, w_broadcast.value, var_broadcast.value, alpha, beta, 200, 
                gamma_init, bayes&&iter>0, warmStart, obj_th)).persist(storageLevel)
      
      val w_stat = wg_dist_updated.mapValues(
            tuple => { var i = 0
                       val stat = new Array[Tuple3[Float, Float, Float]](tuple._1.length)
                       while (i < tuple._1.length) {
                         stat(i) = 
                           if (iter > 0 && bayes) (tuple._2(i)*tuple._3(i), tuple._2(i), tuple._3(i))
                           else (tuple._2(i)*gamma_init, tuple._2(i), gamma_init)
                         i += 1}
                       (tuple._1, stat)}
            ).flatMap(pair => pair._2._1.zip(pair._2._2))
            .reduceByKey((pair1, pair2)=>(pair1._1+pair2._1, pair1._2 + pair2._2, pair1._3 + pair2._3))
            .collect
      
      if (bayes) {
        w_stat.par.foreach(pair => w_average(pair._1) = pair._2._2/featureCount(pair._1))
        w_ave_broadcast = sc.broadcast(w_average)
        val w_var = wg_dist_updated.mapValues( 
            tuple => { var i = 0; val vari = new Array[Float](tuple._1.length)
                       while (i < tuple._1.length) {
                         val residual = w_ave_broadcast.value(tuple._1(i)) - tuple._2(i)
                         vari(i) = residual*residual
                         i += 1}
                       (tuple._1, vari)}
        ).flatMap(pair => pair._2._1.zip(pair._2._2)).reduceByKey(_+_).collect
        w_var.par.foreach(pair => variance(pair._1) = pair._2/featureCount(pair._1))
        var_broadcast = sc.broadcast(variance)
      }
      
      if (l1) {
        w_stat.par.foreach(pair => w_updated(pair._1) = 
          if (pair._2._1 > lambda(pair._1))
            (pair._2._1 - lambda(pair._1))/pair._2._3
          else if (pair._2._1 < -lambda(pair._1))
            (pair._2._1 + lambda(pair._1))/pair._2._3
          else 0.0f )
      }
      else {
        w_stat.par.foreach(pair => {
          w_updated(pair._1) = pair._2._1/(lambda(pair._1) + pair._2._3)
          lambda(pair._1) = 
            if (updateLambda && iter > 0) 
              (0.001f + 0.5f)/(0.001f + 0.5f*w_updated(pair._1)*w_updated(pair._1))
            else lambda_init
        })
      }
      w_broadcast = sc.broadcast(w_updated)
      wg_dist =  wg_dist_updated.mapValues(tuple => (tuple._1, tuple._2, tuple._3))
      
      val innerIter = wg_dist_updated.map(tuple => tuple._2._5).reduce(_+_)*1.0/numSlices
      val innerObj = wg_dist_updated.map(tuple => tuple._2._4).reduce(_+_)*1.0/numSlices
      //evaluation the algorithm using likelihood and prediction accuracy
      obj = trainingData.flatMap(list => list._2.map(data => getLLH(data, w_broadcast.value)))
                            .reduce(_+_)
      val tp = testingDataPos.map(data => 
        getBinPred(data._2, w_broadcast.value, 50)).reduce(_+=_).elements
      val fp = testingDataNeg.map(data => 
        getBinPred(data._2, w_broadcast.value, 50)).reduce(_+=_).elements
      auc = getAUC(tp, fp, numTestPos, numTestNeg)
      
      println("averaged global variance: " + variance.sum/variance.length)
      println("lambda(avg): " + lambda.sum/lambda.length)
      bwLog.write("lambda(avg): " + lambda.sum/lambda.length + "\n")
      println("alpha: " + alpha)
      bwLog.write("alpha: " + alpha + "\n")
      val time = (System.currentTimeMillis() - iterTime)*0.001
      if (w_updated.length < 200) {
        println("w: " + Vector(w_updated))
        println("w_l2norm: " + Vector(w_updated).l2Norm)
        bwLog.write("w_l2norm: " + Vector(w_updated).l2Norm + "\n")
        println("variance: " + Vector(variance))
        wg_dist_updated.mapValues(tuple => tuple._3).collect.foreach(
            pair => if (math.random < 0.1) println("gamma in par: " + pair._1 + Vector(pair._2)))
      }
      else {
        println("w_l2norm: " + Vector(w_updated).l2Norm)
        bwLog.write("w_l2norm: " + Vector(w_updated).l2Norm + "\n")
      }
      
      println("Average number of inner iterations: " + innerIter + "\t average inner obj: " + innerObj)
      println("Iter: " + iter + " time elapsed: " + time + " AUC: " + auc + " obj: " + obj)
      bwLog.write("Average number of inner iterations: " + innerIter 
          + " average inner obj: " + innerObj + '\n')
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
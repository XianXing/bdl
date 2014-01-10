package lr

import utilities._
import lr_preprocess._
import Functions._
import java.io._

import scala.math._
import scala.util.Sorting._

import org.apache.spark.{SparkContext, storage, HashPartitioner}
import org.apache.spark.SparkContext._

import org.apache.commons.cli._
import org.apache.commons.math.special.Gamma

object LR_BADMM extends Settings {
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
  val alpha_init = 1f
  val beta = 1f
  val supTh = 10
  val numSlices = 80
  val bayes = false
  val updateLambda = false
  val interval = 1
  val targetAUC = 0.92
  val l1 = false
  val l2 = true
  val CD = false
  val JOB_NAME = 
    if (bayes) "Logistic Regression (BDL)" 
    else "Logistic Regression (ADMM)"
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
	options.addOption(ITERATION_OPTION, true, "maximum number of iterations")
	options.addOption(CD_OPTION, false, "Coordinate descent option")
	options.addOption(L2_REGULARIZATION, false, "L2 option")
	options.addOption(L1_REGULARIZATION, false, "L1 option")
	options.addOption(LAMBDA, true, "initial guess for lambda")
	options.addOption(GAMMA, true, "initial guess for gamma")
	options.addOption(FEATURE_SUPPORT_THRESHOLD, true, "feature support threshold")
	options.addOption(NUM_PARTITIONS_OPTION, true, "number of min splits of the input")
	options.addOption(NUM_CORES_OPTION, true, "number of cores to use")
    options.addOption(TARGET_AUC_OPTION, true, "convergence AUC")
    options.addOption(BAYESIAN_OPTION, false, "Bayesian ADMM option")
    options.addOption(UPDATE_LAMBDA_OPTION, false, "update lambda option")
	options.addOption(INTERVAL_OPTION, true,
	    "number of iterations after which to output AUC performance")
	
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
	
	val bayes = line.hasOption(BAYESIAN_OPTION)
	val CD = line.hasOption(CD_OPTION)
	val l1 = line.hasOption(L1_REGULARIZATION)
	val l2 = line.hasOption(L2_REGULARIZATION) || !l1
	val updatedLambda = l2 && line.hasOption(UPDATE_LAMBDA_OPTION)
	
    val JOB_NAME = if (bayes) "LR-BDL" else "LR-ADMM"
    val logPath = 
	  if (CD) {
	    (outputDir + JOB_NAME + "_CD" + "_th_" + supTh + "_part_" + numSlices + "_L2_" + l2
	        + "_lambda_" + lambda_init + "_gamma_" + gamma_init)
	  }
	  else {
	    (outputDir + JOB_NAME + "_CG" + "_th_" + supTh + "_part_" + numSlices + "_L2_" + l2
	        + "_lambda_" + lambda_init + "_gamma_" + gamma_init + "_updateLambda_" + updatedLambda)
	  }
	
	System.setProperty("spark.local.dir", "/export/home/spark/spark/tmp")
	System.setProperty("spark.ui.port", "44717")
//    System.setProperty("spark.cores.max", numCores.toString())
    
    
//	System.setProperty("spark.kryoserializer.buffer.mb", "128")
	System.setProperty("spark.worker.timeout", "360")
	System.setProperty("spark.storage.blockManagerSlaveTimeoutMs", "800000")
	System.setProperty("spark.akka.timeout", "60")
	System.setProperty("spark.default.parallelism", numCores.toString)
//	System.setProperty("spark.mesos.coarse", "true")
//    System.setProperty("spark.serializer", "spark.KryoSerializer")
//    System.setProperty("spark.kryo.registrator", "util.MyRegistrator")
    val storageLevel = storage.StorageLevel.MEMORY_AND_DISK_SER
    val bwLog = new BufferedWriter(new FileWriter(new File(logPath)))
    val sc = new SparkContext(mode, JOB_NAME, System.getenv("SPARK_HOME"), JARS)
    
	val rawtrainingData_RowView = sc.textFile(trainingInputPath, numCores)
    // to filter out infrequent features
	
	val featureSet = Preprocess.wordCountAds(rawtrainingData_RowView)
      .filter(pair => pair._2 >= supTh).map(pair => pair._1).collect
//    val featureSet = Preprocess.wordCountKDD2010(rawtrainingData_RowView)
//      .filter(pair => pair._2 >= supTh).map(pair => pair._1).collect
    val P = featureSet.size + 1
    
    val featureMap = sc.broadcast(featureSet.zipWithIndex.toMap)
    
//    val trainingData = rawtrainingData_RowView.map(line => {
//      (math.floor(math.random*numSlices).toInt, 
//          Preprocess.parseKDD2010Line(line, featureMap.value, true))
//    }).groupByKey(numSlices).mapValues(seq => seq.toArray)
//    .partitionBy(new HashPartitioner(numCores)).persist(storageLevel)
//    val testingData = sc.textFile(testingInputPath, numCores).map(line =>
//      Preprocess.parseKDD2010Line(line, featureMap.value, true)).persist(storageLevel)
      
    val trainingData = rawtrainingData_RowView.map(line =>
      (math.floor(math.random*numSlices).toInt, 
          Preprocess.parseAdsLine(line, featureMap.value))
      ).groupByKey(numCores).mapValues(seq => seq.toArray)
      .partitionBy(new HashPartitioner(numCores)).persist(storageLevel)
    val testingData = sc.textFile(testingInputPath, numCores).map(line =>
      (line.split("\t")(0), 
          Preprocess.parseAdsLine(line, featureMap.value))).persist(storageLevel)
    
    val trainingData_ColView = trainingData.mapValues(pair => 
      if (pair(0)._2.isBinary) Preprocess.toColViewBinary(pair)
      else Preprocess.toColView(pair)
      ).persist(storageLevel)
    
    val splitsStats = trainingData.mapValues(array => array.length).collect
    val numTrain = splitsStats.map(pair => pair._2).sum
    val testingDataPos = testingData.filter(_._2._1).persist(storageLevel)
    val testingDataNeg = testingData.filter(!_._2._1).persist(storageLevel)
    val numTestPos = testingDataPos.count.toInt
    val numTestNeg = testingDataNeg.count.toInt
    val numTest = numTestPos + numTestNeg
    
    splitsStats.foreach(pair => println("partition " + pair._1 + " has " + pair._2 + " samples"))
    
    println("number of: features " + P + "; training data " + numTrain  + "; testing data " + numTest
        + " pos " + numTestPos + " neg " + numTestNeg)
    bwLog.write("number of: features " + P + "; training data " + numTrain 
        + "; testing data " + numTest + " pos " + numTestPos + " neg " + numTestNeg + '\n')
    println("Time elapsed after preprocessing " + (System.currentTimeMillis()-currentTime)*0.001 + "(s)")
    bwLog.write("Time elapsed after preprocessing " 
        + (System.currentTimeMillis()-currentTime)*0.001 + "(s)\n")
    
    //initialize the parameters
    var wug_dist = trainingData.mapValues(data => 
      Tuple3(new Array[Float](P), new Array[Float](P), gamma_init)).persist(storageLevel)
        
    val lambda = Array.tabulate(P)(i => lambda_init)
    var alpha = alpha_init
        
    var iterTime = System.currentTimeMillis()
    var iter = 0
    var auc = 0.0
    var old_auc = -10.0
    var obj = 0.0
    val seqP = Array.tabulate(P)(i => i)
    val w_updated = new Array[Float](P)
    val w_average = new Array[Float](P)
    val variance = new Array[Float](P)
    val gamma = Array.tabulate(numSlices)(i => gamma_init)
    var variance_sum = 0f
    var w_broadcast = sc.broadcast(w_updated)
    var w_ave_broadcast = sc.broadcast(w_average)
    
    while (iter < MAX_ITER && auc < targetAUC && math.abs(auc-old_auc) > 1e-5) {
      val filter_th = 0.01*math.pow(iter, 3)
      val warmStart = iter > 0
      val obj_th = math.pow(iter+1, -2)
//      val obj_th = 0
      val wug_dist_updated = 
        if (CD)
          trainingData_ColView.join(wug_dist, numCores).mapValues(
              pair => ADMM_CD(pair._1, pair._2, w_broadcast.value, 
                  if (bayes&&iter>0) variance_sum else gamma_init, alpha, beta, 20, 
                  bayes&&iter>0, warmStart, obj_th)).persist(storageLevel)
        else
          trainingData.join(wug_dist, numCores).mapValues(
              pair => ADMM_CG(pair._1, pair._2, w_broadcast.value, 
                  if (bayes&&iter>0) variance_sum else gamma_init, 
                  10, bayes&&iter>0, warmStart, obj_th)).persist(storageLevel)
      
      if (bayes&&iter>0) {
        wug_dist_updated.mapValues(tuple => tuple._3).collect.foreach(pair=>gamma(pair._1)=pair._2)
        
        //update alpha using empirical Bayesian estimate
//        val g = wug_dist_updated.map(tuple => -math.log(1+1/(2*beta)*tuple._2._5) +
//          Gamma.digamma(alpha+tuple._2._1.size*0.5)-Gamma.digamma(alpha)).sum
//        val gg = wug_dist_updated.map(tuple => 
//          Gamma.trigamma(alpha+tuple._2._1.size*0.5)-Gamma.trigamma(alpha)).sum
//        val alpha_updated = alpha - g/gg
//        alpha = if (alpha_updated > alpha) alpha_updated.toFloat else alpha
      }
      
      val w_stat = wug_dist_updated.mapValues(
          tuple => { var i = 0
            val ave = new Array[Pair[Float, Float]](P)
            while (i < P) {
              ave(i) = if (bayes && iter>0) (tuple._1(i)*tuple._3, tuple._1(i)) 
              else ((tuple._1(i) + tuple._2(i))*gamma_init, tuple._1(i))
              i += 1}
          (seqP, ave)}
          ).flatMap(pair => pair._2._1.zip(pair._2._2))
          .reduceByKey((pair1, pair2)=>(pair1._1+pair2._1, pair1._2 + pair2._2)).collect
      
      if (bayes) {
        w_stat.par.foreach(pair => w_average(pair._1) = pair._2._2/numSlices)
        w_ave_broadcast = sc.broadcast(w_average)
        val w_var = wug_dist_updated.mapValues( 
            tuple => { var i = 0
              val vari = new Array[Float](P)
              while (i < P) {
                val residual = w_ave_broadcast.value(i) - tuple._1(i)
                vari(i) = residual*residual
                i += 1}
            (seqP, vari)}
            ).flatMap(pair => pair._2._1.zip(pair._2._2)).reduceByKey(_+_).collect
        w_var.par.foreach(pair => variance(pair._1) = pair._2/numSlices)
        variance_sum = variance.sum
      }
      
      val gamma_sum = gamma.sum
      if (l1) {
        w_stat.par.foreach(pair => w_updated(pair._1) = 
          if (pair._2._1 > lambda(pair._1))
            (pair._2._1 - lambda(pair._1))/gamma_sum
          else if (pair._2._1 < -lambda(pair._1))
            (pair._2._1 + lambda(pair._1))/gamma_sum
          else 0.0f )
      }
      else {
        w_stat.par.foreach(pair => {
          w_updated(pair._1) = pair._2._1/(lambda(pair._1) + gamma_sum)
          lambda(pair._1) = 
            if (updateLambda && iter > 0) 
              (0.5f + 0.5f)/(0.5f + 0.5f*w_updated(pair._1)*w_updated(pair._1))
            else lambda_init
        })
      }
      w_broadcast = sc.broadcast(w_updated)
      wug_dist =  wug_dist_updated.mapValues(tuple => (tuple._1, tuple._2, tuple._3))
      val innerIter = wug_dist_updated.map(tuple => tuple._2._5).reduce(_+_)*1.0/numSlices
      val w_global = Vector(w_updated)
      obj = wug_dist_updated.map(tuple => tuple._2._4).reduce(_+_) 
        - lambda.sum/(2*P)*w_global.l2Norm
        
      //evaluation the algorithm using likelihood and prediction accuracy
      val tp = testingDataPos.map(data => 
        getBinPred(data._2._2, w_broadcast.value, 50)).reduce(_+=_).elements
      val fp = testingDataNeg.map(data => 
        getBinPred(data._2._2, w_broadcast.value, 50)).reduce(_+=_).elements
           				
      old_auc = auc
      auc = getAUC(tp, fp, numTestPos, numTestNeg)
      
      println("average gamma: " + gamma.sum/gamma.length)
      bwLog.write("average gamma: " + gamma.sum/gamma.length + "\n")
      println("lambda(avg): " + lambda.sum/lambda.length)
      bwLog.write("lambda(avg): " + lambda.sum/lambda.length + "\n")
      println("alpha: " + alpha)
      bwLog.write("alpha: " + alpha + "\n")
      println("variance(avg): " + variance_sum/P)
      bwLog.write("variance(avg): " + variance_sum/P + "\n")
      val time = (System.currentTimeMillis() - iterTime)*0.001
      if (w_updated.length < 200) {
        println("w: " + Vector(w_updated))
        println("w_l2norm: " + Vector(w_updated).l2Norm)
        bwLog.write("w_l2norm: " + Vector(w_updated).l2Norm + "\n")
      }
      else {
        println("w_l2norm: " + Vector(w_updated).l2Norm)
        bwLog.write("w_l2norm: " + Vector(w_updated).l2Norm + "\n")
      }
//      bwLog.write("w: " + w + "\n")
      println("Average number of inner iterations: " + innerIter)
      println("Iter: " + iter + " time elapsed: " + time + " AUC: " + auc + " obj: " + obj)
      bwLog.write("Average number of inner iterations: " + innerIter + '\n')
      bwLog.write("Iter: " + iter + " time elapsed: " + time + " AUC: " + auc + " obj: " + obj + '\n')
      iter += 1
      iterTime = System.currentTimeMillis()
    }
	//output predicted score
    testingData.map{
      case(id, (response, features)) => {
        val stringBuilder = new StringBuilder()
        stringBuilder.append(id).append("\t").append(response).append("\t")
        .append(sigmoid(features.dot(Vector(w_broadcast.value))))
        stringBuilder.toString
      }
    }.saveAsTextFile(outputDir+"prediction_results")
    
    bwLog.write("Total time elapsed " + (System.currentTimeMillis()-currentTime)*0.001 + "(s)")
    bwLog.close()
    println("Total time elapsed " + (System.currentTimeMillis()-currentTime)*0.001 + "(s)")
    System.exit(0)
  }
}
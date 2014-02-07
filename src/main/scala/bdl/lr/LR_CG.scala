package lr

import utilities._
import lr_preprocess._

import java.io._

import scala.math
import org.apache.spark.{SparkContext, storage, HashPartitioner}
import org.apache.spark.SparkContext._

import org.apache.commons.cli._
import Functions._

object LRv3 extends Settings {
//  var trainingInputPath = "input/IJCNN/ijcnn1.tr"
//  var testingInputPath = "input/IJCNN/ijcnn1.t"
  val trainingDataDir = "../datasets/UCI_Adult/a9a"
  val testingDataDir = "../datasets/UCI_Adult/a9a.t"
//  var trainingInputPath = "input/KDDCUP2010/kdda"
//  var testingInputPath = "input/KDDCUP2010/kdda.t"
  
  val outputDir = "output/"
  val mode = "local[8]"
  val MAX_ITER = 500
  val CG = true
  val LBFGS = false
  val stepSize = 0.0001f
  val lambda = 0.0f
  val suppportThres = 0
  val numCores = 8
  val hasInput = true
  val interval = 1
  val JARS = Seq("sparkproject.jar")
  val stopCriteria = 1e-6f
  val logPath = "output/log" + "_CG_" + CG + "_LBFGS_" + LBFGS
  val JOB_NAME = "logistic regression"
  val targetAUC = 0.92
  
  def main(args : Array[String]) {
    
    val startTime = System.currentTimeMillis();
    
//    val options = new Options()
//	options.addOption(HELP, true, "print the help message")
//	options.addOption(RUNNING_MODE_OPTION, true, "running mode option")
//	options.addOption(TRAINING_OPTION, true, "training data input path/directory")
//	options.addOption(TESTING_OPTION, true, "testing data output path/directory")
//	options.addOption(OUTPUT_OPTION, true, "output path/directory")
//    options.addOption(CG_OPTION, false, "conjugate gradient method")
//    options.addOption(QN_OPTION, false, "quasi-Newton method (L-BFGS)")
//	options.addOption(ITERATION_OPTION, true, "maximum number of iterations")
//	options.addOption(REGULARIZER_OPTION, true, "regularization parameter")
//	options.addOption(FEATURE_SUPPORT_THRESHOLD, true, "feature support threshold")
//	options.addOption(NUM_CORES, true, "number of cores to use")
//	options.addOption(INTERVAL, true, "number of iterations after which to output AUC performance")
//	options.addOption(TARGET_AUC, true, "convergence AUC")
//	
//	val parser = new GnuParser();
//	val formatter = new HelpFormatter();
//	val line = parser.parse(options, args);
//	if (line.hasOption(HELP) || args.length == 0) {
//	  formatter.printHelp(JOB_NAME, options);
//	  System.exit(0);
//	}
//	assert(line.hasOption(RUNNING_MODE_OPTION), "running mode needs to be specified")
//	assert(line.hasOption(TRAINING_OPTION), "training data path needs to be specified")
//	assert(line.hasOption(TESTING_OPTION), "testing data path needs to be specified")
//	
//	val mode = line.getOptionValue(RUNNING_MODE_OPTION)
//	
//	val trainingInputPath = line.getOptionValue(TRAINING_OPTION)
//	
//	val testingInputPath = line.getOptionValue(TESTING_OPTION)
//	
//	val outputDir = 
//	  if (line.hasOption(OUTPUT_OPTION))
//	    if (!line.getOptionValue(OUTPUT_OPTION).endsWith(PATH_SEPERATOR))
//	      line.getOptionValue(OUTPUT_OPTION) + PATH_SEPERATOR
//	    else
//	      line.getOptionValue(OUTPUT_OPTION)
//	  else
//	    "output" + PATH_SEPERATOR
//	  
//	val MAX_ITER =
//	  if (line.hasOption(ITERATION_OPTION))
//	    line.getOptionValue(ITERATION_OPTION).toInt
//	  else 10
//	  
//	val CG = line.hasOption(CG_OPTION)
//	    
//	val LBFGS = line.hasOption(QN_OPTION)
//
//	val lambda = 
//	  if (line.hasOption(REGULARIZER_OPTION))
//	    line.getOptionValue(REGULARIZER_OPTION).toFloat
//	  else
//	    0.1f
//	    
//    val suppportThres = 
//	  if (line.hasOption(FEATURE_SUPPORT_THRESHOLD))
//	    line.getOptionValue(FEATURE_SUPPORT_THRESHOLD).toInt
//	  else 0
//	  
//	val numCores =
//	  if (line.hasOption(NUM_CORES))
//	    line.getOptionValue(NUM_CORES).toInt
//	  else 8
//	  
//    val targetAUC =
//	  if (line.hasOption(TARGET_AUC))
//	    line.getOptionValue(TARGET_AUC).toDouble
//	  else 0.9
//	  
//    val interval = 
//	  if (line.hasOption(INTERVAL))
//	    line.getOptionValue(INTERVAL).toInt
//	  else 1
//	    
//	val logPath =
//	  if (CG)
//	    outputDir + "log_CG_iter_" + MAX_ITER + "_th_" + suppportThres + "_lambda_" + lambda
//	  else if (LBFGS)
//	    outputDir + "log_LBFGS_iter_" + MAX_ITER + "_th_" + suppportThres + "_lambda_" + lambda
//	  else 
//	    outputDir + "log_iter_" + MAX_ITER + "_th_" + suppportThres + "_lambda_" + lambda
//    
//    System.setProperty("spark.ui.port", "44717")
	System.setProperty("spark.kryoserializer.buffer.mb", "128")
	System.setProperty("spark.worker.timeout", "360")
	System.setProperty("spark.storage.blockManagerSlaveTimeoutMs", "1000000")
	System.setProperty("spark.akka.timeout", "60")
	System.setProperty("spark.default.parallelism", numCores.toString)
//	  System.setProperty("spark.mesos.coarse", "true")
	
    System.setProperty("spark.serializer", "spark.KryoSerializer")
    System.setProperty("spark.kryo.registrator", "util.MyRegistrator")
//    val storageLevel = spark.storage.StorageLevel.MEMORY_AND_DISK_SER
    val storageLevel = storage.StorageLevel.MEMORY_ONLY_SER_2
//    val storageLevel = spark.storage.StorageLevel.MEMORY_ONLY
    
    val bwLog = new BufferedWriter(new FileWriter(new File(logPath)))
    val sc = new SparkContext(mode, JOB_NAME, System.getenv("SPARK_HOME"), JARS)
    val rawTrainingData = sc.textFile(trainingDataDir, numCores)
    val numTrain = sc.accumulable(0)
    val numTest = sc.accumulable(0)
    val numTestPos = sc.accumulable(0)
    val numTestNeg = sc.accumulable(0)
    
    // to filter out infrequent features
    val stats = rawTrainingData.flatMap(line => line.split(" ")).filter(token => token.length>2)
                .map(token => (token.split(":")(0),1)).reduceByKey(_+_)
    val featureSet = stats.filter(pair => pair._2 >= suppportThres)
    						.map(pair => pair._1.toInt).persist(storageLevel).collect
    val P = featureSet.size + 1
    println("number of features used: " + (P-1))
    
    val featureMap = sc.broadcast(featureSet.zipWithIndex.toMap)
    val trainingData = rawTrainingData.map(line => {
      numTrain += 1
      Preprocess.parseLRLine(line, featureMap.value, false)}
    ).persist(storageLevel)
      
    val testingData = sc.textFile(testingDataDir, numCores).map(line => {
      numTest += 1
      val data = Preprocess.parseLRLine(line, featureMap.value, false)
      if (data._1) numTestPos += 1
      else numTestNeg += 1
      data}
    ).persist(storageLevel)
    
    // use all features
//    val trainingData = rawTrainingData.map(line => {
//      numTrain += 1
//      Preprocess.parseLRLine(line)}).persist(storageLevel)
//    val P = 20216831+1
//    
//    val testingData = sc.textFile(testingInputPath).map(line =>  {
//  	  numTest +=1
//      val data = Preprocess.parseLRLine(line)
//      if (data._1 > 0) numTestPos += 1
//      else numTestNeg += 1
//      data
//    }).persist(storageLevel)
    
    val testingDataPos = testingData.filter(_._1).persist(storageLevel)
    val testingDataNeg = testingData.filter(!_._1).persist(storageLevel)
    
    println("number of: features " + P + "; training data " + numTrain + "; testing data " + numTest)
    bwLog.write("number of: features " + P + "; training data " + numTrain 
        + "; testing data " + numTest + '\n')
    
    println("Time elapsed in parsing " + (System.currentTimeMillis()-startTime)*0.001 + "(s)")
    // Initialize w to a random value
    var w = Vector(Array.tabulate(P)(_=>((math.random-0.5)/math.sqrt(1)).toFloat))
    var delta_w = Vector(Array.tabulate(P)(_=>1.0f))
    var u = Vector(Array.tabulate(P)(_=>1.0f))
    var g_old = Vector(Array.tabulate(P)(_=>1.0f))
    var g = Vector(Array.tabulate(P)(_=>1.0f))
    var iter = 0
    var iterTime = System.currentTimeMillis()
    var auc = 0.5
    while (iter < MAX_ITER && auc < targetAUC) {
      iter += 1
      val w_broadcast = sc.broadcast(w)
      
//      use accumulator:
//      val llh = sc.accumulable(0.0f)
//      g = Vector(trainingData.flatMap(pair => { 
//        val features = pair._2
//        val response = 
//          if (pair._1) 1
//          else -1
//        val sigmoidywx = sigmoid(response * features.dot(w_broadcast.value))
//        llh += math.log(sigmoidywx).toFloat
//        val gradient = features*response*(1-sigmoidywx)
//        gradient.getIndices.zip(gradient.getValues)
//      }).reduceByKey(_+_).collect, P) - lambda*w
      
      //don't use accumulator:
      val g = Vector(trainingData.flatMap(pair => getGradient(pair, w_broadcast.value).getZippedPairs)
          .reduceByKey(_+_).collect, P) - lambda*w
      
      val delta_g = g - g_old
      g_old = g
      if (CG||LBFGS) {
        if (CG) {
          // conjugate gradient descent
          if (iter > 1) {
            val beta = g.dot(delta_g)/u.dot(delta_g)
            u = g - u*beta
          }
          else u = g
        }
        else if (LBFGS) {
          //limited-memory BFGS
          if (iter > 1) {
            val delta_wg = delta_w.dot(delta_g)
            val b = 1 + delta_g.dot(delta_g)/delta_wg
            val a_g = delta_w.dot(g)/delta_wg
            val a_w = delta_g.dot(g)/delta_wg - b*a_g
            u = -g + a_w*delta_w + a_g*delta_g
          }
          else
            u = g
        }
        val u_broadcast = sc.broadcast(u)
        val h = trainingData.map(pair => getHessian(pair, w_broadcast.value, u_broadcast.value))
        	.reduce(_+_)
        delta_w = g.dot(u)/(lambda*u.dot(u) + h)*u
      }
      else {
        //vanilla gradient descent
        delta_w = stepSize*g
      }
      
      // update the weight vector w
      w += delta_w
      //evaluation the algorithm using likelihood and prediction accuracy
      if (iter % interval == 0) {
        val w_broadcast = sc.broadcast(w)
        val llh = trainingData.map(data => getLLH(data, w_broadcast.value)).reduce(_+_)
        val tp = testingDataPos.map(data => 
          getBinPred(data._2, w_broadcast.value.elements, 50)).reduce(_+=_).elements
        val fp = testingDataNeg.map(data => 
          getBinPred(data._2, w_broadcast.value.elements, 50)).reduce(_+=_).elements
        auc = getAUC(tp, fp, numTestPos.value, numTestNeg.value)
        val time = (System.currentTimeMillis() - iterTime)*0.001
        println("Iter: " + iter + " time elapsed: " + time + " AUC: " + auc + " LLH: " + llh)
        bwLog.write("Iter: " + iter + " time elapsed: " + time + " AUC: " + auc + " LLH: " + llh + '\n')
        iterTime = System.currentTimeMillis()
      }
    }
    println("Final w " + w)
//    bwLog.write("Final w " + w + '\n')
    bwLog.write("Total time elapsed " + (System.currentTimeMillis()-startTime)*0.001 + "(s)")
    bwLog.close()
    println("Total time elapsed " + (System.currentTimeMillis()-startTime)*0.001 + "(s)")
    System.exit(0)
  }
}
package lr


import java.util._
import java.io._
import scala.util.Sorting._
import scala.collection.mutable.ArrayBuilder
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, HashPartitioner, storage}
import org.apache.spark.SparkContext._
import org.apache.spark.serializer.KryoRegistrator
import org.apache.commons.cli._
import utilities._
import Functions._
import preprocess.LR._

object LR extends Settings {
//  val trainingDataDir = "input/KDDCUP2010/kdda"
//  val testingDataDir = "input/KDDCUP2010/kdda.t"
  val trainingDataDir = "../datasets/UCI_Adult/a9a"
  val testingDataDir = "../datasets/UCI_Adult/a9a.t"
  val isBinary = true
  val isSeq = false
  val outputDir = "output/"
  val numCores = 1
  val numSlices = 10
  val mode = "local[" + numCores + "]"
  val maxOuterIter = 50
  val maxInnerIter = 20
  val lambda_init = 0.001f
  val gamma_init = 1.0f
  val rho_init = 1.0f
  val featureThre = 1
  val jaak = false
  val bohn = false
  val admm = true
  val emBayes = true
  val ard = true
  val exact = false
  val lbfgs = false
  val cg = false
  val l1 = true
  val tmpDir = "tmp"
  val numReducers = 2*numCores
  val jars = Seq("sparkproject.jar")
  val stopCriteria = 1e-6
  val interval = 1
  
  def toLocal(global: Array[Float], map: Array[Int], shared: Array[Boolean]) = {
    val localNumFeatures = map.length
    Array.tabulate(localNumFeatures)(i => if (shared(i)) global(map(i)) else 0f )
  }
  
  def toLocal(global: Array[Float], map: Array[Int]) = {
    val localNumFeatures = map.length
    Array.tabulate(localNumFeatures)(i => global(map(i)))
  }
  
  def main(args : Array[String]) {
    val currentTime = System.currentTimeMillis()
    
    val options = new Options()
    options.addOption(HELP, false, "print the help message")
    options.addOption(RUNNING_MODE_OPTION, true, "running mode option")
    options.addOption(TRAINING_OPTION, true, "training data path/directory")
    options.addOption(TESTING_OPTION, true, "testing data path/directory")
    options.addOption(OUTPUT_OPTION, true, "output path/directory")
    options.addOption(NUM_CORES_OPTION, true, "number of cores to use")
    options.addOption(NUM_REDUCERS_OPTION, true, "number of reducers to use")
    options.addOption(NUM_SLICES_OPTION, true, "number of slices of the data")
    options.addOption(OUTER_ITERATION_OPTION, true, "max # of outer iterations")
    options.addOption(INNER_ITERATION_OPTION, true, "max # of inner iterations")
    options.addOption(BOHN_BOUND_OPTION, false, "Bohning's bound option")
    options.addOption(JAAK_BOUND_OPTION, false, "Jaakkola's bound option")
    options.addOption(ADMM_OPTION, false, "ADMM option")
    options.addOption(LAMBDA_INIT_OPTION, true, "set lambda value")
    options.addOption(EMPIRICAL_BAYES_OPTION, false, "empirical Bayes option")
    options.addOption(GAMMA_INIT_OPTION, true, "set gamma value")
    options.addOption(JAR_OPTION, true, "the path to find jar file")
    options.addOption(TMP_DIR_OPTION, true, "the local dir for tmp files")
    options.addOption(L1_REGULARIZATION_OPTION, false, "use l1 regularization")
    options.addOption(FEATURE_THRESHOLD_OPTION, true, 
        "threshold on the used features' frequences")
    options.addOption(BINARY_FEATURES_OPTION, false, "binary features option")
    options.addOption(TARGET_AUC_OPTION, true, "targeted AUC option")
    options.addOption(EXACT_OPTION, false, "exact optimization option")
    options.addOption(CG_OPTION, false, "conjugate gradient descent option")
    options.addOption(LBFGS_OPTION, false, "L-BFGS option")
    options.addOption(CD_OPTION, false, "coordinate descent option")
    options.addOption(INTERVAL_OPTION, true, "output interval")
    options.addOption(ARD_OPTION, false, "auto relevence determination option")
    options.addOption(SEQUENCE_FILE_OPTION, false, "sequence file option")
    
    val parser = new GnuParser();
    val formatter = new HelpFormatter();
    val line = parser.parse(options, args);
    if (line.hasOption(HELP) || args.length == 0) {
      formatter.printHelp("Help", options);
      System.exit(0);
    }
    assert(line.hasOption(RUNNING_MODE_OPTION), "running mode not specified")
    assert(line.hasOption(TRAINING_OPTION), "training data path not specified")
    assert(line.hasOption(TESTING_OPTION), "testing data path not specified")
    assert(line.hasOption(JAR_OPTION), "jar file path not specified")
    assert(!line.hasOption(JAAK_BOUND_OPTION) || !line.hasOption(BOHN_BOUND_OPTION), 
        "cannpt specify two types of bound at the same time")
    assert(!line.hasOption(CG_OPTION) || !line.hasOption(LBFGS_OPTION), 
        "cannpt specify two types of optimization algorithm at the same time")
    val mode = line.getOptionValue(RUNNING_MODE_OPTION)
    val trainingDataDir = line.getOptionValue(TRAINING_OPTION)
    val testingDataDir = line.getOptionValue(TESTING_OPTION)
    val jars = Seq(line.getOptionValue(JAR_OPTION))
    val outputDir = 
      if (line.hasOption(OUTPUT_OPTION))
        if (!line.getOptionValue(OUTPUT_OPTION).endsWith(PATH_SEPERATOR))
          line.getOptionValue(OUTPUT_OPTION) + PATH_SEPERATOR
        else
          line.getOptionValue(OUTPUT_OPTION)
      else
        "output" + PATH_SEPERATOR
    val maxOuterIter =
      if (line.hasOption(OUTER_ITERATION_OPTION))
        line.getOptionValue(OUTER_ITERATION_OPTION).toInt
      else 10
    val maxInnerIter = 
      if (line.hasOption(INNER_ITERATION_OPTION))
        line.getOptionValue(INNER_ITERATION_OPTION).toInt
      else 5
    val numCores =
      if (line.hasOption(NUM_CORES_OPTION))
        line.getOptionValue(NUM_CORES_OPTION).toInt
      else 1
    val numReducers =
      if (line.hasOption(NUM_REDUCERS_OPTION))
        line.getOptionValue(NUM_REDUCERS_OPTION).toInt
      else numCores
    val numSlices =
      if (line.hasOption(NUM_SLICES_OPTION))
        line.getOptionValue(NUM_SLICES_OPTION).toInt
      else numCores
    val bohn = line.hasOption(BOHN_BOUND_OPTION)
    val admm = line.hasOption(ADMM_OPTION)
    val emBayes = line.hasOption(EMPIRICAL_BAYES_OPTION)
    val ard = line.hasOption(ARD_OPTION)
    val jaak = line.hasOption(JAAK_BOUND_OPTION)
    val gamma_init = 
      if (line.hasOption(GAMMA_INIT_OPTION)) 
        line.getOptionValue(GAMMA_INIT_OPTION).toFloat
      else 1f
    val lambda_init = 
      if (line.hasOption(LAMBDA_INIT_OPTION))
        line.getOptionValue(LAMBDA_INIT_OPTION).toFloat
      else 0.001f
    val tmpDir = if (line.hasOption(TMP_DIR_OPTION))
      line.getOptionValue(TMP_DIR_OPTION)
      else "tmp"
    val l1 = line.hasOption(L1_REGULARIZATION)
    val isBinary = line.hasOption(BINARY_FEATURES_OPTION)
    val isSeq = line.hasOption(SEQUENCE_FILE_OPTION)
    val featureThre = if (line.hasOption(FEATURE_THRESHOLD_OPTION))
      line.getOptionValue(FEATURE_THRESHOLD_OPTION).toInt
      else 0
    val targetAUC =
      if (line.hasOption(TARGET_AUC_OPTION))
        line.getOptionValue(TARGET_AUC_OPTION).toDouble
      else 0.9
    val exact = line.hasOption(EXACT_OPTION)
    val cg = line.hasOption(CG_OPTION)
    val lbfgs = line.hasOption(LBFGS_OPTION)
    val cd = line.hasOption(CD_OPTION)
    val interval = 
      if (line.hasOption(INTERVAL_OPTION))
        line.getOptionValue(INTERVAL_OPTION).toInt
      else 1
    
    System.setProperty("spark.local.dir", tmpDir)
    System.setProperty("spark.default.parallelism", numReducers.toString)
    System.setProperty("spark.storage.memoryFraction", "0.2")
    System.setProperty("spark.akka.frameSize", "64") //for large .collect() objects
//    System.setProperty("spark.serializer", 
//        "org.apache.spark.serializer.KryoSerializer")
//    System.setProperty("spark.kryo.registrator", "utilities.Registrator")
//    System.setProperty("spark.kryoserializer.buffer.mb", "16")
//    System.setProperty("spark.kryo.referenceTracking", "false")
    
    var jobName = "DIST_LR"
    if (exact) {
      if (cg) jobName += "_Exact_CG"
      else jobName += "_Exact_LBFGS"
    }
    else {
      if (admm) jobName += "_ADMM"
      if (cg) jobName += "_CG"
      else if (lbfgs) jobName += "_LBFGS"
      else {
        jobName += "_CD"
        if (bohn) jobName += "_Bohning"
        else if (jaak) jobName += "_Jaakkola" 
        else jobName += "_Taylor"
        if (l1) jobName += "_l1"
        if (emBayes) {
          jobName += "_emBayes"
          if (ard) jobName += "_ard"
        }
      }
    }
    jobName +=  "_i_" + maxOuterIter + "_b_" + numSlices + 
      "_gi_" + gamma_init + "_li_" + lambda_init + "_t_" + featureThre
    val logPath = outputDir + jobName + ".txt"
    
    val storageLevel = storage.StorageLevel.MEMORY_AND_DISK_SER
    val bwLog = new BufferedWriter(new FileWriter(new File(logPath)))
    val sc = new SparkContext(mode, jobName, System.getenv("SPARK_HOME"), jars)
    
    val featurePartitioner = new HashPartitioner(numReducers)
    val dataPartitioner = new HashPartitioner(numSlices)
    // to filter out infrequent features
    val featureSet = 
      if (featureThre > 0) {
        sc.textFile(trainingDataDir).flatMap(count(_, featureThre)).reduceByKey(_+_)
        .filter(_._2 >= featureThre).map(_._1).collect.sorted
      }
      else null
    
    val featureMapBC = 
      if (featureThre > 0) sc.broadcast(featureSet.zipWithIndex.toMap)
      else null
    
    val rawTrainingData = 
      if (isSeq) {
        if (isBinary) {
          sc.objectFile[Array[Int]](trainingDataDir).map(
            arr => {
              val bid = (math.random*numSlices).toInt
              val response = arr.tail == 1
              val feature = SparseVector(arr.dropRight(1))
              (bid, (response, feature))
            }
          )
        }
        else null
      }
      else {
        sc.textFile(trainingDataDir).map(line => {
        val bid = (math.random*numSlices).toInt
        if (featureThre > 0) (bid, parseLine(line, featureMapBC.value, isBinary))
        else (bid, parseLine(line, isBinary))
       })
      }
    val trainingData = rawTrainingData.groupByKey(dataPartitioner)
       .mapValues(seq => (seq.map(_._1).toArray, SparseMatrix(seq.map(_._2).toArray)))
       .persist(storageLevel)
    
    val testingData = 
      if (isSeq) {
        if (isBinary) {
          sc.objectFile[Array[Int]](testingDataDir)
            .map(arr => (arr.tail == 1, SparseVector(arr.dropRight(1))))
        }
        else null
      }
      else {
        sc.textFile(testingDataDir).map(line => {
        if (featureThre > 0) parseLine(line, featureMapBC.value, isBinary)
        else parseLine(line, isBinary)
       })
      }
    
    val splitsStats = trainingData.mapValues{
      case(responses, features) => (responses.length, features.numRows)
    }.collect
    
    //+1 because of the intercept
    val numFeatures = trainingData.map(_._2._2.rowMap.last).reduce(math.max(_,_)) + 1
    val isSparse = splitsStats.forall(pair => pair._2._2 < 0.5*numFeatures)
    val numTrain = splitsStats.map(_._2._1).reduce(_+_)
    val testingDataPos = testingData.filter(_._1).persist(storageLevel)
    val testingDataNeg = testingData.filter(!_._1).persist(storageLevel)
    val numTestPos = testingDataPos.count.toInt
    val numTestNeg = testingDataNeg.count.toInt
    val numTest = numTestPos + numTestNeg
    testingData.unpersist()
    val featureCount =
      if (!exact) {
        trainingData.map{
          case (id, (responses, features)) => SparseVector(features.rowMap)
        }.reduce(_+_).toArray(numFeatures)
      }
      else Array(0f)
    val localUpdate = featureCount.map(_>1)
    val localUpdateBC = sc.broadcast(localUpdate)
    val numLocalUpdate = localUpdate.count(p => p)
    val percentage = 100.0*numLocalUpdate/numFeatures
    println("number of: features " + numFeatures + "; training data " + numTrain + 
      "; testing data " + numTest)
    bwLog.write("number of: features " + numFeatures + "; training data " + numTrain 
      + "; testing data " + numTest + '\n')
    splitsStats.foreach(pair => println("partition " + pair._1 + 
      " has " + pair._2._1 + " samples and " + pair._2._2 + " features"))
    println("sparse update: " + isSparse)
    bwLog.write("sparse update: " + isSparse + "\n")
    println("Time elapsed after preprocessing " + 
        (System.currentTimeMillis()-currentTime)*0.001 + "(s)")
    bwLog.write("Time elapsed after preprocessing " + 
        (System.currentTimeMillis()-currentTime)*0.001 + "(s)\n")
    
    var iterTime = System.currentTimeMillis()
    var iter = 0
    val th = 0f
    val lambda = lambda_init
    val globalPara = new Array[Float](numFeatures)
    var globalParaBC = sc.broadcast(globalPara)
    val gradient = if (exact) new Array[Float](numFeatures) else null
    val gradient_old = if (exact) new Array[Float](numFeatures) else null
    val direction = if (exact) new Array[Float](numFeatures) else null
    val deltaPara = if (exact && lbfgs) new Array[Float](numFeatures) else null
    
    var localResults = trainingData.mapValues{
      case(responses, features) => {
        if (exact) {
          val para = toLocal(globalParaBC.value, features.rowMap)
          val obj = Functions.getGrad(responses, features, para, para)
          (Model(para), 1, obj, 0f)
        }
        else {
          val model = Model(features.numRows, gamma_init, admm)
          val lu = localUpdateBC.value
//          model.runCD(responses, features, lu, lambda, maxInnerIter, th)
          if (cg || lbfgs) {
            model.runCGQN(responses, features, maxInnerIter, th)
          }
          else model.runCD(responses, features, maxInnerIter, th)
        }
      }
    }
    localResults.persist(storageLevel)
    var rddId = localResults.id
    var rho = rho_init
    var localModels = localResults.mapValues(_._1)
    var localInfo = 
      if (interval == 1) 
        localResults.map{case (bid, (model, iter, obj, sd)) => (iter, obj)}.collect
       else null
    var auc = 0.0
    var old_auc = -10.0
    while (iter < maxOuterIter && math.abs(auc-old_auc) > stopCriteria) {
      if (exact) {
        if (isSparse) {
          trainingData.join(localModels).map{
            case(bid, (data, model)) => model.getParaStatsSpa(data._2.rowMap)
          }.reduce(_+_).toArray(gradient)
        }
        else {
          trainingData.join(localModels).map{
            case(bid, (data, model)) => model.getParaStats(data._2.rowMap, numFeatures)
          }.reduce(_+=_).toArray(gradient)
        }
        var p = 1 //no shrinkage for the intercept
        while (p < numFeatures) {
          gradient(p) -= lambda*globalPara(p)
          p += 1
        }
        if (cg) {
          // conjugate gradient descent
          if (iter > 1) Functions.getCGDirection(gradient, gradient_old, direction)
          else Functions.copy(gradient, direction)
        }
        else {
          //limited-memory BFGS
          if (iter > 1) {
            Functions.getLBFGSDirection(deltaPara, gradient, gradient_old, direction)
          }
          else Functions.copy(gradient, direction)
        }
        val directionBC = sc.broadcast(direction)
        val h = trainingData.map{
          case(id, (responses, features)) => {
            val para = toLocal(globalParaBC.value, features.rowMap)
            val direction = toLocal(directionBC.value, features.rowMap)
            Functions.getHessian(features, para, direction)
          }
        }.sum.toFloat
        p = 1
        var gu = gradient(0)*direction(0)
        var hessian = 0f
        while (p < numFeatures) {
          hessian += direction(p)*direction(p)
          gu += gradient(p)*direction(p)
          p += 1
        }
        hessian *= lambda
        hessian += h
        p = 0
        while (p < numFeatures) {
          gradient_old(p) = gradient(p)
          val delta = gu/h*direction(p)
          if (!cg) deltaPara(p) = delta
          globalPara(p) += delta
          p += 1
        }
      }
      else {
        if (isSparse) {
          if (emBayes) {
            trainingData.join(localModels).map{
              case(bid, (data, model)) => 
                model.getGammaSpa(data._2.rowMap)
            }.reduce(_+_).toArray(featureCount)
          }
          trainingData.join(localModels).map{
            case(bid, (data, model)) => 
              model.getParaStatsSpa(data._2.rowMap, admm, emBayes)
          }.reduce(_+_).toArray(globalPara)
        }
        else {
          if (emBayes) {
            trainingData.join(localModels).map{
              case(bid, (data, model)) => 
                model.getGamma(data._2.rowMap, numFeatures)
            }.reduce(_+=_).toArray(featureCount)
          }
          trainingData.join(localModels).map{
            case(bid, (data, model)) => 
              model.getParaStats(data._2.rowMap, numFeatures, admm, emBayes)
          }.reduce(_+=_).toArray(globalPara)
        }
        var p = 0
        while (p < numFeatures) {
          globalPara(p) = 
            if (l1) Functions.l1Prox(globalPara(p), featureCount(p), lambda)
            else Functions.l2Prox(globalPara(p), featureCount(p), lambda)
          p += 1
        }
      }
      globalParaBC = sc.broadcast(globalPara)
      if ((iter+1) % interval == 0) {
        //calculate the AUC
        val tp = testingDataPos.map(data => 
          getBinPred(data._2, globalParaBC.value, 50)).reduce(_+=_).elements
        val fp = testingDataNeg.map(data => 
          getBinPred(data._2, globalParaBC.value, 50)).reduce(_+=_).elements
        old_auc = auc
        auc = getAUC(tp, fp, numTestPos, numTestNeg)
        val (innerIterSum, obj_gr) = localInfo.reduce((pair1, pair2) => 
          (pair1._1+pair2._1, pair1._2+pair2._2))
        val obj = obj_gr - globalPara.map(p => p*p).sum*lambda/2
        val l1norm = globalPara.map(math.abs(_)).sum/numFeatures
        if (emBayes) {
          val gamma = localModels.map{
            case(bid, model) => model.getGammaMeanVariance(ard)
          }.collect
          print("gamma:\t"); bwLog.write("gamma:\t")
          gamma.foreach{case(mean, variance) => {
            val msg = "(" + mean + ", +-" + math.sqrt(variance) + ") "
            print(msg) 
            bwLog.write(msg)
          }}
          println("\nrho: " + rho)
          bwLog.write("\nrho: " + rho + "\n")
        }
        val time = (System.currentTimeMillis() - iterTime)*0.001
        println("Average number of inner iterations: " + innerIterSum/numSlices)
        println("Iter: " + iter + " time elapsed: " + time + " AUC: " + auc +
          " obj: " + obj)
        println("ave l1 norm: " + l1norm)
        bwLog.write("Average number of inner iterations: " + 
            innerIterSum/numSlices + "\n")
        bwLog.write("Iter: " + iter + " time elapsed: " + time + " AUC: " + auc +
          " obj: " + obj + '\n')
        bwLog.write("ave l1 norm: " + l1norm + '\n')
        iterTime = System.currentTimeMillis()
      }
      
      iter += 1
      if (iter < maxOuterIter) {
        localResults =
          if (exact) {
            trainingData.mapValues{
              case((responses, features)) => {
                val para = toLocal(globalParaBC.value, features.rowMap)
                val obj = Functions.getGrad(responses, features, para, para)
                (Model(para), 1, obj, 0f)
              }
            }
          }
          else {
            trainingData.join(localModels).mapValues{
              case((responses, features), model) => {
                val featureMap = features.rowMap
                val lu = localUpdateBC.value
                val gp = globalParaBC.value
                val prior = toLocal(gp, featureMap, lu)
                val th = 0.001f
                val maxIter = maxInnerIter
                if (cg || lbfgs) {
                  model.runCGQN(responses, features, maxIter, th, cg, rho, admm, prior)
                }
                else {
                  model.runCD(responses, features, maxIter, th, rho, 
                    admm, l1, bohn, jaak, emBayes, ard, prior)
                }
              }
            }
          }
        if (!exact || (iter+1) % interval == 0) {
          localResults.persist(storageLevel)
          localInfo = localResults.map{
            case (bid, (model, iter, obj, squaredDiff)) => (iter, obj)
          }.collect
        }
        localModels = localResults.mapValues(_._1)
        if (emBayes) {
          val ssd = localResults.map(_._2._4).reduce(_+_)
          val num = trainingData.map(_._2._2.numRows).reduce(_+_)
          rho = num / (ssd + 0.05f)
        }
        if (!exact || iter % interval == 0) {
          println("Let's remove RDD: " + rddId)
          sc.getPersistentRDDs(rddId).unpersist(false)
        }
        if (!exact || (iter+1) % interval == 0) rddId = localResults.id
      }
    }
    if (numFeatures < 200) {
      val str = globalPara.mkString(" ")
      println("global: " + str)
      bwLog.write("global: " + str + '\n')
    }
    bwLog.write("Total time elapsed " 
        + (System.currentTimeMillis()-currentTime)*0.001 + "(s)\n")
    bwLog.close()
    println("Total time elapsed " 
        + (System.currentTimeMillis()-currentTime)*0.001 + "(s)")
    System.exit(0)
  }
}
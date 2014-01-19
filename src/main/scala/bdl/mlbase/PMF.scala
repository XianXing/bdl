/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package mlbase

import java.io._

import scala.collection.mutable.{ArrayBuffer, BitSet}
import scala.util.Random
import scala.util.Sorting

import org.apache.spark.{HashPartitioner, Partitioner, SparkContext}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.rdd._
import org.apache.spark.serializer.KryoRegistrator
import org.apache.spark.SparkContext._

import org.apache.hadoop.io.NullWritable

import com.esotericsoftware.kryo.Kryo

import preprocess.MF._
import utilities.Record
/**
 * Out-link information for a user or product block. This includes the original user/product IDs
 * of the elements within this block, and the list of destination blocks that each user or
 * product will need to send its feature vector to.
 */
private[mlbase] case class OutLinkBlock(elementIds: Array[Int], shouldSend: Array[BitSet])


/**
 * In-link information for a user (or product) block. This includes the original user/product IDs
 * of the elements within this block, as well as an array of indices and ratings that specify
 * which user in the block will be rated by which products from each product block (or vice-versa).
 * Specifically, if this InLinkBlock is for users, ratingsForBlock(b)(i) will contain two arrays,
 * indices and ratings, for the i'th product that will be sent to us by product block b (call this
 * P). These arrays represent the users that product P had ratings for (by their index in this
 * block), as well as the corresponding rating for each one. We can thus use this information when
 * we get product block b's message to update the corresponding users.
 */
private[mlbase] case class InLinkBlock(
  elementIds: Array[Int], ratingsForBlock: Array[Array[(Array[Int], Array[Float])]])


/**
 * A more compact class to represent a rating than Tuple3[Int, Int, Float].
 */
case class Rating(val user: Int, val product: Int, val rating: Float) {
  def this (tuple: (Int, Int, Float)) = this(tuple._1, tuple._2, tuple._3)
  def this (record: Record) = this(record.rowIdx, record.colIdx, record.value)
}

/**
 * Alternating Least Squares matrix factorization.
 *
 * This is a blocked implementation of the PMF factorization algorithm that groups the two sets
 * of factors (referred to as "users" and "products") into blocks and reduces communication by only
 * sending one copy of each user vector to each product block on each iteration, and only for the
 * product blocks that need that user's feature vector. This is achieved by precomputing some
 * information about the ratings matrix to determine the "out-links" of each user (which blocks of
 * products it will contribute to) and "in-link" information for each product (which of the feature
 * vectors it receives from each user block it will depend on). This allows us to send only an
 * array of feature vectors between each user block and product block, and have the product block
 * find the users' ratings and update the products based on these messages.
 */


class PMF private (var numBlocks: Int, var rank: Int, var iterations: Int, vb: Boolean,
    var gamma_r_init: Float, var gamma_c_init: Float) extends Serializable {
  
  val STORAGE_LEVEL = StorageLevel.MEMORY_AND_DISK_SER
  
  /**
   * Set the number of blocks to parallelize the computation into; pass -1 for an auto-configured
   * number of blocks. Default: -1.
   */
  def setBlocks(numBlocks: Int): PMF = {
    this.numBlocks = numBlocks
    this
  }

  /** Set the rank of the feature matrices computed (number of features). Default: 10. */
  def setRank(rank: Int): PMF = {
    this.rank = rank
    this
  }

  /** Set the number of iterations to run. Default: 10. */
  def setIterations(iterations: Int): PMF = {
    this.iterations = iterations
    this
  }

  /** Set the row regularization parameter, gamma_r_init. Default: 0.1. */
  def setGamma_r_init(gamma_r_init: Float): PMF = {
    this.gamma_r_init = gamma_r_init
    this
  }
  
  /** Set the col regularization parameter, gamma_c_init. Default: 0.1. */
  def setGamma_c_init(gamma_c_init: Float): PMF = {
    this.gamma_c_init = gamma_c_init
    this
  }

  /**
   * Run PMF with the configured parameters on an input RDD of (user, product, rating) triples.
   * Returns a MatrixFactorizationModel with feature vectors for each user and product.
   */
  def run(sc: org.apache.spark.SparkContext,
      trainingData: RDD[Rating], testingData: RDD[Rating], scale: Float, 
      logPath: String, interval: Int) = {
    val numBlocks = 
      if (this.numBlocks == -1) {
      math.max(trainingData.context.defaultParallelism, 
          trainingData.partitions.size / 2)
      } else {
        this.numBlocks
      }
    val bwLog = new BufferedWriter(new FileWriter(new File(logPath)))
    val partitioner = new HashPartitioner(numBlocks)
    
    val (numUsers, numProducts, numTrainingData) = trainingData.map(
        r => (r.user, r.product, 1)).reduce(
            (t1, t2) => (math.max(t1._1, t2._1), math.max(t1._2, t2._2), t1._3+t2._3))
    val numTestingData = testingData.count
    
    val blockedTestingData = testingData.map(rating => 
      ((rating.user%numBlocks, rating.product%numBlocks), rating))
      .groupByKey(partitioner).mapValues(_.sortWith(
          (r1, r2) => (r1.user < r2.user) 
          || ((r1.user==r2.user)&&(r1.product < r2.product)))
      ).persist(STORAGE_LEVEL)
    
    println("Number of rows: " + numUsers)
    println("Number of columns: " + numProducts)
    println("Number of training samples: " + numTrainingData)
    println("Number of testing samples: " + numTestingData)    
    bwLog.write("Number of rows: " + numUsers + "\n")
    bwLog.write("Number of columns: " + numProducts + "\n")
    bwLog.write("Number of training samples: " + numTrainingData + "\n")
    bwLog.write("Number of testing samples: " + numTestingData + "\n")
    
    val trainingDataByUserBlock = 
      trainingData.map{ rating => (rating.user % numBlocks, rating) }
    val trainingDataByProductBlock = trainingData.map{ rating =>
      (rating.product % numBlocks, Rating(rating.product, rating.user, rating.rating))
    }
    
    val (userInLinks, userOutLinks) = 
      makeLinkRDDs(numBlocks, trainingDataByUserBlock, partitioner)
    val (productInLinks, productOutLinks) = 
      makeLinkRDDs(numBlocks, trainingDataByProductBlock, partitioner)
      
    // Initialize user and product factors randomly, 
    //  but use a deterministic seed for each partition
    // so that fault recovery works
    val seedGen = new Random()
    val seed1 = seedGen.nextInt()
    val seed2 = seedGen.nextInt()
    // Hash an integer to propagate random bits at all positions, similar to java.util.HashTable
    def hash(x: Int): Int = {
      val r = x ^ (x >>> 20) ^ (x >>> 12)
      r ^ (r >>> 7) ^ (r >>> 4)
    }
//    var users = userOutLinks.mapPartitionsWithIndex ( (index, itr) => {
//      val rand = new Random(hash(seed1 ^ index))
//      itr.map { case (x, y) =>
//        (x, y.elementIds.map(_ => randomFactor(rank, rand)))
//      }}, true
//    )
    var usersPara = userOutLinks.mapPartitionsWithIndex( (index, itr) => {
      val rand = new Random(hash(seed1 ^ index))
      itr.map { 
        case (x, y) =>
          (x, (y.elementIds.map(_ => randomFactor(rank, rand)),
            y.elementIds.map(_=>(Array.fill(rank)(Float.MaxValue)))))
      }
    }, true).persist(STORAGE_LEVEL)
    usersPara.count
    var userRDDID = usersPara.id
    var productsPara = productOutLinks.mapValues(outLink => (
      outLink.elementIds.map(_=>(Array.ofDim[Float](rank))), 
      outLink.elementIds.map(_=>(Array.ofDim[Float](rank))))).persist(STORAGE_LEVEL)
    productsPara.count
    var productRDDID = productsPara.id
    var iter = 0
    while (iter < iterations) {
      iter += 1
      // perform PMF update
      val iterTime = System.currentTimeMillis()
      productsPara = updateCCD(usersPara, productsPara, userOutLinks, productInLinks, 
         partitioner, rank, gamma_c_init, vb).persist(STORAGE_LEVEL)
      productsPara.count
      sc.getPersistentRDDs(productRDDID).unpersist(false)
      productRDDID = productsPara.id
      usersPara = updateCCD(productsPara, usersPara, productOutLinks, userInLinks, 
         partitioner, rank, gamma_r_init, vb).persist(STORAGE_LEVEL)
      usersPara.count
      sc.getPersistentRDDs(userRDDID).unpersist(false)
      userRDDID = usersPara.id
      if (iter % interval == 0 || iter == iterations) {
        
        val trainingRMSE =
          if (numUsers > numProducts)
            scale*math.sqrt(calculateRMSE(productsPara, usersPara, productOutLinks, 
              userInLinks, partitioner).map(pair => pair._2).reduce(_+_)
              /numTrainingData)
          else
            scale*math.sqrt(calculateRMSE(usersPara, productsPara, userOutLinks, 
              productInLinks, partitioner).map(pair => pair._2).reduce(_+_)
              /numTrainingData)
        
        val testingRMSE = scale*math.sqrt(usersPara.join(userOutLinks)
          .cartesian(productsPara.join(productOutLinks)).map{
            case(users, products) => 
              ((users._1, products._1), ((users._2._1._1, users._2._2.elementIds),
              (products._2._1._1, products._2._2.elementIds)))
          }.join(blockedTestingData).map(pair => calculateRMSE(pair._2))
          .reduce(_+_)/numTestingData)
        
        val time = (System.currentTimeMillis() - iterTime)*0.001
        println("\nIter: " + iter + " time elapsed: " + time)
        println("Training RMSE: "+ trainingRMSE + "\tTesting RMSE: " + testingRMSE)
        bwLog.write("\nIter: " + iter + " time elapsed: " + time + "\n")
        bwLog.write("Training RMSE: "+ trainingRMSE + "\ttesting RMSE: "
            + testingRMSE + "\n")
      }
    }
    
    // Flatten and cache the two final RDDs to un-block them
    val usersMean = usersPara.join(userOutLinks).flatMap { 
      case (b, (usersPara, outLinkBlock)) =>
      for (i <- 0 until usersPara._1.length) 
        yield (outLinkBlock.elementIds(i), usersPara._1(i))
    }
    val productsMean = productsPara.join(productOutLinks).flatMap { 
      case (b, (productsPara, outLinkBlock)) =>
      for (i <- 0 until productsPara._1.length) 
        yield (outLinkBlock.elementIds(i), productsPara._1(i))
    }
    
    usersMean.persist(STORAGE_LEVEL)
    productsMean.persist(STORAGE_LEVEL)
    bwLog.close()
  }
  
  /**
   * Make RDDs of InLinkBlocks and OutLinkBlocks given an RDD of (blockId, (u, p, r)) values for
   * the users (or (blockId, (p, u, r)) for the products). We create these simultaneously to avoid
   * having to shuffle the (blockId, (u, p, r)) RDD twice, or to cache it.
   */
  private def makeLinkRDDs(numBlocks: Int, ratings: RDD[(Int, Rating)],
      partitioner: Partitioner)
    : (RDD[(Int, InLinkBlock)], RDD[(Int, OutLinkBlock)]) = {
    val links = ratings.groupByKey(partitioner).mapValues{ grouped => {
      val ratingsArray = grouped.toArray
      val inLinkBlock = makeInLinkBlock(numBlocks, ratingsArray)
      val outLinkBlock = makeOutLinkBlock(numBlocks, ratingsArray)
      (inLinkBlock, outLinkBlock)
    }}
    links.persist(STORAGE_LEVEL)
    (links.mapValues(_._1), links.mapValues(_._2))
  }
  
  /**
   * Make the out-links table for a block of the users (or products) dataset given the list of
   * (user, product, rating) values for the users in that block (or the opposite for products).
   */
  private def makeOutLinkBlock(numBlocks: Int, ratings: Array[Rating]): OutLinkBlock = {
    val userIds = ratings.map(_.user).distinct.sorted
    val numUsers = userIds.length
    val userIdToPos = userIds.zipWithIndex.toMap
    val shouldSend = Array.fill(numUsers)(new BitSet(numBlocks))
    for (r <- ratings) {
      shouldSend(userIdToPos(r.user))(r.product % numBlocks) = true
    }
    OutLinkBlock(userIds, shouldSend)
  }

  /**
   * Make the in-links table for a block of the users (or products) dataset given a list of
   * (user, product, rating) values for the users in that block (or the opposite for products).
   */
  private def makeInLinkBlock(numBlocks: Int, ratings: Array[Rating]): InLinkBlock = {
    val userIds = ratings.map(_.user).distinct.sorted
    val numUsers = userIds.length
    val userIdToPos = userIds.zipWithIndex.toMap
    // Split out our ratings by product block
    val blockRatings = Array.fill(numBlocks)(new ArrayBuffer[Rating])
    for (r <- ratings) {
      blockRatings(r.product % numBlocks) += r
    }
    val ratingsForBlock = new Array[Array[(Array[Int], Array[Float])]](numBlocks)
    for (productBlock <- 0 until numBlocks) {
      // Create an array of (product, Seq(Rating)) ratings
      val groupedRatings = blockRatings(productBlock).groupBy(_.product).toArray
      // Sort them by product ID
      val ordering = new Ordering[(Int, ArrayBuffer[Rating])] {
        def compare(a: (Int, ArrayBuffer[Rating]), b: (Int, ArrayBuffer[Rating]))
          : Int = a._2(0).product - b._2(0).product
      }
      Sorting.quickSort(groupedRatings)(ordering)
      // Translate the user IDs to indices based on userIdToPos
      ratingsForBlock(productBlock) = groupedRatings.map { case (p, rs) =>
        (rs.view.map(r => userIdToPos(r.user)).toArray, rs.view.map(_.rating).toArray)
      }
    }
    InLinkBlock(userIds, ratingsForBlock)
  }
  
  /**
   * Make a random factor vector with the given random.
   */
  private def randomFactor(rank: Int, rand: Random): Array[Float] = {
    Array.fill(rank)(0.1f*(rand.nextFloat-0.5f))
  }
  
  private def updateCCD(
      productsPara: RDD[(Int, (Array[Array[Float]], Array[Array[Float]]))],
      usersPara: RDD[(Int, (Array[Array[Float]], Array[Array[Float]]))],
      productOutLinks: RDD[(Int, OutLinkBlock)],
      userInLinks: RDD[(Int, InLinkBlock)],
      partitioner: Partitioner,
      rank: Int,
      lambda: Float,
      vb: Boolean)
    : RDD[(Int, (Array[Array[Float]], Array[Array[Float]]))] = {
    val numBlocks = productsPara.partitions.size
    
    productOutLinks.join(productsPara).flatMap { case (bid, (outLinkBlock, paras)) =>
        val toSendMean = Array.fill(numBlocks)(new ArrayBuffer[Array[Float]])
        val toSendPrecision = Array.fill(numBlocks)(new ArrayBuffer[Array[Float]])
        for (p <- 0 until outLinkBlock.elementIds.length; userBlock <- 0 until numBlocks) {
          if (outLinkBlock.shouldSend(p)(userBlock)) {
            toSendMean(userBlock) += paras._1(p)
            toSendPrecision(userBlock) += paras._2(p)
          }
        }
        toSendMean.zip(toSendPrecision).zipWithIndex.map{ 
          case (buf, idx) => (idx, (bid, buf._1.toArray, buf._2.toArray)) }
    }.groupByKey(partitioner)
     .join(usersPara.join(userInLinks))
     .mapValues{ case (messages, (usersPara, inLinkBlock))
        => ccd(messages, usersPara, inLinkBlock, rank, lambda, vb) }
  }
  
  def ccd(messages: Seq[(Int, Array[Array[Float]], Array[Array[Float]])], 
      usersPara: (Array[Array[Float]], Array[Array[Float]]),
      inLinkBlock: InLinkBlock, rank: Int, lambda: Float, vb: Boolean)
    : (Array[Array[Float]], Array[Array[Float]]) = {
    // Sort the incoming block factor messages by block ID and make them an array
    val sorted = messages.sortBy(_._1)
    val blockMeans = sorted.map(_._2).toArray // Array[Array[Float]]
    val blockPrecisions = sorted.map(_._3).toArray
    val numBlocks = blockMeans.length
    val numUsers = inLinkBlock.elementIds.length
    val usersMean = usersPara._1
    val usersPrecision = usersPara._2
    val rank = usersMean(0).length
    // calculate the residuals
    for (productBlock <- 0 until numBlocks) {
      if (inLinkBlock.ratingsForBlock.length <= productBlock)
        println
      for (p <- 0 until blockMeans(productBlock).length) {
        val product = blockMeans(productBlock)(p)
        if (inLinkBlock.ratingsForBlock(productBlock).length <= p)
          println("length:" + inLinkBlock.ratingsForBlock(productBlock).length + " p: " + p)
        val (us, rs) = inLinkBlock.ratingsForBlock(productBlock)(p)
        for (i <- 0 until us.length) {
          val user = usersMean(us(i))
          var k = 0; var pred = 0.0f; while (k<rank) {pred += product(k)*user(k); k+=1}
          rs(i) -= pred
        }
      }
    }
    
    //coordinate descent
    val numerator = Array.fill(numUsers)(0.0f)
    val denominator = Array.fill(numUsers)(lambda)
    for (k <- 0 until rank) {
      for (productBlock <- 0 until numBlocks) {
        for (p <- 0 until blockMeans(productBlock).length) {
          val productMean = blockMeans(productBlock)(p)
          val productPrecision = blockPrecisions(productBlock)(p)
          val (us, res) = inLinkBlock.ratingsForBlock(productBlock)(p)
          var i=0
          while (i < us.length) {
            val userMean = usersMean(us(i))
            if (k>0) res(i) -= userMean(k-1)*productMean(k-1)
            res(i) += userMean(k)*productMean(k)
            numerator(us(i)) += res(i)*productMean(k)
            denominator(us(i)) += productMean(k)*productMean(k)
            if (vb)  denominator(us(i)) += 1/productPrecision(k)
            i += 1
          }
        }
      }
      for (u <- 0 until numUsers) {
        usersMean(u)(k) = numerator(u)/denominator(u)
        usersPrecision(u)(k) = denominator(u)
        numerator(u) = 0
        denominator(u) = lambda
      }
    }
    (usersMean, usersPrecision)
  }
   private def calculateRMSE(
      products: RDD[(Int, (Array[Array[Float]], Array[Array[Float]]))],
      users: RDD[(Int, (Array[Array[Float]], Array[Array[Float]]))],
      productOutLinks: RDD[(Int, OutLinkBlock)],
      userInLinks: RDD[(Int, InLinkBlock)],
      partitioner: Partitioner)
    : RDD[(Int, Float)] = {
    val numBlocks = products.partitions.size
    
    productOutLinks.join(products).flatMap { case (bid, (outLinkBlock, paras)) =>
        val toSend = Array.fill(numBlocks)(new ArrayBuffer[Array[Float]])
        for (p <- 0 until outLinkBlock.elementIds.length; 
          userBlock <- 0 until numBlocks) {
          if (outLinkBlock.shouldSend(p)(userBlock)) {
            toSend(userBlock) += paras._1(p)
          }
        }
        toSend.zipWithIndex.map{ case (buf, idx) => (idx, (bid, buf.toArray)) }
    }.groupByKey(partitioner)
     .join(users.join(userInLinks))
     .mapValues{ case (messages, (users, inLinkBlock))
        => getBlockSE(messages, users._1, inLinkBlock) }
  }
  
  def getBlockSE(
      messages: Seq[(Int, Array[Array[Float]])], users: Array[Array[Float]], 
      inLinkBlock: InLinkBlock) : Float = {
    // Sort the incoming block factor messages by block ID and make them an array
    val blockFactors = messages.sortBy(_._1).map(_._2).toArray // Array[Array[Float]]
    val numBlocks = blockFactors.length
    val numUsers = inLinkBlock.elementIds.length
    val rank = users(0).length
    var se = 0.0f
    // calculate the residuals
    for (productBlock <- 0 until numBlocks) {
      for (p <- 0 until blockFactors(productBlock).length) {
        val product = blockFactors(productBlock)(p)
        val (us, rs) = inLinkBlock.ratingsForBlock(productBlock)(p)
        for (i <- 0 until us.length) {
          val user = users(us(i))
          val res = rs(i) - 
            product.view.zip(user).map{ case (a,b) => a*b }.reduceLeft(_+_)
          se += res*res
        }
      }
    }
    se
  }
  
  def calculateRMSE(
      pair: (((Array[Array[Float]], Array[Int]), (Array[Array[Float]], Array[Int])), 
          Seq[mlbase.Rating]))= {
    val userFeatures = pair._1._1._1
    val userIndices = pair._1._1._2
    val productFeatures = pair._1._2._1
    val productIndices = pair._1._2._2
    val data = pair._2
    var i = 0; var ui = 0; var pi = 0; var res = 0.0; var pred = 0.0
    val nnz = data.length
    while (i < nnz) {
      val u = data(i).user; val p = data(i).product; val r = data(i).rating
      while (userIndices(ui) < u) {ui += 1; pi = 0}
      while (productIndices(pi) < p) pi += 1
      pred = userFeatures(ui).zip(productFeatures(pi)).map(pair => pair._1*pair._2).reduce(_+_)
      res += (r-pred)*(r-pred)
      i += 1
    }
    res
  }
}


/**
 * Top-level methods for calling Alternating Least Squares (PMF) matrix factorizaton.
 */
object PMF {
  /**
   * Train a matrix factorization model given an RDD of ratings given by users to some products,
   * in the form of (userID, productID, rating) pairs. We approximate the ratings matrix as the
   * product of two lower-rank matrices of a given rank (number of features). To solve for these
   * features, we run a given number of iterations of PMF. This is done using a level of
   * parallelism given by `blocks`.
   *
   * @param ratings    RDD of (userID, productID, rating) pairs
   * @param rank       number of features to use
   * @param iterations number of iterations of PMF (recommended: 10-20)
   * @param lambda     regularization factor (recommended: 0.01)
   * @param blocks     level of parallelism to split computation into
   */
  def train(
      sc: org.apache.spark.SparkContext,
      trainingData: RDD[Rating],
      testingData: RDD[Rating],
      rank: Int,
      iterations: Int,
      vb: Boolean,
      gamma_r_init: Float,
      gamma_c_init: Float,
      scale: Float,
      blocks: Int,
      logPath: String = "log",
      interval: Int = 1) =
  {
    new PMF(blocks, rank, iterations, vb, gamma_r_init, gamma_c_init)
      .run(sc, trainingData, testingData, scale, logPath, interval)
  }

  private class PMFRegister extends KryoRegistrator {
    override def registerClasses(kryo: Kryo) {
      kryo.register(classOf[Rating])
    }
  }

  def main(args: Array[String]) {
    
    if (args.length != 15) {
      println("Usage: PMF <master> <jar_file> <training_files> <testing_files> " +
                      "<output_dir> <rank> <iterations> <vb> <gamma_r_init>" +
                      " <gamma_c_init> <blocks> <mean> <scale> <tmp_dir> <interval>")
      System.exit(1)
    }
    args.foreach(println(_))
    val (master, jar, trainingPath, testingPath, outputDir, rank, iters, vb,
        gamma_r_init, gamma_c_init, blocks, mean, scale, tmpDir, interval) =
      (args(0), Seq(args(1)), args(2), args(3), args(4), args(5).toInt, args(6).toInt,
          args(7).toBoolean, args(8).toFloat, args(9).toFloat, args(10).toInt,
          args(11).toFloat, args(12).toFloat, args(13), args(14).toInt)
    
//    val master = "local[2]"
//    val trainingPath = "input/EachMovie-GL/1.train"
//    val testingPath = "input/EachMovie-GL/1.validate"
//    val rank = 20
//    val iters = 20
//    val outputDir = "output"
//    val blocks = 4
//    val gamma_c_init = 10
//    val gamma_r_init = 10
//    val vb = true
//    val mean = 4.037181f
//    val scale = 1
//    val interval = 1
//    val jar = Seq("/Users/xianxingzhang/Documents/workspace/spark/" +
//      "examples/target/scala-2.9.3/spark-examples_2.9.3-0.8.0-SNAPSHOT.jar")
//    val tmpDir = "tmp"
    
    val job = if (vb) "SparkVBMF" else "SparkMF"
    val logPath = outputDir + "/" + job + "_rank_" + rank + "_iter_" + 
      iters + "_blocks_" + blocks + "_gamma_r_" + gamma_r_init +
      "_gamma_c_" + gamma_c_init + "_scale_" + scale + ".txt"
    System.setProperty("spark.serializer", 
        "org.apache.spark.serializer.KryoSerializer")
    System.setProperty("spark.kryo.registrator", classOf[PMFRegister].getName)
    System.setProperty("spark.kryo.referenceTracking", "false")
//    System.setProperty("spark.kryoserializer.buffer.mb", "128")
    System.setProperty("spark.local.dir", tmpDir)
//    System.setProperty("spark.ui.port", "44717")
    System.setProperty("spark.locality.wait", "10000")
    val STORAGE_LEVEL = StorageLevel.MEMORY_AND_DISK_SER
    val sc = new SparkContext(master, job, System.getenv("SPARK_HOME"), jar)
    val trainingData = 
      if (trainingPath.toLowerCase.contains("eachmovie"))
        sc.textFile(trainingPath).map(line => 
          new Rating(parseLine(line, " ", mean, scale)))
      else if (trainingPath.toLowerCase.contains("syn"))
        sc.sequenceFile[NullWritable, Record](trainingPath)
          .map(pair => 
            new Rating(pair._2.rowIdx, pair._2.colIdx, (pair._2.value-mean)/scale))
      else sc.textFile(trainingPath).flatMap(line => parseLine(line, mean, scale))
        .map(record => new Rating(record))
    trainingData.persist(STORAGE_LEVEL)
    val testingData = 
      if (testingPath.toLowerCase.contains("eachmovie")) 
        sc.textFile(testingPath).map(line => 
          new Rating(parseLine(line, " ", mean, scale)))
      else if (testingPath.toLowerCase.contains("syn"))
        sc.sequenceFile[NullWritable, Record](testingPath)
          .map(pair => 
            new Rating(pair._2.rowIdx, pair._2.colIdx, (pair._2.value-mean)/scale))
      else 
        sc.textFile(testingPath).flatMap(line => parseLine(line, mean, scale))
          .map(record => new Rating(record))
    testingData.persist(STORAGE_LEVEL)
    
    val time = System.currentTimeMillis()
    val model = 
      PMF.train(sc, trainingData, testingData, rank, iters, vb, gamma_r_init,
          gamma_c_init, scale, blocks, logPath, interval)
    println("Total time elapsed: " + (System.currentTimeMillis() - time)*0.001)
    System.exit(0)
  }
}
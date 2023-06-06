package com.example.clustering

/**
 * @author ${user.name}
 */

import org.apache.spark.sql.SparkSession
import scala.io.Source
import org.apache.spark.rdd.RDD
import scala.util.control._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD // RDD 형식 파일
import org.apache.spark._
import org.apache.spark.sql.{Row,SparkSession,SQLContext,DataFrame}
import org.apache.spark.sql.functions.lit
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.sql.types.{DataTypes,StructType, StructField, StringType, IntegerType, FloatType, MapType, DoubleType, ArrayType}
import org.apache.spark.sql.functions.{col,from_json,split,explode,abs,exp,pow,sqrt,broadcast}
import scala.collection.mutable.Map
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions._
import java.util.concurrent.TimeUnit.NANOSECONDS
import scala.collection.immutable._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.row_number
import java.util.ArrayList
import scala.collection.mutable
import scala.util.control.Breaks._
import java.math.MathContext
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.functions._
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import breeze.linalg.DenseMatrix
import org.apache.spark.mllib.recommendation.Rating

object App {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
                                .appName("Clustering Application")
                                .getOrCreate()

    val sc = spark.sparkContext
    val sqlContext = new org.apache.spark.sql.SQLContext(sc) //데이터 프레임
    sqlContext.setConf("spark.sql.shuffle.partitions","40")
    import sqlContext.implicits._ // 데이터 프레임 라이브러리       
    
    // val feature_path_query = "http://nginx-file-server.nginx-file-server.svc.cluster.local/dblab/ysh/zata_file/query_rows.csv"
    // val feature_path_whole = "http://nginx-file-server.nginx-file-server.svc.cluster.local/dblab/ysh/zata_file/millon.csv"

    // sc.addFile("http://nginx-file-server.nginx-file-server.svc.cluster.local/dblab/ysh/zata_file/query_rows.csv")
    // sc.addFile("http://nginx-file-server.nginx-file-server.svc.cluster.local/dblab/ysh/zata_file/millon.csv")

    // val feature_path_query = org.apache.spark.SparkFiles.get("query_rows.csv")
    // val feature_path_whole = org.apache.spark.SparkFiles.get("millon.csv")

    // val feature_path_query = SparkFiles.get("query_rows.csv")
    // val feature_path_whole = SparkFiles.get("millon.csv")
    
    val feature_path_query = "query_rows.csv"
    val feature_path_whole = "millon.csv"

    val img_list_query = data_process(feature_path_query, sc).flatMap(_._2)
    val img_list_whole = data_process(feature_path_whole, sc)

    val point = 1000
    //val list = img_list_query.collect.map(x => ((x * point).toInt))
    val list = img_list_query.collect.map(x => x)
    //시간 측정
    var start = System.nanoTime() //나노 초로 시간 측정 
        
    val res = img_list_whole.map { line => 
    //val data = line._2.map(x => ((x * point).toInt))
    val data = line._2.map(x => x)
    val data_res = rbf_kernel(data, list)
    (line._1 , data_res)
    } 
    println("\n\n")
    var end = System.nanoTime()  // 나노 초로 끝 시간 측
    res.sortBy(_._2).take(10).foreach(println) 
    println("\n\n")
    println("==========================================================================================")
    println("                                      RBF Kernel                                          ")
    println("==========================================================================================")     
    println("Completed")
    println("\n\n")
    println("==========================================================================================")
    println("                                      Time Taken                                          ")
    println("==========================================================================================")
    println(s"Time Taken: ${NANOSECONDS.toNanos(end-start)}ns")
    println("\n\n")
    sc.stop()
    //SparkSession 종료     

    spark.stop()
  }
  def data_process (feature_file : String, sc : SparkContext) : RDD[(String , Array[Float])] = {   
    val slength = 4096

    val feature_RDD = sc.textFile(feature_file, minPartitions=8) //RDD생성

    val list = feature_RDD.map { x => 
    val spl = x.replace("\\[|\\]","")
                .replace("(","")
                .replace("))","")
                .replace("Vector","").split(",")
    val key = spl(0)
    val fvals = spl.slice(1,slength).map(x => x.toFloat)    
    (key,fvals)
    }
    return list
  }
  def rbf_kernel(fv1 : Array[Float], fv2 : Array[Float]) = {
    // 0.1 , 0.5 , 10
    val gamma = 0.5
    val sum = fv1.zip(fv2).map {
            case (p1, p2) => 
                val d = p2 - p1
                scala.math.pow(d,2)
            }.sum 
    val res = scala.math.exp(-gamma / sum)
    res
  }
}



// case class DataPoint(value: Vector, time: String)

// object App {
//   def main(args: Array[String]): Unit = {
//     val spark = SparkSession.builder()
//                                 .appName("Clustering Application")
//                                 .getOrCreate()

//     // 두 개의 특징 벡터 리스트 생성
//      val vectorList1 = List(
//       DataPoint(Vectors.dense(1.0, 2.0, 3.0), "2019.10.23 13:00"),
//       DataPoint(Vectors.dense(4.0, 5.0, 6.0), "2019.10.23 14:00"),
//       DataPoint(Vectors.dense(7.0, 8.0, 9.0), "2019.10.23 15:00")
//     )
//     val vectorList2 = List(
//       DataPoint(Vectors.dense(2.0, 3.0, 4.0), "2019.10.23 13:00"),
//       DataPoint(Vectors.dense(5.0, 6.0, 7.0), "2019.10.23 14:00"),
//       DataPoint(Vectors.dense(8.0, 9.0, 10.0), "2019.10.23 15:00")
//     )
//     // 두 개의 특징 벡터 리스트를 RDD로 변환
//     val rdd1 = spark.sparkContext.parallelize(vectorList1.map(_.value))
//     val rdd2 = spark.sparkContext.parallelize(vectorList2.map(_.value))

//     // DTW 연산 수행
//     val startTime = System.currentTimeMillis()
//     val dtwDistances = rdd1.cartesian(rdd2)
//       .map { case (vector1, vector2) => (vector1, vector2, dtwDistance(vector1, vector2)) }
//       .collect()
//     val endTime = System.currentTimeMillis()

//     // DTW 결과 출력
//     dtwDistances.foreach { case (vector1, vector2, distance) =>
//       val point1 = vectorList1.find(_.value == vector1).get
//       val point2 = vectorList2.find(_.value == vector2).get
//       //println(s"DTW distance between ${vector1.toArray.mkString(",")} and ${vector2.toArray.mkString(",")}: $distance")
//       println(s"${point1.time} 시간의 ${vector1.toArray.mkString(",")}와 ${point2.time} 시간의 ${vector2.toArray.mkString(",")} 사이의 DTW 거리: $distance")
//     }

//     // 연산 시간 출력
//     val totalTime = endTime - startTime
//     println(s"Total time: $totalTime ms")
//     //SparkSession 종료     

//     spark.stop()
//   }
//   def dtwDistance(vector1: Vector, vector2: Vector): Double = {
//   val n = vector1.size
//     val m = vector2.size

//     // DTW 행렬 초기화
//     val dtwMatrix = Array.ofDim[Double](n, m)
//     for (i <- 0 until n; j <- 0 until m) {
//       dtwMatrix(i)(j) = Double.PositiveInfinity
//     }
//     dtwMatrix(0)(0) = 0.0

//     // DTW 행렬 계산
//     for (i <- 1 until n; j <- 1 until m) {
//       val cost = distance(vector1(i), vector2(j)) // 두 데이터 간의 거리 함수를 사용하여 거리 계산
//       dtwMatrix(i)(j) = cost + math.min(math.min(dtwMatrix(i-1)(j), dtwMatrix(i)(j-1)), dtwMatrix(i-1)(j-1))
//     }

//     // DTW 거리 반환
//     dtwMatrix(n-1)(m-1)
//   } 
//   def distance(x: Double, y: Double): Double = {
//     // 두 데이터 간의 거리 함수를 구현합니다.
//     // 예를 들어, 유클리드 거리를 사용할 수 있습니다.
//     math.abs(x - y)
//   }
// }
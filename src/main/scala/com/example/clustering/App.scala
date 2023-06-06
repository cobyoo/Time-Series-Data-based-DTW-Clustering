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

case class DataPoint(value: Vector, time: String)

object App {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
                                .appName("DTW Clustering")
                                .getOrCreate()

    // 두 개의 특징 벡터 리스트 생성
     val vectorList1 = List(
      DataPoint(Vectors.dense(0.14528900,0.74144714,0.09088054,0.32330546,0.09808596), "2019.10.23 13:00"),
      DataPoint(Vectors.dense(0.14528901,0.74144715,0.09088055,0.32330547,0.09808597), "2019.10.23 14:00"),
      DataPoint(Vectors.dense(0.14528902,0.74144716,0.09088056,0.32330548,0.09808598), "2019.10.23 15:00"),
      DataPoint(Vectors.dense(0.14528903,0.74144714,0.09088054,0.32330546,0.09808596), "2019.10.23 13:00"),
      DataPoint(Vectors.dense(0.14528904,0.74144715,0.09088055,0.32330547,0.09808597), "2019.10.23 14:00"),
      DataPoint(Vectors.dense(0.14528905,0.74144716,0.09088056,0.32330548,0.09808598), "2019.10.23 15:00"),
      DataPoint(Vectors.dense(0.14528906,0.74144714,0.09088054,0.32330546,0.09808596), "2019.10.23 13:00"),
      DataPoint(Vectors.dense(0.14528907,0.74144715,0.09088055,0.32330547,0.09808597), "2019.10.23 14:00"),
      DataPoint(Vectors.dense(0.14528908,0.74144716,0.09088056,0.32330548,0.09808598), "2019.10.23 15:00"),
      DataPoint(Vectors.dense(0.14528909,0.74144714,0.09088054,0.32330546,0.09808596), "2019.10.23 13:00"),
      DataPoint(Vectors.dense(0.14528901,0.74144715,0.09088055,0.32330547,0.09808597), "2019.10.23 14:00"),
      DataPoint(Vectors.dense(0.14528902,0.74144716,0.09088056,0.32330548,0.09808598), "2019.10.23 15:00"),
      DataPoint(Vectors.dense(0.14528900,0.74144714,0.09088054,0.32330546,0.09808596), "2019.10.23 13:00"),
      DataPoint(Vectors.dense(0.14528901,0.74144715,0.09088055,0.32330547,0.09808597), "2019.10.23 14:00"),
      DataPoint(Vectors.dense(0.14528902,0.74144716,0.09088056,0.32330548,0.09808598), "2019.10.23 15:00"),
      DataPoint(Vectors.dense(0.14528900,0.74144714,0.09088054,0.32330546,0.09808596), "2019.10.23 13:00"),
      DataPoint(Vectors.dense(0.14528901,0.74144715,0.09088055,0.32330547,0.09808597), "2019.10.23 14:00"),
      DataPoint(Vectors.dense(0.14528902,0.74144716,0.09088056,0.32330548,0.09808598), "2019.10.23 15:00"),
      DataPoint(Vectors.dense(0.14528900,0.74144714,0.09088054,0.32330546,0.09808596), "2019.10.23 13:00"),
      DataPoint(Vectors.dense(0.14528901,0.74144715,0.09088055,0.32330547,0.09808597), "2019.10.23 14:00"),
      DataPoint(Vectors.dense(0.14528902,0.74144716,0.09088056,0.32330548,0.09808598), "2019.10.23 15:00")
    )
    val vectorList2 = List(
      DataPoint(Vectors.dense(0.14528903,0.74144717,0.09088057,0.32330549,0.09808599), "2019.10.23 13:00")
    )
    // 두 개의 특징 벡터 리스트를 RDD로 변환
    val rdd1 = spark.sparkContext.parallelize(vectorList1.map(_.value))
    val rdd2 = spark.sparkContext.parallelize(vectorList2.map(_.value))

    // DTW 연산 수행
    val startTime = System.currentTimeMillis()
    val dtwDistances = rdd1.cartesian(rdd2)
      .map { case (vector1, vector2) => (vector1, vector2, dtwDistance(vector1, vector2)) }
      .collect()
    val endTime = System.currentTimeMillis()

    // DTW 결과 출력
    dtwDistances.foreach { case (vector1, vector2, distance) =>
      val point1 = vectorList1.find(_.value == vector1).get
      val point2 = vectorList2.find(_.value == vector2).get
      //println(s"DTW distance between ${vector1.toArray.mkString(",")} and ${vector2.toArray.mkString(",")}: $distance")
      println(s"${point1.time} 시간의 ${vector1.toArray.mkString(",")}와 ${point2.time} 시간의 ${vector2.toArray.mkString(",")} 사이의 DTW 거리: $distance")
    }

    // 연산 시간 출력
    val totalTime = endTime - startTime
    println(s"Total time: $totalTime ms")
    //SparkSession 종료     

    spark.stop()
  }
  def dtwDistance(vector1: Vector, vector2: Vector): Double = {
  val n = vector1.size
    val m = vector2.size

    // DTW 행렬 초기화
    val dtwMatrix = Array.ofDim[Double](n, m)
    for (i <- 0 until n; j <- 0 until m) {
      dtwMatrix(i)(j) = Double.PositiveInfinity
    }
    dtwMatrix(0)(0) = 0.0

    // DTW 행렬 계산
    for (i <- 1 until n; j <- 1 until m) {
      val cost = distance(vector1(i), vector2(j)) // 두 데이터 간의 거리 함수를 사용하여 거리 계산
      dtwMatrix(i)(j) = cost + math.min(math.min(dtwMatrix(i-1)(j), dtwMatrix(i)(j-1)), dtwMatrix(i-1)(j-1))
     }

     // DTW 거리 반환
     dtwMatrix(n-1)(m-1)
   } 
   def distance(x: Double, y: Double): Double = {
     // 두 데이터 간의 거리 함수를 구현합니다.
     // 예를 들어, 유클리드 거리를 사용할 수 있습니다.
     math.abs(x - y)
   }
 }

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.NDArrayIndex
import scala.util.Random

object SingleLinearRegression {

  val numSamples: Int = 1000
  val hyperParams = Map(
    "batchSize" -> 200,
    "epochs" -> 100,
    "learningRate" -> 1e-6,
    "weight" -> Random.nextFloat(),
    "bias" -> Random.nextFloat(),
  )

  val yParams = Map(
    "A" -> 12.34,
    "B" -> 40.23
  )

  def prepareData(numSamples: Int): INDArray = {
    val X = Nd4j.rand(numSamples).mul(numSamples) // Vector X of dimension [1 x numSamples]
    val Y = X.mul(yParams("A")).add(yParams("B"))
    Nd4j.create(Array(X.toFloatVector,Y.toFloatVector)).transpose()
  }

  def forwardProp(weight: Float, bias: Float, X: INDArray): INDArray = {
    // Assuming Activation as LINEAR i.e. f(x) = x => YPred = f(w.X + b) = w.X = b
    X.mul(weight).add(bias)
  }

  def backwardProp(YPred: INDArray, YTrue: INDArray, X: INDArray): (Float, Float) = {
    // dW = (YPred - YTrue).dot(X) / numSamples
    // dB = (YPred - YTrue).sum / numSamples
    val numSamples = X.length()
    val dWDotProd : Float = YPred.sub(YTrue).mmul(X).getFloat(0.asInstanceOf[Long]) // (YPred - YTrue).dot(X)
    val dBDotProd : Float = YPred.sub(YTrue).toFloatVector.sum // (YPred - YTrue).sum
    (dWDotProd/numSamples, dBDotProd/numSamples)
  }

  def updateParams(weight: Float, bias: Float, dW: Float, dB: Float, learningRate: Double): Map[String, Float] = {
    val weightNew = weight - (learningRate*dW)
    val biasNew = bias - dB
    Map(
      "weight" -> weightNew.asInstanceOf[Float],
      "bias" -> biasNew.asInstanceOf[Float]
    )
  }

  def costFunction(YPred: INDArray, YTrue: INDArray): Float = {
    val numElements = YPred.length()
    // [(YPred - YTrue).dot(YPred - YTrue)] / (2*numElements)
    val costDotProd : Float = YPred.sub(YTrue).mmul(YPred.sub(YTrue)).getFloat(0.asInstanceOf[Long])
    costDotProd/(2*numElements)
  }

  def singleRegressionModel(mode: String, X: INDArray, weight: Float, bias: Float, batchSize: Int, epochs: Int): Map[String, Float] = {
    var updatedParams : Map[String,Float] = Map(
      "weight" -> weight,
      "bias" -> bias
    )
    var cost : Float = 0
    var xCurrent: INDArray = Nd4j.empty()

    for(idx <- 0 to X.rows()/batchSize) {
      if((idx + 1)*batchSize > X.rows()) {
        xCurrent = X.get(
          NDArrayIndex.interval(idx*batchSize, 1, X.rows()),
          NDArrayIndex.interval(0, 1, X.columns())
        )
      }
      else {
        xCurrent = X.get(
          NDArrayIndex.interval(idx*batchSize, 1, (idx + 1)*batchSize),
          NDArrayIndex.interval(0, 1, X.columns())
        )
      }

      for(iEpoch <- 1 to epochs) {
        // Forward Propagation
        val yPred: INDArray = forwardProp(weight = updatedParams("weight"), bias = updatedParams("bias"), X = xCurrent.getColumn(0))

        // Cost
        cost = costFunction(YPred = yPred, YTrue = xCurrent.getColumn(1))

        if(mode.toLowerCase.equals("train")) {
          // Backward Propagation or Optimization
          val (dW, dB) = backwardProp(YPred = yPred, YTrue = xCurrent.getColumn(1), xCurrent.getColumn(0))

          // Update parameters
          updatedParams = updateParams(weight = updatedParams("weight"), bias = updatedParams("bias"), dW = dW, dB = dB, learningRate = hyperParams("learningRate").asInstanceOf[Double])
        }

        // Status at Epochs
        if(iEpoch % 10 == 0) {
          println(f"Batch - $idx%d, Epoch - $iEpoch%d, Weight - ${updatedParams("weight")}%2.3f, Bias - ${updatedParams("bias")}%2.3f, Cost - $cost%2.2f")
        }

      }
      print("\n")
    }

    Map(
      "weight" -> updatedParams("weight"),
      "bias" -> updatedParams("bias"),
      "cost" -> cost
    )
  }

  def main(args: Array[String]): Unit = {
    // Set Data Parameters
    val splitPercent: Double = 75.00
    val allData: INDArray = prepareData(numSamples)

    // Prepare Train and Validation data
    val xTrain: INDArray = allData.get(
      NDArrayIndex.interval(0,1,(splitPercent*0.01*allData.rows()).asInstanceOf[Int]),
      NDArrayIndex.interval(0,1,allData.columns())
    )
    val xValid: INDArray = allData.get(
      NDArrayIndex.interval((splitPercent*0.01*allData.rows()).asInstanceOf[Int],1,allData.rows()),
      NDArrayIndex.interval(0,1,allData.columns())
    )

    // Training
    val trainStatus : Map[String,Float] = singleRegressionModel(
      mode = "TRAIN",
      X = xTrain,
      weight = hyperParams("weight").asInstanceOf[Float],
      bias = hyperParams("bias").asInstanceOf[Float],
      batchSize = hyperParams("batchSize").asInstanceOf[Int],
      epochs = hyperParams("epochs").asInstanceOf[Int]
    )
    println(f"\nModel after Training : weight - ${trainStatus("weight")}%2.2f, bias - ${trainStatus("bias")}%2.2f")
    println(f"Final Cost - ${trainStatus("cost")}%2.2f\n")

    // Validation
    val validStatus : Map[String,Float] = singleRegressionModel(
      mode = "VALIDATION",
      X = xValid,
      weight = trainStatus("weight"),
      bias = trainStatus("bias"),
      batchSize = hyperParams("batchSize").asInstanceOf[Int],
      epochs = hyperParams("epochs").asInstanceOf[Int]
    )
    println(f"Final Cost after Validation - ${validStatus("cost")}%2.2f")
  }
}
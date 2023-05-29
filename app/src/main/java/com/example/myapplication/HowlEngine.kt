package com.example.myapplication

import com.example.myapplication.AudioConstants.bitsPerSample
import com.example.myapplication.AudioConstants.channelCount
import com.example.myapplication.AudioConstants.sampleRate
import com.example.myapplication.AudioStreamingUtils.BytetoFloatArray
import android.content.res.AssetManager
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor
import java.util.Date

class HowlEngine(assetManager: AssetManager) {
    // Properties
    private var howlModule: Module? = null
    private var predHistory: List<Pair<Long, FloatArray>> = emptyList()
    private var labelHistory: List<Pair<Long, Int>> = emptyList()
    private val sequence: IntArray = intArrayOf(0, 1)

    private val smoothingWindowMs: Float = 50.0f
    private val inferenceWindowMs: Long = 2000L
    private val toleranceWindowMs: Float = 500.0f

    private val inferenceByteBufferDuration: Float = 0.5f
    private val inferenceByteBufferSize: Int = (sampleRate * channelCount * bitsPerSample * inferenceByteBufferDuration / 8).toInt()
    private var inferenceByteBuffer: ByteArray = ByteArray(inferenceByteBufferSize)

    private val chunkByteBufferDuration: Float = 0.063f
    private val chunkByteBufferSize: Int = (sampleRate * channelCount * bitsPerSample * chunkByteBufferDuration / 8).toInt()
    private var chunkByteBuffer: ByteArray = ByteArray(chunkByteBufferSize)

    private var currTime: Long = 0L
    private var currLabel: Int? = null
    private var targetState: Int = 0
    private var lastValidTimestamp: Long = 0L

    init {
        this.howlModule = LiteModuleLoader.loadModuleFromAsset(assetManager, "howl_mobile.ptl")
        println("Model loaded")
    }

    private fun findInSequence(labelHistory:List<Pair<Long,Int>>): Boolean {
        for (history in labelHistory) {

            val (currTimestamp, label) = history
            val targetLabel = sequence[targetState]
            if (label == targetLabel) {
                // move to next state
                targetState += 1
                if (targetState == sequence.size) {
                    // goal state is reached
                    return true
                }
                currLabel = sequence[targetState - 1]
                lastValidTimestamp = currTimestamp
            } else if (label == currLabel) {
                // label has not changed, only update lastValidTimestamp
                lastValidTimestamp = currTimestamp
            } else if (lastValidTimestamp + toleranceWindowMs < currTimestamp) {
                // out of tolerance window, start from the first state
                currLabel = null
                targetState = 0
                lastValidTimestamp = 0
            }

        }
        return false
    }

    private fun getMaxValueFromLattice(): Int {
        val lattice = Array(predHistory.size) { i ->
            predHistory[i].second
        }

        // Stack the 2D array along the first axis
        val stackedLattice = Array(lattice[0].size) { i ->
            FloatArray(lattice.size) { j ->
                lattice[j][i]
            }
        }

        // Find the maximum value along the first axis of the stacked 2D array
        val latticeMax = FloatArray(stackedLattice.size) { i ->
            var maxVal = Float.NEGATIVE_INFINITY
            for (j in stackedLattice[i].indices) {
                if (stackedLattice[i][j] > maxVal) {
                    maxVal = stackedLattice[i][j]
                }
            }
            maxVal
        }

        // Find the index of the maximum value
        var maxLabel = 0
        for (i in latticeMax.indices) {
            if (latticeMax[i] > latticeMax[maxLabel]) {
                maxLabel = i
            }
        }

        return maxLabel
    }

    fun reset(){
        inferenceByteBuffer = ByteArray(inferenceByteBufferSize)
        chunkByteBuffer = ByteArray(chunkByteBufferSize)
        predHistory = emptyList()
        labelHistory = emptyList()
        currLabel = null
        targetState = 0
        lastValidTimestamp = 0
    }

    fun infer(byteBuffer: ByteArray): Boolean {
        val audioFloatArr = BytetoFloatArray(byteBuffer)
        val shape = longArrayOf(1, audioFloatArr.size.toLong())
        val audioTensor = Tensor.fromBlob(audioFloatArr, shape)

        val prediction = howlModule?.forward(IValue.from(audioTensor))?.toTensor()
        val predictionFloatArray = prediction?.dataAsFloatArray

        currTime = Date().time
        predHistory = predHistory.plus(Pair(currTime,predictionFloatArray!!))
        predHistory = predHistory.dropWhile { currTime - it.first > smoothingWindowMs }.toMutableList()

        val maxLabel = getMaxValueFromLattice()

        labelHistory = labelHistory.plus(Pair(currTime, maxLabel))

        //state machine
        currTime = Date().time

        labelHistory = labelHistory.dropWhile { currTime - it.first > inferenceWindowMs }.toMutableList()

        return if(findInSequence(labelHistory)){
            reset()
            true

        } else{
            false
        }


    }

    // ... Additional methods as needed ...
}
package com.example.myapplication

import android.content.res.AssetManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.util.Base64
import android.util.Log
import android.util.MutableLong
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.databinding.DataBindingUtil
import com.example.myapplication.databinding.ActivityMainBinding
import com.example.myapplication.AudioStreamingUtils.windowStriding
import io.socket.client.IO
import io.socket.client.Socket
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay
import org.json.JSONObject
import org.pytorch.IValue
import java.net.URISyntaxException

import org.pytorch.Module
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder.LITTLE_ENDIAN
import java.util.Date
import java.nio.FloatBuffer


class AudioWebSocket(mainBinding: ActivityMainBinding, assetManager: AssetManager) {
    private val TAG = "AudioWebSocket"
    private val baseURL = "http://10.0.2.2:8011"
    private val ws_path = "/ws/socket.io/"
    private val audioContentType = "audio/wav"
    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val channelCount = if (channelConfig == AudioFormat.CHANNEL_IN_MONO) 1 else 2
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val bitsPerSample = 16 // for ENCODING_PCM_FLOAT
    private val bufferDuration = 0.1 // 100ms buffer
    private val bufferSize = (sampleRate * channelCount * bitsPerSample * bufferDuration / 8).toInt()
    private val NUMBER_OF_CHUNKS = 10 //was 10
//    private val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
//    private var client = OkHttpClient()
//    private var webSocket: WebSocket? = null
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var binding: ActivityMainBinding = mainBinding
    private var assetManager: AssetManager ?= assetManager
    private var howlModule: Module? = null
    private var howlEngine: WakeWordEngine? = null
    private var sileroVAD: Module? = null
    private lateinit var mSocket: Socket
    public enum class MyEnum(val value: String) {
        HEY("halo"),
        //FIRE("fire"),
       // FOX("fox"),
        NON_WAKE_WORD("non_wake_word");

        companion object {
            fun fromValue(value: Int) = values().first { it.ordinal == value }
        }
    }
    private val eval_stride_size_ms: Float = 63.00f
    private var curr_time: Float = 0.00f



    //საჭირო ცვალდები
    private var predHistory = emptyList<Pair<Long,FloatArray>>()
    private var labelHistory = emptyList<Pair<Long,Int>>()
    private val sequence: IntArray = intArrayOf(0, 1)//intArrayOf(0, 1, 2, 3)

    private val smoothing_window_ms: Float = 50.0f
    private val inferenceWindowMs: Long = 2000L
    private val toleranceWindowMs = 500.0f

    private val inferenceByteBufferDuration = 0.5
    private val inferenceByteBufferSize = (sampleRate * channelCount * bitsPerSample * inferenceByteBufferDuration / 8).toInt()
    private val inferenceByteBuffer = ByteArray(inferenceByteBufferSize)
    private val chunkByteBufferDuration = 0.063
    private val chunkByteBufferSize = (sampleRate * channelCount * bitsPerSample * chunkByteBufferDuration / 8).toInt()
    private val chunkByteBuffer = ByteArray(chunkByteBufferSize)

    init {
        this.howlModule =  LiteModuleLoader.loadModuleFromAsset(assetManager,"howl_mobile.ptl")
        this.howlEngine = WakeWordEngine(assetManager)
        this.sileroVAD = LiteModuleLoader.loadModuleFromAsset(assetManager,"sileroVAD.ptl")

        println("model loaded")
    }



    fun startRecording() {
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioFormat,
            bufferSize
        )
        // Get the AudioManager instance
        audioRecord?.startRecording()
        isRecording = true
        connect()

    }

    fun stopRecording(delayMillis: Long = 1000) {
        GlobalScope.launch {
            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null
            isRecording = false
            mSocket.disconnect()
        }
    }



     private fun connect() {

        try {
            val options = IO.Options.builder().setPath(ws_path).build()
            mSocket = IO.socket("http://10.0.2.2:8011/", options).connect()
            mSocket.on("transcribed") { data ->
                val message = (data[0] as JSONObject).getString("transcription")
                binding.transcribedText.text = message
            }
            sendAudio()

        } catch (e: URISyntaxException) {
            Log.e(TAG, e.message.toString())
        }


    }

    fun stride(audioData: Tensor, windowMs: Int, strideMs: Int, sampleRate: Int, dropIncomplete: Boolean = true): Sequence<Tensor> {
        val chunkSize = (windowMs / 1000.0 * sampleRate).toInt()
        val strideSize = (strideMs / 1000.0 * sampleRate).toInt()
        var currIdx = 0

        return generateSequence {
            if (currIdx + chunkSize >= audioData.shape()[1]) {
                null
            } else {
                val scliceFloatArray = audioData.dataAsFloatArray.slice(currIdx until currIdx + chunkSize).toFloatArray()
                val shape = longArrayOf(1, scliceFloatArray.size.toLong())
                val sliced = Tensor.fromBlob(scliceFloatArray,shape)
                currIdx += strideSize
                sliced
            }
        }
    }

    fun decode_label(predictionFloatArray:FloatArray?): MyEnum {
        var maxIndex = 0
        for (i in 1 until predictionFloatArray!!.size) {
            if (predictionFloatArray[i] > predictionFloatArray[maxIndex]) {
                maxIndex = i
            }
        }
        return MyEnum.fromValue(maxIndex)
    }

    private fun findInSequence(labelHistory:List<Pair<Long,Int>>): Boolean {
        var currLabel: Nothing? = null
        var targetState: Int = 0
        var lastValidTimestamp: Long  = 0L

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
                val currLabel = sequence[targetState - 1]
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

    private fun sendAudio() {

        Thread {

//            var bufferToSend = ByteArray(0)

//            val bufferArray = ByteArray(bufferSize) // buffer size is in bytes, so divide by 2 to get the number of shorts


            while (isRecording) {



//                System.arraycopy(inferenceByteBuffer, chunkByteBufferSize, inferenceByteBuffer, 0, inferenceByteBuffer.size - chunkByteBufferSize)
//
//                // Copy the new audio data to the end of the buffer
//                System.arraycopy(chunkByteBuffer, 0, inferenceByteBuffer, inferenceByteBuffer.size - chunkByteBufferSize, chunkByteBuffer.size)



                if(howlEngine!!.infer(inferenceByteBuffer)) {
                    binding.transcribedText.text = "                                 სოფიფი"
                    GlobalScope.launch {
                        delay(2000)
                        binding.transcribedText.text = ""
                        //

                    }
//                    val audioFloatArr = AudioStreamingUtils.BytetoFloatArray(inferenceByteBuffer)
//                    val shape = longArrayOf(1, audioFloatArr.size.toLong())
//                    val audioTensor = Tensor.fromBlob(audioFloatArr, shape)
////
//                    val iValue = IValue.from(audioTensor)
//                    val srTensor = Tensor.fromBlob(floatArrayOf(16000f), longArrayOf(1))
//                    val srIValue = IValue.from(srTensor)
                    val wav = FloatArray(4000) // Fill this with your actual audio data

                    // Create tensors from the data
                    val wavTensor = Tensor.fromBlob(wav.sliceArray(0 until 800), longArrayOf(1, 800))
                    val rateTensor = Tensor.fromBlob(longArrayOf(16000), longArrayOf(1))

                    // Wrap the tensors in IValues
                    val wavIValue = IValue.from(wavTensor)
                    val rateIValue = IValue.from(rateTensor)
                    val inputs = IValue.listFrom(wavIValue, rateIValue)

                    val prediction = sileroVAD?.forward(inputs)?.toTensor()


                }
                else{
                    val bytesRead = audioRecord?.read(chunkByteBuffer, 0, chunkByteBufferSize) ?: 0
                    windowStriding(inferenceByteBuffer,chunkByteBuffer,chunkByteBufferSize)

                }






//////////////////////////////////////////////////////////////////////////////////////////////////////////


//                // Read audio data from AudioRecord
//                if (bufferToSend.size != NUMBER_OF_CHUNKS * bufferSize){
//
//                    val bytesRead = audioRecord?.read(bufferArray, 0, bufferSize) ?: 0
//                    bufferToSend += bufferArray
//                }
//                else{
//
//
//                    val shorts = ShortArray(bufferToSend.size / 2)
//                    ByteBuffer.wrap(bufferToSend).order(LITTLE_ENDIAN).asShortBuffer()[shorts]
//                    var floatArr = FloatArray(shorts.size)
//
//                    for (i in shorts.indices) {
//                        floatArr[i] = shorts[i] / Short.MAX_VALUE.toFloat()
//
//                    }
//
//
//
//
//                    val shape = longArrayOf(1, floatArr.size.toLong())
//
//                    val audioData = Tensor.fromBlob(floatArr, shape)
//
////                    val audioData = tensor // initialize the audio data tensor
//                    val windowMs = 500// set the window size in milliseconds
//                    val strideMs = 63// set the stride size in milliseconds
//                    val sampleRate = 16000// set the sample rate
//                    val dropIncomplete = true // set the flag to drop incomplete tensors
//
//                    val tensorSequence = stride(audioData, windowMs, strideMs, sampleRate, dropIncomplete)
//
//
//                    for (tensor in tensorSequence) {
//                        if (tensor == null){
//                            continue
//                        }
//                        val prediction = howlModule?.forward(IValue.from(tensor))?.toTensor()
//
//
//                        val predictionFloatArray = prediction?.dataAsFloatArray
////                        println(predictionFloatArray.contentToString())
//                        pred_history.plus(Pair(curr_time,predictionFloatArray))
//                        var maxIndex = 0
//                        for (i in 1 until predictionFloatArray!!.size) {
//                            if (predictionFloatArray[i] > predictionFloatArray[maxIndex]) {
//                                maxIndex = i
//                            }
//                        }
//
//
//                        println(MyEnum.fromValue(maxIndex))
//                    }
//
//
//
////                    val base64EncodedString = Base64.encodeToString(bufferToSend, Base64.DEFAULT)
////                    val message = JSONObject()
////                    message.put("audio_chunk", base64EncodedString)
////                    // Send the request through the WebSocket
////                    mSocket?.emit( "transcribe", message )
////                    // reset bufferToSend
//                    bufferToSend = ByteArray(0)
//                }




            }
        }.start()
    }

}


class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var audioWebSocket: AudioWebSocket
    private var REQUEST_RECORD_AUDIO = 13
//    private val modelPath = Paths.get("assets/howl_mobile.ptl")
    private val assetPath = "assets/howl_mobile.ptl"
//    private val inputStream = assetManager.open(assetPath)
    private lateinit var assetManager: AssetManager
//    LiteModuleLoader.loadModuleFromAsset(assetManager,assetPath)
//    private lateinit var startButton: Button
//    private lateinit var stopButton: Button


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        fun requestMicrophonePermission() {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(android.Manifest.permission.RECORD_AUDIO),
                    REQUEST_RECORD_AUDIO
                )
            }
        }


//        val assetManager = applicationContext.assets
//        mModuleEncoder = LiteModuleLoader.loadModuleFromAsset(assetManager,assetPath)


        assetManager = applicationContext.assets




        requestMicrophonePermission()

        binding = DataBindingUtil.setContentView(this, R.layout.activity_main)

        //Create an instance of AudioWebSocket
        audioWebSocket = AudioWebSocket(binding,assetManager)


        // Start recording and sending audio through WebSocket when button is clicked
        binding.mButton.setOnClickListener {
            if (binding.mButton.text == "Start"){
                audioWebSocket.startRecording()
                binding.mButton.text = "Listening... Stop"
            }
            else{
                audioWebSocket.stopRecording()
                binding.mButton.text = "Start"
            }

        }

    }
}
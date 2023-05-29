package com.example.myapplication

import java.nio.ByteBuffer
import java.nio.ByteOrder

object AudioStreamingUtils {
    fun BytetoFloatArray(byteBuffer:ByteArray): FloatArray {
        val shorts = ShortArray(byteBuffer.size / 2)
        ByteBuffer.wrap(byteBuffer).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()[shorts]
        var audioFloatArr = FloatArray(shorts.size)

        for (i in shorts.indices) {
            audioFloatArr[i] = shorts[i] / Short.MAX_VALUE.toFloat()

        }
        return audioFloatArr
    }
}
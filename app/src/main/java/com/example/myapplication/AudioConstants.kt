package com.example.myapplication

import android.media.AudioFormat

object AudioConstants {
    const val sampleRate = 16000
    const val channelConfig = AudioFormat.CHANNEL_IN_MONO
    var channelCount = if (channelConfig == AudioFormat.CHANNEL_IN_MONO) 1 else 2
    const val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    const val bitsPerSample = 16 // for ENCODING_PCM_FLOAT
}
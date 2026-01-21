package com.k2fsa.sherpa.onnx

data class HomophoneReplacerConfig(
    var lexicon: String = "",
    var ruleFsts: String = "",
)
# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in the SDK tools.

# Preserve JNI methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Preserve sherpa-onnx classes
-keep class com.k2fsa.sherpa.onnx.** { *; }

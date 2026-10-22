plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "ai.dogsense.hud"
    compileSdk = 35
    defaultConfig {
        applicationId = "ai.dogsense.hud"
        minSdk = 28
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        // Inference server on the LAN (GPU box). Override: ./gradlew assembleDebug -PserverUrl=ws://10.0.0.5:8765/ws
        buildConfigField("String", "SERVER_URL", "\"${project.findProperty("serverUrl") ?: "ws://192.168.1.100:8765/ws"}\"")
    }
    buildFeatures { buildConfig = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    val camerax = "1.4.1"
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}

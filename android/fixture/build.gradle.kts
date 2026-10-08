plugins { id("com.android.application") }
android {
    namespace = "ro.ubb.uicollector.fixture"
    compileSdk = 35
    defaultConfig { applicationId = "ro.ubb.uicollector.fixture"; minSdk = 30; targetSdk = 35; versionCode = 1; versionName = "1.0" }
    flavorDimensions += "visit"
    productFlavors {
        create("a") { dimension = "visit"; applicationIdSuffix = ".a"; resValue("string","app_name","Collector Test A") }
        create("b") { dimension = "visit"; applicationIdSuffix = ".b"; resValue("string","app_name","Collector Test B") }
    }
}

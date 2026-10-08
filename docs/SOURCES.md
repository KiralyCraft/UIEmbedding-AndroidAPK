# Primary implementation references

These informed platform decisions. They do not constitute evidence that the delivered source has been compiled or run on a phone.

- Android MediaProjection: consent per session, foreground service, whole-display sharing, single-use token/VirtualDisplay, resize/stop callbacks and screen-lock behavior: https://developer.android.com/media/grow/media-projection
- Android UsageEvents.Event: activity lifecycle events and class/package metadata: https://developer.android.com/reference/android/app/usage/UsageEvents.Event
- Android AccessibilityEvent: event/window classes and package information: https://developer.android.com/reference/android/view/accessibility/AccessibilityEvent
- WorkManager persistent work: https://developer.android.com/develop/background-work/background-tasks/persistent/getting-started
- LiteRT GPU Interpreter integration and same-thread requirement: https://developers.google.com/edge/litert/android/gpu
- LiteRT GPU supported operations, partitioning and performance considerations: https://developers.google.com/edge/litert/performance/gpu
- TIMM 1.0.20 MobileNet implementation: https://github.com/huggingface/pytorch-image-models/blob/v1.0.20/timm/models/mobilenetv3.py
- TIMM 1.0.20 inverted residual/BatchNormAct/layer-scale behavior: https://github.com/huggingface/pytorch-image-models/blob/v1.0.20/timm/models/_efficientnet_blocks.py and https://github.com/huggingface/pytorch-image-models/blob/v1.0.20/timm/layers/norm_act.py
- GPU delegate precision/sustained-speed option definitions: https://github.com/tensorflow/tensorflow/blob/v2.19.0/tensorflow/lite/delegates/gpu/java/src/main/java/org/tensorflow/lite/gpu/GpuDelegateFactory.java
- OpenCV external ByteBuffer/stride-backed Mat: https://docs.opencv.org/4.10.0/javadoc/org/opencv/core/Mat.html
- Android native-library 16-KB-page compatibility: https://developer.android.com/guide/practices/page-sizes

The model/preprocessing authority is the user-supplied source and checkpoint, not a generic MobileNet configuration. See models/provenance.json and the retained original implementation under vendor/ui_foundation.

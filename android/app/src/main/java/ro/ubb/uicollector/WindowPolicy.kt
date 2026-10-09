package ro.ubb.uicollector

/** Window descriptors contain package identity and geometry/focus metadata, never UI text. */
enum class WindowKind { APPLICATION, INPUT_METHOD, SYSTEM, ACCESSIBILITY_OVERLAY, OTHER }
data class VisibleWindow(val id: Int, val kind: WindowKind, val layer: Int, val focused: Boolean, val active: Boolean, val packageName: String?, val className: String? = null)
data class CaptureWindowContext(val screenKind: String = "application", val keyboardVisible: Boolean = false, val visiblePackages: List<String> = emptyList())
data class ResolvedWindow(val primary: VisibleWindow, val context: CaptureWindowContext)

/** The IME is part of its host screen, never the app label for that screen. */
fun resolveCaptureWindow(windows: List<VisibleWindow>): ResolvedWindow? {
    val candidates=windows.filter { it.kind != WindowKind.INPUT_METHOD && it.kind != WindowKind.OTHER }
    val applications=candidates.filter { it.kind == WindowKind.APPLICATION }
    val primary=candidates.filter { it.focused || it.active }.maxByOrNull { it.layer }
        ?: applications.singleOrNull() ?: return null
    if(primary.packageName.isNullOrBlank()) return null
    val packages=(windows.filter { it.kind == WindowKind.APPLICATION || it.kind == WindowKind.INPUT_METHOD || it.id == primary.id }
        .mapNotNull { it.packageName } + primary.packageName).distinct().sorted()
    return ResolvedWindow(primary,CaptureWindowContext(
        screenKind=if(primary.kind==WindowKind.APPLICATION) "application" else "system_overlay",
        keyboardVisible=windows.any { it.kind==WindowKind.INPUT_METHOD },
        visiblePackages=packages
    ))
}

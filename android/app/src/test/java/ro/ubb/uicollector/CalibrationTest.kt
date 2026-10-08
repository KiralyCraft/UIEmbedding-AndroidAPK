package ro.ubb.uicollector

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.ceil

class CalibrationTest {
    private fun simulate(calibration: Calibration, sample: (Double, Boolean) -> Pair<Boolean, Double>): Calibration {
        var now=1000L
        repeat(2000) {
            val fps=calibration.fps
            val (rejected,work)=sample(fps,calibration.description.startsWith("Sustained"))
            val start=now
            now+=ceil(1000.0/fps).toLong()
            if(calibration.record(start,now,work,!rejected,false,rejected)) return calibration
        }
        error("Calibration did not terminate")
    }

    @Test fun screenshotRejectionEndsAscentAndConfirmsLastPassingRate() {
        val calibration=simulate(Calibration()) { fps,_ -> Pair(fps>=4.0,80.0) }
        assertTrue(calibration.hasSustainableRate)
        assertEquals(3.5,calibration.selectedFps,0.0)
        assertEquals(1,calibration.results.count { it.rateLimited })
        assertTrue(calibration.results.none { it.targetFps>4.0 })
        assertTrue(calibration.results.last().confirmation)
        assertTrue(calibration.results.last().durationMs>=30_000)
    }

    @Test fun pipelineBudgetCanBeTheLimitWithoutAndroidRejection() {
        val calibration=simulate(Calibration()) { _,_ -> Pair(false,280.0) }
        assertEquals(2.5,calibration.selectedFps,0.0)
        assertTrue(calibration.results.none { it.rateLimited })
        assertTrue(calibration.hasSustainableRate)
    }

    @Test fun failedConfirmationMustConfirmTheLowerRate() {
        val calibration=simulate(Calibration()) { fps,confirming -> Pair(fps>=4.0 || (confirming && fps>=3.5),80.0) }
        assertEquals(3.0,calibration.selectedFps,0.0)
        assertTrue(calibration.results.last().confirmation && calibration.results.last().sustainable)
    }

    @Test fun noScreenshotsNeverProducesSuccessfulCalibration() {
        val calibration=Calibration()
        var finished=false
        var now=1000L
        repeat(20) {
            if(!finished) { finished=calibration.record(now,now+4000,1.0,false,false);now+=4000 }
        }
        assertTrue(finished)
        assertFalse(calibration.hasSustainableRate)
        assertEquals(0.0,calibration.results.last().achievedFps,0.0)
    }
}

package ro.ubb.uicollector

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RateControllerTest
{
    @Test
    fun slowFramesReduceRateWithoutExceedingCeiling()
    {
        val controller = RateController(30.0)
        repeat(40) { controller.observe(200.0, 0) }
        assertEquals(3.5, controller.fps, 0.0)
        repeat(40) { controller.observe(20.0, 3) }
        assertEquals(1.75, controller.fps, 0.0)
        val limited = RateController(5.0)
        repeat(400) { limited.observe(1.0, 0) }
        assertTrue(limited.fps <= 5.0)
    }

    @Test
    fun percentileUsesTailRatherThanAverage()
    {
        assertEquals(1000.0, percentile(List(19) { 1.0 } + listOf(1000.0, 1000.0), 0.95), 0.0)
    }
}

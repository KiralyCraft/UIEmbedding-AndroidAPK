package ro.ubb.uicollector

import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test

/** Run before granting permissions, on an onboarding test installation. */
class OnboardingInstrumentedTest {
    @get:Rule val compose=createAndroidComposeRule<MainActivity>()
    @Test fun notificationStepCannotAdvanceWithoutGrantAndMenusRemainAccessible() {
        val app=compose.activity.application as RecorderApp
        assertTrue("Revoke this app's notification permission before running this test",Permissions.inspect(app).notifications != GrantState.GRANTED)
        val completed=app.preferences.getBoolean("onboarding_complete",false)
        val step=app.preferences.getInt("onboarding_step",0)
        try {
            app.preferences.edit().putBoolean("onboarding_complete",false).putInt("onboarding_step",0).commit()
            compose.activityRule.scenario.recreate()
            compose.onNodeWithText("Next").performClick()
            compose.onNodeWithText("Grant Notifications").assertExists()
            compose.onNodeWithText("Next").assertIsNotEnabled()
            compose.onNodeWithText("Applications").performClick()
            compose.onNodeWithText("All applications by default").assertExists()
            compose.onNodeWithText("Settings").performClick()
            compose.onNodeWithText("Permissions and readiness").performClick()
            compose.onNodeWithText("Capture method").assertExists()
            assertFalse(Permissions.inspect(app).ready)
        } finally {
            app.preferences.edit().putBoolean("onboarding_complete",completed).putInt("onboarding_step",step).commit()
        }
    }
}

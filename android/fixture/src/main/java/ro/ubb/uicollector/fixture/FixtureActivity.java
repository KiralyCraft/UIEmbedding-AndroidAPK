package ro.ubb.uicollector.fixture;

import android.app.Activity;
import android.os.Bundle;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.View;
import android.view.WindowManager;

/** Non-sensitive repeatable screen; test APK only, never bundled in the collector. */
public class FixtureActivity extends Activity {
    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        final boolean animate=getIntent().getBooleanExtra("animate",true);
        final boolean variant=getPackageName().endsWith(".b");
        setContentView(new View(this) {
            final Paint paint=new Paint(Paint.ANTI_ALIAS_FLAG);
            int phase=0;
            @Override protected void onDraw(Canvas canvas) {
                canvas.drawColor(variant ? Color.rgb(235,245,255) : Color.rgb(250,235,245));
                paint.setColor(Color.DKGRAY);paint.setTextSize(58);
                canvas.drawText(variant ? "Controlled test application B" : "Controlled test application A",30,180,paint);
                for(int i=0;i<18;i++) {
                    paint.setColor(Color.rgb((phase+i*31)%255,(i*47)%255,150));
                    float x=(phase*3+i*70)%Math.max(getWidth(),1);
                    canvas.drawRect(x,240+i*52,x+110,275+i*52,paint);
                }
                phase=(phase+1)%10000;
                if(animate) postInvalidateDelayed(16);
            }
        });
    }
}

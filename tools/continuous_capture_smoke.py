#!/usr/bin/env python3
"""Exercise continuous recording and unlock resume on an explicitly selected phone.

Install both :fixture flavors first. Other Accessibility services and queued data
are preserved. Recording stops in finally, including on test failures.
"""
from __future__ import annotations
import argparse
from collections.abc import Callable
import json
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as ET


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial")
    parser.add_argument("--saved-calibration",action="store_true")
    parser.add_argument("--rates-only",action="store_true",help="Calibrate and exercise all three modes without locking the phone")
    args=parser.parse_args()
    if args.rates_only and args.saved_calibration:
        parser.error("--rates-only requires fresh calibration; omit --saved-calibration")
    report=Path(__file__).resolve().parents[1]/"reports/work"
    report.mkdir(parents=True,exist_ok=True)

    def adb(*parts: str) -> str:
        return subprocess.check_output(["adb","-s",args.serial,*parts],text=True,timeout=45).strip()

    def preferences() -> dict[str,str]:
        root=ET.fromstring(adb("shell","run-as","ro.ubb.uicollector","cat","shared_prefs/settings.xml"))
        return {item.attrib["name"]:item.attrib.get("value",item.text or "") for item in root}

    def tap(label: str) -> None:
        for attempt in range(5):
            adb("shell","uiautomator","dump","/data/local/tmp/ui-collector-smoke.xml")
            root=ET.fromstring(adb("shell","cat","/data/local/tmp/ui-collector-smoke.xml"))
            parents={child:parent for parent in root.iter() for child in parent}
            for node in root.iter("node"):
                if node.get("text")!=label: continue
                target=node
                while target.get("clickable")!="true" and target in parents:
                    if label in ("Battery Saver","Balanced","Maximum Detail") and target.get("checked")=="true": return
                    target=parents[target]
                if label in ("Battery Saver","Balanced","Maximum Detail") and target.get("checked")=="true": return
                if target.get("enabled")!="true" or target.get("clickable")!="true": continue
                x1,y1,x2,y2=map(int,re.findall(r"\d+",target.get("bounds","")))
                adb("shell","input","tap",str((x1+x2)//2),str((y1+y2)//2))
                time.sleep(1)
                return
            if attempt>0: adb("shell","input","swipe","540","1700","540","700","350")
        raise RuntimeError(f"Enabled control not found: {label}")

    def launch(variant: str, seconds: int=10) -> None:
        adb("shell","am","start","-W","-n",f"ro.ubb.uicollector.fixture.{variant}/ro.ubb.uicollector.fixture.FixtureActivity")
        print(f"Controlled application {variant}: {seconds}s",flush=True)
        time.sleep(seconds)

    def stop() -> None:
        result=adb("shell","run-as","ro.ubb.uicollector","am","startservice","--user","0","-n","ro.ubb.uicollector/.CaptureService","-a","STOP")
        if "Exception" in result or "Error" in result: raise RuntimeError("Could not stop recording")
        time.sleep(2)

    adb("shell","am","start","-W","-n","ro.ubb.uicollector/.MainActivity")
    stop()
    try:
        adb("shell","pm","grant","ro.ubb.uicollector","android.permission.POST_NOTIFICATIONS")
        adb("shell","appops","set","ro.ubb.uicollector","GET_USAGE_STATS","allow")
        adb("shell","appops","set","ro.ubb.uicollector","ACCESS_RESTRICTED_SETTINGS","allow")
        current=adb("shell","settings","get","secure","enabled_accessibility_services")
        services=[] if current in ("null","") else current.split(":")
        # Instrumentation deliberately kills its target, which can leave this service
        # enabled but in Android's crashed/unbound list. Rebind only our component.
        others=[item for item in services if not item.startswith("ro.ubb.uicollector/")]
        adb("shell","settings","put","secure","enabled_accessibility_services",":".join(others))
        time.sleep(1)
        services=others+["ro.ubb.uicollector/ro.ubb.uicollector.LabelAccessibilityService"]
        adb("shell","settings","put","secure","enabled_accessibility_services",":".join(services))
        adb("shell","settings","put","secure","accessibility_enabled","1")
        time.sleep(2)
        bound=adb("shell","dumpsys","accessibility").split("Bound services:",1)[1].split("Enabled services:",1)[0]
        collector=re.search(r"Service\[label=(?:Embedding collector|UI Embedding Collector)[^\n]*capabilities=(\d+)",bound)
        assert collector and int(collector.group(1)) & 128, "Collector must be bound with screenshot capability"
        adb("shell","am","start","-W","-n","ro.ubb.uicollector/.MainActivity")
        previous=preferences().get("last_benchmark")
        if preferences().get("onboarding_complete")!="true":
            for _ in range(4-int(preferences().get("onboarding_step","0"))): tap("Next")
            tap("Start calibration")
        elif args.saved_calibration:
            tap("Start recording")
        else:
            tap("Settings");tap("Calibration");tap("Recalibrate and start")
        deadline=time.monotonic()+15
        while "isForeground=true" not in adb("shell","dumpsys","activity","services","ro.ubb.uicollector"):
            if time.monotonic()>deadline: raise RuntimeError("Capture never started; refusing to advance to fixture applications")
            time.sleep(1)
        print("Continuous calibration started; leaving animation visible",flush=True)
        deadline=time.monotonic()+300
        while not args.saved_calibration and time.monotonic()<deadline:
            current_report=preferences().get("last_benchmark")
            if current_report and current_report!=previous:
                print(f"Calibration complete: {json.loads(current_report)['selected_fps']} samples/s",flush=True)
                break
            time.sleep(5)
        else:
            if not args.saved_calibration: raise RuntimeError("Calibration did not finish within five minutes")
        if args.rates_only:
            for mode in ("Maximum Detail","Balanced","Battery Saver"):
                adb("shell","am","start","-W","-n","ro.ubb.uicollector/.MainActivity")
                tap(mode)
                launch("a",20)
            stop()
            adb("shell","am","start","-W","-n","ro.ubb.uicollector/.MainActivity")
            tap("Maximum Detail")
            print("All three modes exercised; default restored",flush=True)
        else:
            exercise_unlock(adb,launch)
            adb("shell","am","start","-W","-n","ro.ubb.uicollector/.MainActivity")
            tap("Battery Saver")
            launch("b",12)
    finally:
        stop()
        print("Recording stopped; encrypted data retained",flush=True)
    suite="AdaptiveRatesInstrumentedTest" if args.rates_only else "RecordedVisitsInstrumentedTest"
    report_name="adaptive-rates" if args.rates_only else "controlled-visits"
    result=adb("shell","am","instrument","-w","-r","-e","class",f"ro.ubb.uicollector.{suite}","ro.ubb.uicollector.test/androidx.test.runner.AndroidJUnitRunner")
    (report/f"{report_name}-tests.txt").write_text(result+"\n")
    if "OK (1 test)" not in result: raise RuntimeError(result)
    (report/f"{report_name}-report.json").write_text(adb("shell","run-as","ro.ubb.uicollector","cat",f"cache/{report_name}-report.json")+"\n")
    print("Controlled capture acceptance passed",flush=True)


def exercise_unlock(adb: Callable[..., str], launch: Callable[..., None]) -> None:
    launch("a");launch("b");launch("a")
    print("Locking for eight seconds; checking automatic resume after unlock",flush=True)
    adb("shell","input","keyevent","223");time.sleep(8)
    adb("shell","input","keyevent","224")
    print("WAITING_FOR_OWNER_UNLOCK: please unlock the phone normally",flush=True)
    deadline=time.monotonic()+300
    while not re.search(r"\bshowing=false\b",adb("shell","dumpsys","window","policy")):
        if time.monotonic()>deadline: raise RuntimeError("Phone is still locked; owner must unlock before this test")
        time.sleep(1)
    launch("a",12)
    assert "null" in adb("shell","dumpsys","media_projection"), "Continuous capture must not own a projection"


if __name__=="__main__":
    main()

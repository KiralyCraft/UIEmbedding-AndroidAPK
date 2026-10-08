# Validation scope

Use [DEVICE_RESULTS.md](DEVICE_RESULTS.md) for measured results and outstanding limitations. Raw reports, device identifiers, captures and generated model files are intentionally excluded from Git.

Checks include strict F6 checkpoint loading; 36 export comparisons; actual-phone complete-encoder OpenCL delegation and 100-invocation parity; RGBA preprocessing parity across eight aspect ratios; repeated preprocessing backend comparisons; permission gating; recording policies and rate ceilings; SQLite durability, acknowledgement races and atomic ownership transfer in isolated databases; and SQLite-backed server API tests.

The controlled continuous-capture run checks actual full-display capture, encrypted committed samples, repeated application visits, screen lock/unlock and Battery Saver. A successful model benchmark alone does not establish continuous recording. Android's capture rate limit is separate from encoder speed.

Not established by these checks: 24-hour endurance, energy consumption while charging, real MySQL/proxy integration or 16-KB native-library compatibility. Consult [the acceptance checklist](../docs/DEVICE_ACCEPTANCE.md) before deployment beyond the tested device.

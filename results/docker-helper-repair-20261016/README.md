# Approved Docker helper repair and resumed Model 2 setup

Docker Desktop 4.95.0 (242831) now starts successfully. `docker version` reports
both client and server, Docker Engine 29.8.2, server linux/arm64.

After explicit user approval, unloaded the two stale Docker helper services,
removed only /Library/PrivilegedHelperTools/com.docker.vmnetd and com.docker.socket,
then replaced them with the verified current copies bundled inside Docker.app,
restored root:wheel ownership and mode 755, and loaded their existing services.
This follows Docker's documented helper replacement procedure:
https://docs.docker.com/desktop/cert-revoke-solution/

macOS administrator authentication was handled through its own dialog. Both new
system helpers pass codesign verification outside the sandbox and cmp confirms
they are identical to the bundled copies. The signed Docker team is
Docker Inc (9BNSXJN65R); the app previously passed Gatekeeper notarization checks.
The old vmnetd signature timestamp was July 2024, contrasting with the current
bundled October 2026 helper. Stale Docker Desktop processes were stopped and
Docker restarted cleanly. No Docker data, images, containers, development
environments, model artifacts or macOS security settings were removed/changed.

Resumed the staged Participant 1 FVP project using the now-installed runtime and
device packs, CMSIS-Toolbox 2.15.0, CMake 4.2.1, Ninja 1.13.2, Arm Compiler 6.24.0.
The build reaches the CMake compiler test, then fails:

```
armclang: error: Failed to check out a license.
The license file could not be found.
```

Full diagnostic: model2-build.log. No EEG application/firmware build or FVP
execution has passed. No model/operator error has been established.

Remaining user-controlled prerequisites:
1. Activate the free Keil MDK Community license through Keil Studio using an Arm
   account, then use its tool-activated terminal so the compiler sees the license.
2. Personally review and accept FVP 11.32.23 EULA before the container is downloaded
   or run: https://artifacts.tools.arm.com/avh/11.32.23/license_agreement.txt
   Earlier user approval applied to the helper repair, not this license.
3. Re-run the Model 2 build/run/comparison instructions in
   results/fvp-integration-20261016/README.md. Model 1 remains deferred until
   actual Model 2 execution succeeds.

178 original artifact/result files were SHA256-verified unchanged. No source-code
changes or new tests were needed for this system repair. Branch remains
feature/tinyeegnet-training. No commit or push.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Campire ("Camera + Umpire") is an undergraduate AI&ML major project: a low-cost, four-phone system that gives umpires in amateur cricket decision support for WIDE, NO BALL and BOWLER FOUL. It assists the umpire rather than replacing them, so outputs should be presented as decision support, not rulings.

Two documents describe the project, and neither matches the code exactly:

- `appQ/MGdocs/report.docx` is the design report for the full intended system. It is git-ignored, so it exists only locally.
- `appQ/architecture.md` is the original scaffold spec. It is out of date (phone controllers, six remotes, `CommandChannel`), so trust the code over it.

## Repository layout

Everything is in one git repo, `Muraliganesh19/Campire`:

- **`pose/`, `calibration/`, `download/`:** Python experiments.
- **`simulation/`:** test data for the app, documented in the README. Only `videomaker.py` is tracked; the data is git-ignored.
- **`appQ/`:** the Flutter app. Run `flutter` commands from here. It was merged in from `Muraliganesh19/app4CAMPIRE` with its history and the `v1` and `v2` tags, rewritten so plain `git log -- appQ/...` works. Commit IDs therefore differ from `app4CAMPIRE`.

`appQ/.agents/` holds vendored Stitch agent plugins, not app code. The root `MGdocs/` is git-ignored and holds `campire.apk`, `recovered_from_apk/`, `builds/` and `appQ-repo-backup.git` (the old `appQ/.git`), so nothing there is backed up by a push.

## Project status

The AI stages are still in development. Until they exist, the app is tested with simulated results: a Python script, `videomaker.py`, turns hand-annotated data into videos, and the decisions come from a JSON file. The original script was lost along with other work done after the last pushes (`Campire` 2026-07-20, `app4CAMPIRE` 2026-08-21), so `simulation/videomaker.py` is a rewrite.

`simulation/` holds that data, and the README documents its layout: each delivery in `ip_vid/` is a `type*` folder with four camera clips (`c1` to `c4`, each with `images/` and `annotations.xml`) and one `decisions.json`. Clips of a delivery differ in frame count and resolution, so they don't line up frame by frame. `videomaker.py` draws the annotations onto the frames, rendering the four clips in parallel, and writes `op_vid/<type>/c1.mp4` to `c4.mp4` plus `decisions.json`, assuming 30 fps. ANALYZE does this live, in about 25 seconds per delivery. `type1/decisions.json` is currently empty, so `type1` shows `NIL`.

`MGdocs/campire.apk` is a debug build from the author's phone, and its `kernel_blob.bin` embeds the Dart source. `MGdocs/recovered_from_apk/` holds what `extract_kernel_sources.py` recovered, and the two files newer than GitHub, plus the app icons, have been applied to `appQ/`. Dart files nothing imports never reach the APK, so their absence there doesn't mean they were deleted. The APK is 114 MB, over GitHub's 100 MB file limit, so don't commit it.

## Report vs. code

Check this table before assuming a component exists:

| Report stage | Status here |
|---|---|
| Four-camera capture | Implemented in `appQ`, but as clips recorded on each phone and uploaded after stopping. The report's live OpenCV streaming exists only as the two-camera IP Webcam prototype in `calibration/test2.py`. |
| MediaPipe bowler pose | Offline scripts in `pose/`, using the legacy `mp.solutions.pose` API. Not connected to the app. |
| YOLOv8 and RT-DETR detection | In development: no code, weights or training config here. |
| TrackNet ball tracking | In development: nothing here. |
| Event detection and decision logic | In development: nothing here. |
| Result display | The host screen: four video panels plus decision tiles. The phone app is only the camera remote and shows no results. |

The app's only link to any analysis is the script described under "Analysis hand-off". The report's result screenshots show a single UPLOAD button and decision colours that the code doesn't have, so don't use them as a UI spec.

## Commands

Flutter, run from `appQ/`:

```bash
flutter pub get
flutter test                                            # all tests; no device needed
flutter test test/network/clock_sync_service_test.dart  # one file
flutter test --plain-name "computeAverageOffset"        # by test name
flutter analyze                                         # baseline: 1 warning, 26 infos
flutter run -d linux                                    # host on Linux
flutter run -d windows                                  # host (Windows machine only)
flutter run -d <android-device-id>                      # remote camera
flutter build apk --release
flutter build linux --release
flutter build windows --release                         # Windows machine only
```

- Linux builds need `libmpv-dev` and Linux runs need `libmpv2`.
- After adding a plugin, check that it appears in `linux/flutter/generated_plugins.cmake`, and run `flutter pub get` again if not. Without `media_kit_libs_linux` in that list, the build still succeeds but the video plugin compiles as a stub with blank panels. To confirm, run `ldd build/linux/x64/release/bundle/lib/libmedia_kit_video_plugin.so | grep mpv`.
- `.github/workflows/build-windows.yml` builds the Windows app on GitHub Actions for pushes to `main` that touch `appQ/`. It has not been run yet.
- `appQ/requirements.txt` is a hand-written list of Dart packages. `pubspec.yaml` is the source of truth.

Python runs from `.venv` at the repo root, which is git-ignored and uses Python 3.11 from `uv` (`~/.local/bin/uv`). The scripts have no tests and no command-line arguments. Inputs are constants at the top of each script, mostly hardcoded to `E:\campire\...` or LAN IPs, so edit them before running:

```bash
source .venv/bin/activate
python pose/batch_process.py  # pose/input_vid/{legal,illegal}/ to pose/output/{csv,pose_video}/<class>/
python pose/analysis.py       # plots and statistics for one CSV (set CSV_PATH)
python calibration/test2.py   # two-phone IP Webcam viewer (set URL_A, URL_B)
```

To recreate the environment:

```bash
uv venv --python 3.11 --seed .venv
UV_HTTP_TIMEOUT=900 UV_CONCURRENT_DOWNLOADS=2 uv pip install --python .venv/bin/python -r requirements.txt
```

- The long timeout matters: the default 30 seconds fails on the large NVIDIA packages on a slow connection.
- `mediapipe` must stay below 0.10.30, because later releases removed the `mp.solutions` API that `pose/` uses. That version also holds `numpy` below 2.
- `torch` uses the `cu132` build to match the CUDA 13.2 NVIDIA driver on this machine's GTX 1650. Newer CUDA builds won't suit this driver.

## App architecture

### Roles and ownership

`role_select_screen.dart` picks the role by platform: Android offers only the remote, Windows and Linux only the host, and other platforms both. The host screen is still named `WindowsHostScreen`. It and `RemoteHomeScreen` each construct their own services and wire the callbacks themselves, so there is no service layer between them and a protocol change usually touches both screens plus `models/sync_message.dart`. Of the Provider classes, only `DeviceListProvider` holds state in use.

Every message is a `SyncMessage` with JSON fields `type`, `senderDeviceId`, `timestampMicros` and `payload`. Timestamps are integer microseconds, and unknown types throw on purpose. Ports come from `NetworkConstants` (45010 UDP, 45011 TCP), not the unused values in `AppConstants`.

### Session flow

1. **SCAN (host):** starts `HostServerService` (shelf on 45011, with `/ws` and `/upload`) and UDP discovery broadcasts to the assumed `/24` subnet.
2. **Connect (remote):** opening the screen pre-warms the camera so that recording later only has to flip `startVideoRecording()`. Discovery only fills the host's device list. The real connection happens when the operator types the host's IP on the phone, and the host screen does not display that IP. `NodeClientService` then opens the WebSocket and sends `handshake_request`; the host registers a device from the first message on a socket.
3. **Clock sync (host, automatic on connect):** eight NTP-style samples over the WebSocket, dropping the three with the highest round-trip time. The device then shows as `synced`.
4. **START:** the host broadcasts `start_at` with a target 700 ms ahead (`syncStartMargin`). Each remote sets a `Timer`, starts recording and replies with `ack` carrying target and actual times.
5. **STOP:** the host broadcasts `stop`. Each remote saves its clip to the `Campire` gallery album, then POSTs it to `/upload` with a SHA-256 in `x-file-hash`. The host verifies the hash and stores the file in `<systemTemp>/campire_uploads`.

### Analysis hand-off

Paths come from `core/host_paths.dart`. Session files live in `Input` and `Output` under `E:\campire` on Windows and `~/campire` elsewhere, or `CAMPIRE_BASE_DIR` if set. Both folders are wiped on every new upload.

- **UPLOAD VIDEOS (live session):** the four oldest uploaded clips by modification time become `liveip<N>_cam<1-4>.mp4`, with a `liveip<N>.json`. ANALYZE then runs no analysis: it copies them to `liveop<N>_*` in Output and shows the JSON's `NIL` decisions.
- **UPLOAD FILES (file session):** what happens depends on the chosen folder.
  - A delivery from `simulation/ip_vid`, recognised by its `c1` to `c4` subfolders, isn't copied. ANALYZE runs `videomaker.py process <folder> <session>`, which renders into `op_vid/<type>` and copies the videos to Output.
  - From any other folder, up to four `.mp4`, `.avi` or `.mov` files sorted by name, their same-named `.xml` sidecars and any `.json` become `fileip<N>_*`. Videos keep their contents but always get a `.mp4` name. ANALYZE runs the script with no arguments, and it copies them to Output.

Both use `HostPaths.videomakerScript` (override with `CAMPIRE_VIDEOMAKER`), run with the system `python3`. `videomaker.py` reruns itself under `.venv` when it needs OpenCV, so the no-argument path must stay standard-library only. The script must write `fileip<N>_cam<1-4>.mp4` to Output, and may write `fileip<N>.json` there; otherwise the app reads the copy in Input.

The decision JSON has the keys `session_id`, `bowler_stance`, `delivery` and `wicket`, defaulting to `"NIL"`. An empty JSON file makes ANALYZE stop before the videos play. Tiles colour only exact values: green for `LEGAL`, `VALID`, `NOT-HIT`; red for `ILLEGAL`, `INVALID`, `HIT`, `WIDE`, `NOBALL`, `NO-BALL`; grey for `NIL`.

## Known issues

- **Clock offsets are never applied.** The host stores each offset in its own `ClockSyncService.offsets`, but `RecordingManager.handleStartAt` runs on the remote and reads the remote's empty map, so the offset is always 0. Start times therefore depend on the phones' wall clocks, and `synced` only means the handshake finished. Fixing this means sending the offset to the remote or having the host adjust a per-device target.
- **The Linux host draws 1080x1920 clips smaller in their panels** than 720x1280 ones. `videomaker.py` avoids this by capping clips at 1280 pixels, but live clips from phones aren't resized.
- **`cam<N>` is not a stable camera identity.** It comes from upload order, so the same phone can land in a different slot each session.
- **`pose/pose_extractor.py` drops the frames it should keep.** `writer.write` is called only in the no-pose branch, so `_pose.mp4` files contain only frames with no detected pose. Detected rows also never set `pose_detected: True`, so the `== True` filter in `analysis.py` removes every row whenever a clip has at least one no-pose frame.
- **`download/gdown.py` can't run under that name.** It shadows the `gdown` package and fails with a circular-import `AttributeError`.
- **Tests cover pure logic only:** models, clock math and discovery messages. Nothing exercises the host and remote together, and `recording_manager_test.dart` has an empty TODO test that passes, so timing changes need checking on real phones.
- **Unused code:**
  - `network/command_channel.dart`
  - `services/mock_pipeline_service.dart` and `tool/setup_mock_pipeline.dart`
  - `SessionStateProvider` and `RecordingSession`
  - `WakelockHelper` and `BatteryOptimizationHelper`, so nothing keeps the screen awake while recording
  - `udpClockSyncPort`

# Multi-Device Sync Camera App — Flutter Project Spec

## Purpose

This is a starting architecture spec for an AI coding tool to scaffold from. It defines the project structure, module responsibilities, and key architectural decisions — not the full implementation.

## Goals (recap)

- One app, two roles: **Controller** (triggers shutter on remotes, optional tiled preview in v2) and **Remote** (records on controller's mark)
- Up to 6 remote devices
- Recording start synchronized within 0.35s across all devices
- WiFi-only (local network / hotspot), no internet dependency
- Runs on Android devices up to ~3 years old across different SoCs
- Priority: stability and consistency over features
- Resilient to thermal throttling and frame drops

## Architectural Decisions (assumptions — flag if you want these changed)

1. **Single codebase, dual role** — one app where the user picks "Controller" or "Remote" on launch, rather than two separate apps. Easier to maintain and test as a beginner, and any phone can serve either role.
2. **State management: `provider`** — simpler learning curve than Riverpod/Bloc for a first app; swap later if needed.
3. **Networking: raw `dart:io` sockets** (`RawDatagramSocket` for UDP discovery, `Socket`/`WebSocket` for the command channel) rather than a discovery plugin — needed for direct control over latency and timing.
4. **Camera: pre-warmed session** — camera is opened and held in a ready/previewing state on remotes ahead of time; the shutter trigger only flips `startVideoRecording()`, not the full camera init, to minimize per-device startup variance.
5. **Sync protocol is scheduled-start based** (controller tells each remote to start at a future absolute timestamp, adjusted for that device's clock offset) rather than "start now on receipt." Full message schema is a separate spec — ask for it next if you want it before handing this off.

## Suggested `pubspec.yaml` dependencies

```yaml
dependencies:
  flutter:
    sdk: flutter
  camera: ^0.11.0          # check latest stable at build time
  provider: ^6.1.0
  wakelock_plus: ^1.2.0    # keep screen awake during recording
  permission_handler: ^11.3.0
  device_info_plus: ^10.1.0
  path_provider: ^2.1.0    # for saved video file locations
  intl: ^0.19.0            # timestamp formatting/logging

dev_dependencies:
  flutter_test:
    sdk: flutter
  mocktail: ^1.0.0         # for mocking network/camera in unit tests
```

## Directory Structure

```
lib/
├── main.dart                          # app entry point, sets up providers
├── app.dart                           # MaterialApp + role-based routing
│
├── core/
│   ├── constants.dart                 # ports, timeouts, sync margins, video config
│   └── logger.dart                    # simple timestamped logger (crucial for debugging sync)
│
├── models/
│   ├── device_info.dart               # id, name, IP, role, connection status
│   ├── sync_message.dart              # message types: discover, handshake, start_at, ack
│   └── recording_session.dart         # session id, start time, participating devices
│
├── network/
│   ├── discovery_service.dart         # UDP broadcast (controller) / listen (remote)
│   ├── clock_sync_service.dart        # round-trip handshake → per-device clock offset
│   ├── command_channel.dart           # TCP/WebSocket connection per remote device
│   └── network_constants.dart         # port numbers, discovery broadcast address
│
├── camera/
│   ├── camera_controller_service.dart # wraps `camera` plugin; open/hold/release lifecycle
│   └── recording_manager.dart         # schedules local Timer to hit target timestamp,
│                                       # triggers start/stop, tracks actual vs. target time
│
├── state/
│   ├── app_role_provider.dart         # controller vs. remote, selected once at launch
│   ├── device_list_provider.dart      # controller's view of discovered/connected remotes
│   └── session_state_provider.dart    # current recording session status
│
├── ui/
│   ├── role_select_screen.dart        # first screen: "Controller" or "Remote camera"
│   ├── controller/
│   │   ├── controller_home_screen.dart
│   │   ├── device_list_tile.dart      # per-remote status: connected, synced, recording
│   │   └── preview_grid.dart          # [v2] tiled live preview thumbnails
│   ├── remote/
│   │   ├── remote_home_screen.dart    # shows pairing status, camera preview, big status text
│   │   └── remote_status_indicator.dart
│   └── shared/
│       └── connection_status_badge.dart
│
└── utils/
    ├── permissions.dart                # camera/mic/storage permission requests
    ├── wakelock_helper.dart            # keep screen on during active session
    └── battery_optimization_helper.dart # prompt user to exempt app from battery optimization

test/
├── network/
│   ├── clock_sync_service_test.dart
│   └── discovery_service_test.dart
├── camera/
│   └── recording_manager_test.dart
└── models/
    └── sync_message_test.dart
```

## Module responsibility notes

| Module | Responsibility | Why it's separate |
|---|---|---|
| `network/discovery_service.dart` | Finds devices on the LAN | Isolated so it can be tested without real sockets |
| `network/clock_sync_service.dart` | Computes offset between controller clock and each remote's clock | This accuracy directly determines your sync window — keep it isolated and unit-testable |
| `network/command_channel.dart` | Reliable message delivery per device | Separate from discovery so connection drops/retries don't affect device-finding logic |
| `camera/camera_controller_service.dart` | Camera lifecycle only (open, hold, release) | Keeps camera plugin quirks contained to one file |
| `camera/recording_manager.dart` | "Start at time T" scheduling and execution | This is the module most worth hand-tuning per SoC later — keep its interface clean |
| `models/` | Plain Dart classes, no Flutter/widget imports | Keeps them fast to unit test in isolation |

## Testing priority

For a stability-focused app, prioritize unit tests on `clock_sync_service`, `recording_manager`, and the `sync_message` serialization — these are where timing bugs will actually hide. UI can stay lightly tested for v1.

## Suggested build order against this structure

1. `models/` + `network/discovery_service.dart` — get two devices finding each other
2. `network/clock_sync_service.dart` — verify offset calculation against a known delay
3. `camera/camera_controller_service.dart` — pre-warm camera, confirm hold state works across a couple of test devices
4. `camera/recording_manager.dart` + `network/command_channel.dart` — wire up the actual scheduled trigger
5. UI screens last, once the core sync loop is proven on 2 devices
6. Scale to 6 devices, then add `preview_grid.dart` (v2)

## Not covered here

The exact JSON schema for `sync_message.dart` (discovery/handshake/start_at/ack payloads) — happy to spec that out next if useful before you hand this off.

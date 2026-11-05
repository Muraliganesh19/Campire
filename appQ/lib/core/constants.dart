import 'package:camera/camera.dart';

/// Global configuration constants for the app.
class AppConstants {
  // Ports
  static const int udpDiscoveryPort = 4545;
  static const int tcpCommandPort = 4546;

  // Timeouts
  static const Duration discoveryTimeout = Duration(seconds: 10);
  static const Duration connectionTimeout = Duration(seconds: 5);
  static const Duration clockSyncTimeout = Duration(seconds: 3);

  // Sync margins
  /// Maximum accepted clock variance (e.g. 0.35s / 350ms)
  static const Duration maxSyncWindow = Duration(milliseconds: 350);

  // ── Video / Camera Config ───────────────────────────────────────────────

  /// Target resolution preset passed to [CameraController].
  ///
  /// [ResolutionPreset.high] typically maps to 1080p on most devices.
  /// CameraX picks the closest supported match — actual achieved resolution
  /// varies by device and should be verified from the post-init log output
  /// or from the recorded video file properties.
  static const ResolutionPreset targetResolutionPreset = ResolutionPreset.veryHigh;

  /// Target frames-per-second for video recording.
  ///
  /// 60 fps is a good balance of motion capture quality and broad device
  /// compatibility. Change this between test sessions as needed.
  /// If a device rejects this value, the camera service falls back to the
  /// platform default fps automatically.
  static const int targetFps = 60;

  static const String videoExtension = 'mp4';

  // Sync Recording
  /// Delay between shutter press and scheduled recording start.
  /// Generous on purpose — better to wait a bit too long than miss the
  /// window for a remote to receive and process the start_at command.
  static const Duration syncStartMargin = Duration(milliseconds: 700);
}

import 'package:permission_handler/permission_handler.dart';
import 'package:campire/core/logger.dart';

/// Helper to check and request Android battery optimization exemption.
///
/// Prevents the OS from killing or throttling the background sync socket
/// and camera session during recording.
class BatteryOptimizationHelper {

  /// Checks if the app is already exempted from battery optimization.
  ///
  /// Returns `true` if the app is whitelisted, `false` otherwise.
  /// On non-Android platforms this always returns `true` (no restriction).
  static Future<bool> isExempted() async {
    try {
      final status = await Permission.ignoreBatteryOptimizations.isGranted;
      return status;
    } catch (e) {
      // Platform may not support this permission — treat as exempted.
      SyncLogger.error(
        'BatteryOptimizationHelper: failed to check exemption',
        e,
      );
      return true;
    }
  }

  /// Prompts the user to exempt the app from battery optimization.
  ///
  /// On Android this uses `permission_handler`'s
  /// [Permission.ignoreBatteryOptimizations] which triggers the system
  /// dialog for `REQUEST_IGNORE_BATTERY_OPTIMIZATIONS`.
  ///
  /// On other platforms this is a no-op.
  static Future<bool> requestExemption() async {
    try {
      final status = await Permission.ignoreBatteryOptimizations.request();
      SyncLogger.info(
        'BatteryOptimizationHelper: exemption status = $status',
      );
      return status.isGranted;
    } catch (e) {
      SyncLogger.error(
        'BatteryOptimizationHelper: failed to request exemption',
        e,
      );
      // TODO: If permission_handler's ignoreBatteryOptimizations doesn't
      // work reliably on the target device, an alternative is to launch
      // the system settings intent directly via a platform channel:
      //   Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
      //          Uri.parse("package:$packageName"))
      // This requires a small MethodChannel or the `android_intent_plus`
      // package and should be added if the permission_handler path fails
      // in testing on specific OEMs.
      return false;
    }
  }
}

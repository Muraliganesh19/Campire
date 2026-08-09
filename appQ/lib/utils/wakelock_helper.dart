import 'package:wakelock_plus/wakelock_plus.dart';
import 'package:campire/core/logger.dart';

/// Helper to manage the screen awake / wakelock state.
///
/// Enable when entering an active session (post role-select) so the screen
/// stays on during discovery, sync, and recording. Disable on exit.
class WakelockHelper {

  /// Enables the wakelock — screen will stay on.
  static Future<void> enable() async {
    try {
      await WakelockPlus.enable();
      SyncLogger.info('WakelockHelper: wakelock enabled');
    } catch (e) {
      // Platform exceptions on unsupported platforms are non-fatal.
      SyncLogger.error('WakelockHelper: failed to enable wakelock', e);
    }
  }

  /// Disables the wakelock — screen can sleep normally.
  static Future<void> disable() async {
    try {
      await WakelockPlus.disable();
      SyncLogger.info('WakelockHelper: wakelock disabled');
    } catch (e) {
      SyncLogger.error('WakelockHelper: failed to disable wakelock', e);
    }
  }
}

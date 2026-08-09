import 'dart:async';

import 'package:campire/camera/camera_controller_service.dart';
import 'package:campire/core/constants.dart';
import 'package:campire/core/logger.dart';
import 'package:campire/models/device_info.dart';
import 'package:campire/models/sync_message.dart';
import 'package:campire/network/clock_sync_service.dart';
import 'package:campire/network/command_channel.dart';

/// Orchestrates synchronised video recording across controller and remotes.
///
/// ## Controller side
///
/// On shutter press: computes a target timestamp in the future, sends a
/// `start_at` message to every connected+synced remote, and collects `ack`
/// responses (each carrying the remote's actual start timestamp for sync
/// accuracy analysis). Stop is broadcast immediately.
///
/// ## Remote side
///
/// On receiving `start_at`: converts the controller's target timestamp into
/// local time using the cached clock offset, schedules a [Timer], and when
/// the timer fires calls [CameraControllerService.startRecording]. Sends
/// an `ack` back with the actual local start timestamp.
class RecordingManager {
  final String localDeviceId;

  Timer? _scheduleTimer;
  final CameraControllerService? _cameraService; // Used only on remote

  // ignore: prefer_initializing_formals
  RecordingManager({
    CameraControllerService? cameraService,
    required this.localDeviceId,
  }) : _cameraService = cameraService;

  // ═══════════════════════════════════════════════════════════════════════
  // Controller side
  // ═══════════════════════════════════════════════════════════════════════

  /// Triggers a synchronised recording start on all connected remotes.
  ///
  /// 1. Computes `target = now + syncStartMargin` (700 ms by default).
  /// 2. Sends a `start_at` message to each remote via its [CommandChannel].
  /// 3. Starts recording locally at the same target time.
  /// 4. Returns the target timestamp (microseconds) for logging / UI.
  ///
  /// The caller should subscribe to each channel's [CommandChannel.messages]
  /// for incoming `ack` messages — this method does not block waiting for
  /// them.
  Future<int> triggerSyncRecording({
    required Iterable<DeviceInfo> syncedDevices,
    required Map<String, CommandChannel> channels,
  }) async {
    final now = DateTime.now();
    final target = now.add(AppConstants.syncStartMargin);
    final targetMicros = target.microsecondsSinceEpoch;

    SyncLogger.sync(
      'RecordingManager [controller]: scheduling start_at '
      '$targetMicros (in ${AppConstants.syncStartMargin.inMilliseconds} ms)',
    );

    // Build the start_at message.
    final startMessage = SyncMessage(
      type: SyncMessageType.startAt,
      senderDeviceId: localDeviceId,
      timestampMicros: DateTime.now().microsecondsSinceEpoch,
      payload: {
        'targetStartMicros': targetMicros,
      },
    );

    final futures = <Future<void>>[];
    for (final device in syncedDevices) {
      final channel = channels[device.id];
      if (channel == null) continue;

      if (device.isRecording) {
        SyncLogger.info('RecordingManager [controller]: skipping ${device.id} for group trigger — already recording');
        continue;
      }

      futures.add(channel.send(startMessage));
    }

    await Future.wait(futures);

    return targetMicros;
  }

  /// Sends a `start_now` message to a single remote channel.
  Future<void> triggerSingleRecording(DeviceInfo device, CommandChannel channel) async {
    SyncLogger.sync('RecordingManager [controller]: sending start_now to ${device.id}');
    final startNowMsg = SyncMessage(
      type: SyncMessageType.startNow,
      senderDeviceId: localDeviceId,
      timestampMicros: DateTime.now().microsecondsSinceEpoch,
    );
    await channel.send(startNowMsg);
  }

  /// Sends a `stop` message to a single remote channel.
  Future<void> triggerSingleStop(DeviceInfo device, CommandChannel channel) async {
    SyncLogger.sync('RecordingManager [controller]: sending stop to ${device.id}');
    final stopMsg = SyncMessage(
      type: SyncMessageType.stop,
      senderDeviceId: localDeviceId,
      timestampMicros: DateTime.now().microsecondsSinceEpoch,
    );
    await channel.send(stopMsg);
  }

  /// Broadcasts a `stop` message to all connected remotes and stops the
  /// local recording immediately. No scheduled timing for stop — only
  /// start-time sync matters per the spec.
  Future<void> triggerStop({
    required List<CommandChannel> remoteChannels,
  }) async {
    SyncLogger.sync('RecordingManager [controller]: broadcasting stop');

    final stopMessage = SyncMessage(
      type: SyncMessageType.stop,
      senderDeviceId: localDeviceId,
      timestampMicros: DateTime.now().microsecondsSinceEpoch,
    );

    await Future.wait(
      remoteChannels.map((ch) => ch.send(stopMessage)),
    );
  }

  /// Processes an incoming `ack` message from a remote.
  ///
  /// Logs the sync accuracy — the difference between the intended target
  /// start time and the remote's actual start time.
  void handleAck(SyncMessage ack) {
    final remoteActualMicros = ack.payload['actualStartMicros'] as int?;
    final targetMicros = ack.payload['targetStartMicros'] as int?;

    if (remoteActualMicros != null && targetMicros != null) {
      final diffMs = (remoteActualMicros - targetMicros) / 1000.0;
      SyncLogger.sync(
        'RecordingManager [controller]: ack from ${ack.senderDeviceId} — '
        'actual start offset: ${diffMs.toStringAsFixed(2)} ms '
        '(positive = late, negative = early)',
      );
    } else {
      SyncLogger.info(
        'RecordingManager [controller]: ack from ${ack.senderDeviceId} '
        '(no timing data)',
      );
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Remote side
  // ═══════════════════════════════════════════════════════════════════════

  /// Handles a `start_at` message from the controller.
  ///
  /// Converts the controller's target timestamp into local time using the
  /// cached clock offset from [clockSyncService], schedules a [Timer] for
  /// the remaining duration, starts recording when it fires, and sends an
  /// `ack` back.
  void handleStartAt({
    required SyncMessage message,
    required ClockSyncService clockSyncService,
    required CommandChannel controllerChannel,
  }) {
    final targetMicros = message.payload['targetStartMicros'] as int;

    // The controller's targetMicros is in the CONTROLLER's clock.
    // Our cached offset = remote_clock - controller_clock.
    // So local_target = targetMicros + offset.
    final offset = clockSyncService.offsets[message.senderDeviceId] ?? 0;
    final localTargetMicros = targetMicros + offset;

    final nowMicros = DateTime.now().microsecondsSinceEpoch;
    final delayMicros = localTargetMicros - nowMicros;

    SyncLogger.sync(
      'RecordingManager [remote]: start_at received — '
      'controllerTarget=$targetMicros, offset=$offset µs, '
      'localTarget=$localTargetMicros, delay=${delayMicros / 1000} ms',
    );

    if (delayMicros <= 0) {
      // Target is in the past — start immediately.
      SyncLogger.sync(
        'RecordingManager [remote]: target already passed — starting now',
      );
      _startRecordingAndAck(
        targetMicros: targetMicros,
        controllerChannel: controllerChannel,
      );
    } else {
      _scheduleTimer?.cancel();
      _scheduleTimer = Timer(
        Duration(microseconds: delayMicros),
        () => _startRecordingAndAck(
          targetMicros: targetMicros,
          controllerChannel: controllerChannel,
        ),
      );
    }
  }

  /// Handles a `stop` message from the controller.
  ///
  /// Stops recording immediately — no scheduling needed for stop.
  Future<void> handleStop() async {
    SyncLogger.sync('RecordingManager [remote]: stop received');
    cancelScheduledStart();
    await _cameraService?.stopRecording();
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Internal helpers
  // ═══════════════════════════════════════════════════════════════════════

  /// Starts recording and sends an `ack` back to the controller.
  Future<void> _startRecordingAndAck({
    required int targetMicros,
    required CommandChannel controllerChannel,
  }) async {
    final actualStartMicros = DateTime.now().microsecondsSinceEpoch;
    await _cameraService?.startRecording();

    SyncLogger.sync(
      'RecordingManager [remote]: recording started at $actualStartMicros '
      '(target $targetMicros, diff '
      '${(actualStartMicros - targetMicros) / 1000} ms)',
    );

    // Send ack with actual start timestamp so the controller can
    // log/compare the real spread across devices.
    final ack = SyncMessage(
      type: SyncMessageType.ack,
      senderDeviceId: localDeviceId,
      timestampMicros: actualStartMicros,
      payload: {
        'targetStartMicros': targetMicros,
        'actualStartMicros': actualStartMicros,
      },
    );

    await controllerChannel.send(ack);
  }

  /// Cancels any pending scheduled start timer.
  void cancelScheduledStart() {
    _scheduleTimer?.cancel();
    _scheduleTimer = null;
  }
}

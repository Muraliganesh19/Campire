import 'dart:async';

import 'package:campire/camera/camera_controller_service.dart';
import 'package:campire/core/constants.dart';
import 'package:campire/core/logger.dart';
import 'package:campire/models/sync_message.dart';
import 'package:campire/network/clock_sync_service.dart';
import 'package:campire/network/host_server_service.dart';
import 'package:camera/camera.dart';

class RecordingManager {
  final String localDeviceId;

  Timer? _scheduleTimer;
  final CameraControllerService? _cameraService;

  // ignore: prefer_initializing_formals
  RecordingManager({
    CameraControllerService? cameraService,
    required this.localDeviceId,
  }) : _cameraService = cameraService;

  // ═══════════════════════════════════════════════════════════════════════
  // Controller side
  // ═══════════════════════════════════════════════════════════════════════

  Future<int> triggerSyncRecording({
    required Iterable<String> connectedDeviceIds,
    required HostServerService hostServerService,
  }) async {
    final now = DateTime.now();
    final target = now.add(AppConstants.syncStartMargin);
    final targetMicros = target.microsecondsSinceEpoch;

    SyncLogger.sync(
      'RecordingManager [controller]: scheduling start_at '
      '$targetMicros (in ${AppConstants.syncStartMargin.inMilliseconds} ms)',
    );

    final startMessage = SyncMessage(
      type: SyncMessageType.startAt,
      senderDeviceId: localDeviceId,
      timestampMicros: DateTime.now().microsecondsSinceEpoch,
      payload: {'targetStartMicros': targetMicros},
    );

    await hostServerService.broadcast(startMessage);
    return targetMicros;
  }

  Future<void> triggerStop({
    required HostServerService hostServerService,
  }) async {
    SyncLogger.sync('RecordingManager [controller]: broadcasting stop');

    final stopMessage = SyncMessage(
      type: SyncMessageType.stop,
      senderDeviceId: localDeviceId,
      timestampMicros: DateTime.now().microsecondsSinceEpoch,
    );

    await hostServerService.broadcast(stopMessage);
  }

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

  void handleStartAt({
    required SyncMessage message,
    required ClockSyncService clockSyncService,
    required void Function(SyncMessage) sendAck,
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
      SyncLogger.sync(
        'RecordingManager [remote]: target already passed — starting now',
      );
      _startRecordingAndAck(targetMicros: targetMicros, sendAck: sendAck);
    } else {
      _scheduleTimer?.cancel();
      _scheduleTimer = Timer(
        Duration(microseconds: delayMicros),
        () =>
            _startRecordingAndAck(targetMicros: targetMicros, sendAck: sendAck),
      );
    }
  }

  Future<XFile?> handleStop() async {
    SyncLogger.sync('RecordingManager [remote]: stop received');
    cancelScheduledStart();
    return await _cameraService?.stopRecording();
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Internal helpers
  // ═══════════════════════════════════════════════════════════════════════

  Future<void> _startRecordingAndAck({
    required int targetMicros,
    required void Function(SyncMessage) sendAck,
  }) async {
    final actualStartMicros = DateTime.now().microsecondsSinceEpoch;
    await _cameraService?.startRecording();

    SyncLogger.sync(
      'RecordingManager [remote]: recording started at $actualStartMicros '
      '(target $targetMicros, diff '
      '${(actualStartMicros - targetMicros) / 1000} ms)',
    );

    final ack = SyncMessage(
      type: SyncMessageType.ack,
      senderDeviceId: localDeviceId,
      timestampMicros: actualStartMicros,
      payload: {
        'targetStartMicros': targetMicros,
        'actualStartMicros': actualStartMicros,
      },
    );

    sendAck(ack);
  }

  void cancelScheduledStart() {
    _scheduleTimer?.cancel();
    _scheduleTimer = null;
  }
}

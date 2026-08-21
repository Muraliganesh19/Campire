import 'dart:async';

import 'package:campire/core/logger.dart';
import 'package:campire/models/device_info.dart';
import 'package:campire/models/sync_message.dart';
import 'package:campire/network/network_constants.dart';

// ═══════════════════════════════════════════════════════════════════════════
// Pure offset math — fully testable with no I/O dependencies
// ═══════════════════════════════════════════════════════════════════════════

class ClockSyncSample {
  final int t1; // controller sends request   (controller clock)
  final int t2; // remote receives request    (remote clock)
  final int t3; // remote sends response      (remote clock)
  final int t4; // controller receives response (controller clock)

  const ClockSyncSample({
    required this.t1,
    required this.t2,
    required this.t3,
    required this.t4,
  });

  int get offsetMicros => ((t2 - t1) + (t3 - t4)) ~/ 2;
  int get roundTripMicros => (t4 - t1) - (t3 - t2);

  @override
  String toString() =>
      'ClockSyncSample(t1: $t1, t2: $t2, t3: $t3, t4: $t4, '
      'offset: $offsetMicros µs, rtt: $roundTripMicros µs)';
}

int computeAverageOffset(
  List<ClockSyncSample> samples, {
  int discardCount = NetworkConstants.clockSyncDiscardCount,
}) {
  if (samples.length <= discardCount) {
    throw ArgumentError(
      'Need more than $discardCount samples to compute offset, '
      'got ${samples.length}',
    );
  }

  final sorted = List<ClockSyncSample>.of(samples)
    ..sort((a, b) => a.roundTripMicros.compareTo(b.roundTripMicros));

  final kept = sorted.sublist(0, sorted.length - discardCount);

  final totalOffset = kept.fold<int>(0, (sum, s) => sum + s.offsetMicros);
  return totalOffset ~/ kept.length;
}

// ═══════════════════════════════════════════════════════════════════════════
// Network-facing service (controller + remote roles via WebSocket)
// ═══════════════════════════════════════════════════════════════════════════

class ClockSyncService {
  ClockSyncService({required this.localDeviceId});

  final String localDeviceId;

  final Map<String, int> offsets = {};

  final Map<String, Completer<ClockSyncSample>> _activeCompleters = {};
  final Map<String, int> _activeT1s = {};

  // ── Controller: compute offset via WebSocket round-trip ────────────────

  Future<int> computeOffset(
    DeviceInfo remote,
    Future<void> Function(SyncMessage) sendMessage,
  ) async {
    final samples = <ClockSyncSample>[];
    const sampleCount = NetworkConstants.clockSyncSampleCount;
    const timeout = NetworkConstants.clockSyncSampleTimeout;
    const gap = NetworkConstants.clockSyncSampleGap;

    for (var i = 0; i < sampleCount; i++) {
      try {
        final t1 = DateTime.now().microsecondsSinceEpoch;

        _activeT1s[remote.id] = t1;
        _activeCompleters[remote.id] = Completer<ClockSyncSample>();

        final request = SyncMessage(
          type: SyncMessageType.clockSyncRequest,
          senderDeviceId: localDeviceId,
          timestampMicros: t1,
          payload: {'t1': t1},
        );

        await sendMessage(request);

        final sample = await _activeCompleters[remote.id]!.future.timeout(timeout);

        samples.add(sample);
        SyncLogger.sync(
          'Clock sync sample ${i + 1}/$sampleCount to ${remote.id}: $sample',
        );
      } on TimeoutException {
        SyncLogger.info(
          'Clock sync sample ${i + 1}/$sampleCount to ${remote.id}: timed out — skipping',
        );
      } catch (e, stack) {
        SyncLogger.error(
          'Clock sync sample ${i + 1}/$sampleCount to ${remote.id}: failed',
          e,
          stack,
        );
      } finally {
        _activeCompleters.remove(remote.id);
        _activeT1s.remove(remote.id);
      }

      if (i < sampleCount - 1) {
        await Future<void>.delayed(gap);
      }
    }

    if (samples.length <= NetworkConstants.clockSyncDiscardCount) {
      SyncLogger.error(
        'Clock sync to ${remote.id} failed: only ${samples.length} samples collected',
      );
      throw ClockSyncException(
        'Not enough samples for ${remote.id}: ${samples.length}',
      );
    }

    final offset = computeAverageOffset(samples);
    offsets[remote.id] = offset;

    SyncLogger.sync(
      'Clock sync to ${remote.id} complete: offset = $offset µs '
      '(${offset / 1000} ms)',
    );

    return offset;
  }

  void handleResponse(SyncMessage response) {
    if (response.type != SyncMessageType.clockSyncResponse) return;
    
    final remoteId = response.senderDeviceId;
    final activeT1 = _activeT1s[remoteId];
    final t1InResponse = response.payload['t1'] as int?;
    
    if (activeT1 == null || t1InResponse != activeT1) return;

    final t2 = response.payload['t2'] as int;
    final t3 = response.payload['t3'] as int;
    final t4 = DateTime.now().microsecondsSinceEpoch;

    final completer = _activeCompleters[remoteId];
    if (completer != null && !completer.isCompleted) {
      completer.complete(
        ClockSyncSample(t1: activeT1, t2: t2, t3: t3, t4: t4),
      );
    }
  }

  // ── Remote: auto-respond to clock sync requests ────────────────────────

  SyncMessage? handleRequest(SyncMessage request) {
    if (request.type != SyncMessageType.clockSyncRequest) return null;
    
    final t2 = DateTime.now().microsecondsSinceEpoch;
    SyncLogger.sync('Remote: received clock_sync_request from ${request.senderDeviceId}');

    final t3 = DateTime.now().microsecondsSinceEpoch;
    return SyncMessage(
      type: SyncMessageType.clockSyncResponse,
      senderDeviceId: localDeviceId,
      timestampMicros: t3,
      payload: {
        't1': request.payload['t1'],
        't2': t2,
        't3': t3,
      },
    );
  }

  void dispose() {
    offsets.clear();
    _activeCompleters.clear();
    _activeT1s.clear();
  }
}

class ClockSyncException implements Exception {
  final String message;
  const ClockSyncException(this.message);

  @override
  String toString() => 'ClockSyncException: $message';
}

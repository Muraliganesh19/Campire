import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:campire/core/logger.dart';
import 'package:campire/models/device_info.dart';
import 'package:campire/models/sync_message.dart';
import 'package:campire/network/network_constants.dart';

// ═══════════════════════════════════════════════════════════════════════════
// Pure offset math — fully testable with no I/O dependencies
// ═══════════════════════════════════════════════════════════════════════════

/// The four timestamps captured during a single NTP-style round-trip.
///
/// All values are in **microseconds since epoch** (matching
/// [SyncMessage.timestampMicros]).
///
///   Controller              Remote
///   ──────────              ──────
///   T1 ──── request ────▶  T2
///                          T3 ──── response ────▶ T4
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

  /// Clock offset in microseconds.
  ///
  /// Positive → remote clock is ahead of controller.
  /// Negative → remote clock is behind controller.
  ///
  ///   offset = ((T2 - T1) + (T3 - T4)) / 2
  int get offsetMicros => ((t2 - t1) + (t3 - t4)) ~/ 2;

  /// Network round-trip delay in microseconds (excludes remote processing
  /// time).
  ///
  ///   rtt = (T4 - T1) - (T3 - T2)
  int get roundTripMicros => (t4 - t1) - (t3 - t2);

  @override
  String toString() =>
      'ClockSyncSample(t1: $t1, t2: $t2, t3: $t3, t4: $t4, '
      'offset: $offsetMicros µs, rtt: $roundTripMicros µs)';
}

/// Takes a list of [samples], discards the [discardCount] with the highest
/// round-trip delay (most jitter), and returns the arithmetic mean of the
/// remaining offsets.
///
/// Throws [ArgumentError] if there aren't enough samples after discarding.
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

  // Sort by RTT ascending — we'll keep the low-RTT (high-quality) samples.
  final sorted = List<ClockSyncSample>.of(samples)
    ..sort((a, b) => a.roundTripMicros.compareTo(b.roundTripMicros));

  final kept = sorted.sublist(0, sorted.length - discardCount);

  final totalOffset = kept.fold<int>(0, (sum, s) => sum + s.offsetMicros);
  return totalOffset ~/ kept.length;
}

// ═══════════════════════════════════════════════════════════════════════════
// Network-facing service (controller + remote roles)
// ═══════════════════════════════════════════════════════════════════════════

/// Performs NTP-style clock offset measurement between the controller and
/// each remote device over a **dedicated UDP socket** (port
/// [NetworkConstants.udpClockSyncPort]).
///
/// ## Controller side
///
/// Call [computeOffset] with a discovered remote's [DeviceInfo]. The method
/// collects [NetworkConstants.clockSyncSampleCount] round-trip samples,
/// discards the noisiest, averages the rest, caches the result, and returns
/// the offset in microseconds.
///
/// ## Remote side
///
/// Call [startResponder] once after discovery. The remote binds a UDP socket
/// on the clock sync port and automatically replies to every incoming
/// `clock_sync_request` with a `clock_sync_response` carrying its own
/// timestamps. No further action is needed.
///
/// ## Design notes
///
/// * Uses UDP (not TCP) to avoid TCP's handshake/ACK overhead that adds
///   noise to sub-millisecond timing measurements.
/// * Does **not** implement automatic periodic resync — sessions are short
///   enough that a single sync before recording is sufficient.
// TODO: Consider periodic resync for long sessions or if devices show
// thermal-throttling-induced clock drift during recording.
class ClockSyncService {
  ClockSyncService({required this.localDeviceId});

  final String localDeviceId;

  // ── Offset cache ───────────────────────────────────────────────────────

  /// Cached offsets keyed by remote device ID, in microseconds.
  /// Positive = remote clock ahead, negative = remote clock behind.
  ///
  /// Populated by [computeOffset]; looked up later by recording_manager
  /// to adjust scheduled start times.
  final Map<String, int> offsets = {};

  // ── Controller: compute offset via UDP round-trip ──────────────────────

  /// Measures the clock offset to [remote] using
  /// [NetworkConstants.clockSyncSampleCount] NTP-style UDP round-trips.
  ///
  /// Discards the [NetworkConstants.clockSyncDiscardCount] noisiest samples
  /// and averages the rest. Returns the offset in microseconds (positive if
  /// the remote's clock is ahead).
  ///
  /// The result is cached in [offsets] for later lookup.
  Future<int> computeOffset(DeviceInfo remote) async {
    final samples = <ClockSyncSample>[];
    RawDatagramSocket? socket;
    StreamSubscription<RawSocketEvent>? subscription;

    try {
      socket = await RawDatagramSocket.bind(
        InternetAddress.anyIPv4,
        0, // ephemeral port — the remote replies to the source port
        reuseAddress: true,
      );

      final remoteAddress = InternetAddress(remote.ipAddress);
      const port = NetworkConstants.udpClockSyncPort;
      const sampleCount = NetworkConstants.clockSyncSampleCount;
      const timeout = NetworkConstants.clockSyncSampleTimeout;
      const gap = NetworkConstants.clockSyncSampleGap;

      // ── Single persistent listener for the entire handshake ──────────
      // RawDatagramSocket is a single-subscription stream — we must NOT
      // re-listen per sample.  Instead we keep one subscription alive and
      // route each incoming response to the current sample's Completer,
      // matched by the t1 nonce embedded in the request payload.
      Completer<ClockSyncSample>? activeCompleter;
      int? activeT1;

      subscription = socket.listen((event) {
        if (event != RawSocketEvent.read) return;

        final datagram = socket?.receive();
        if (datagram == null) return;

        try {
          final t4 = DateTime.now().microsecondsSinceEpoch;

          final json = jsonDecode(utf8.decode(datagram.data))
              as Map<String, dynamic>;
          final response = SyncMessage.fromJson(json);

          if (response.type != SyncMessageType.clockSyncResponse) return;
          if (response.senderDeviceId == localDeviceId) return;

          final t1InResponse = response.payload['t1'] as int?;
          if (t1InResponse == null || t1InResponse != activeT1) return;

          final t2 = response.payload['t2'] as int;
          final t3 = response.payload['t3'] as int;

          final completer = activeCompleter;
          if (completer != null && !completer.isCompleted) {
            completer.complete(
              ClockSyncSample(t1: activeT1!, t2: t2, t3: t3, t4: t4),
            );
          }
        } catch (e, stack) {
          SyncLogger.error(
            'Clock sync response parse error '
            '(${e.runtimeType}: $e)',
            e,
            stack,
          );
        }
      });

      // ── Collect samples ──────────────────────────────────────────────
      for (var i = 0; i < sampleCount; i++) {
        try {
          // T1: record *just before* sending to minimise measurement bias.
          final t1 = DateTime.now().microsecondsSinceEpoch;

          // Set up a fresh Completer for this sample, keyed by t1.
          activeT1 = t1;
          activeCompleter = Completer<ClockSyncSample>();

          final request = SyncMessage(
            type: SyncMessageType.clockSyncRequest,
            senderDeviceId: localDeviceId,
            timestampMicros: t1,
            payload: {'t1': t1},
          );

          final bytes = utf8.encode(jsonEncode(request.toJson()));
          final sent = socket.send(bytes, remoteAddress, port);

          if (sent <= 0) {
            SyncLogger.info(
              'Clock sync sample ${i + 1}/$sampleCount to ${remote.id}: '
              'send failed (sent=$sent) — skipping',
            );
            continue;
          }

          // Wait for the listener to complete this sample's Completer.
          final sample =
              await activeCompleter.future.timeout(timeout);

          samples.add(sample);
          SyncLogger.sync(
            'Clock sync sample ${i + 1}/$sampleCount to ${remote.id}: '
            '$sample',
          );
        } on TimeoutException {
          SyncLogger.info(
            'Clock sync sample ${i + 1}/$sampleCount to ${remote.id}: '
            'timed out — skipping',
          );
        } catch (e, stack) {
          // Diagnostic: log the ACTUAL exception type, not a generic string.
          SyncLogger.error(
            'Clock sync sample ${i + 1}/$sampleCount to ${remote.id}: '
            'failed (${e.runtimeType}: $e)',
            e,
            stack,
          );
        } finally {
          // Clear the active completer so stale responses are ignored.
          activeCompleter = null;
          activeT1 = null;
        }

        // Small gap between samples to avoid burst-induced jitter.
        if (i < sampleCount - 1) {
          await Future<void>.delayed(gap);
        }
      }
    } finally {
      await subscription?.cancel();
      socket?.close();
    }

    if (samples.length <= NetworkConstants.clockSyncDiscardCount) {
      SyncLogger.error(
        'Clock sync to ${remote.id} failed: only ${samples.length} '
        'samples collected, need > ${NetworkConstants.clockSyncDiscardCount}',
      );
      throw ClockSyncException(
        'Not enough samples for ${remote.id}: ${samples.length}',
      );
    }

    final offset = computeAverageOffset(samples);
    offsets[remote.id] = offset;

    SyncLogger.sync(
      'Clock sync to ${remote.id} complete: offset = $offset µs '
      '(${offset / 1000} ms) from ${samples.length} samples',
    );

    return offset;
  }

  /// Re-runs the clock sync handshake for [remote], replacing the cached
  /// offset.
  Future<int> resync(DeviceInfo remote) async {
    SyncLogger.info('Resync requested for ${remote.id}');
    return computeOffset(remote);
  }

  // ── Remote: auto-respond to clock sync requests ────────────────────────

  RawDatagramSocket? _responderSocket;

  /// Starts listening for `clock_sync_request` messages on the dedicated
  /// clock sync port and automatically replying with `clock_sync_response`.
  ///
  /// Call this on the **remote** device after discovery.
  Future<void> startResponder() async {
    try {
      _responderSocket = await RawDatagramSocket.bind(
        InternetAddress.anyIPv4,
        NetworkConstants.udpClockSyncPort,
        reuseAddress: true,
      );

      SyncLogger.info(
        'ClockSync responder listening on port '
        '${NetworkConstants.udpClockSyncPort}',
      );
    } catch (e, stack) {
      SyncLogger.error('ClockSync responder: failed to bind socket', e, stack);
      return;
    }

    _responderSocket!.listen(
      (event) {
        if (event != RawSocketEvent.read) return;
        _handleClockSyncRequest();
      },
      onError: (Object error, StackTrace stack) {
        SyncLogger.error('ClockSync responder error', error, stack);
      },
    );
  }

  void _handleClockSyncRequest() {
    final datagram = _responderSocket?.receive();
    if (datagram == null) return;

    try {
      final t2 = DateTime.now().microsecondsSinceEpoch;

      final json = jsonDecode(utf8.decode(datagram.data))
          as Map<String, dynamic>;
      final request = SyncMessage.fromJson(json);

      if (request.type != SyncMessageType.clockSyncRequest) return;
      if (request.senderDeviceId == localDeviceId) return;

      SyncLogger.sync('Remote: received clock_sync_request from ${request.senderDeviceId}');

      final t3 = DateTime.now().microsecondsSinceEpoch;

      final response = SyncMessage(
        type: SyncMessageType.clockSyncResponse,
        senderDeviceId: localDeviceId,
        timestampMicros: t3,
        payload: {
          't1': request.payload['t1'],
          't2': t2,
          't3': t3,
        },
      );

      final bytes = utf8.encode(jsonEncode(response.toJson()));
      _responderSocket!.send(
        bytes,
        datagram.address,
        datagram.port,
      );
      SyncLogger.sync('Remote: sent clock_sync_response to ${request.senderDeviceId}');
    } catch (e, stack) {
      SyncLogger.error(
        'ClockSync responder: ignoring malformed packet from '
        '${datagram.address.address}:${datagram.port}',
        e,
        stack,
      );
    }
  }

  /// Stops the clock sync responder.
  void stopResponder() {
    _responderSocket?.close();
    _responderSocket = null;
    SyncLogger.info('ClockSync responder stopped');
  }

  /// Releases all resources.
  void dispose() {
    stopResponder();
    offsets.clear();
  }
}

/// Thrown when clock sync fails (e.g. not enough samples).
class ClockSyncException implements Exception {
  final String message;
  const ClockSyncException(this.message);

  @override
  String toString() => 'ClockSyncException: $message';
}
